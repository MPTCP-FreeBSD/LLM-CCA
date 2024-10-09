#维度不匹配
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque

INF = 1e5

class OfflineRLPolicy(nn.Module):
    def __init__(
            self,
            state_feature_dim,
            bitrate_levels,
            state_encoder,  # 用于处理 [RTT, CWND, Throughput]
            plm,
            plm_embed_size,
            max_length=None,
            max_ep_len=100,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            device_out=None,
            residual=False,
            which_layer=-1,  # 早停
            **kwargs
    ):
        super().__init__()
        
        if device_out is None:
            device_out = device

        self.bitrate_levels = bitrate_levels
        self.max_length = max_length

        self.plm = plm
        self.plm_embed_size = plm_embed_size

        # =========== Multimodal Encoder (start) ===========
        self.state_encoder = state_encoder
        self.state_feature_dim = state_feature_dim
        self.embed_timestep = nn.Embedding(max_ep_len + 1, plm_embed_size).to(device)
        self.embed_return = nn.Linear(1, plm_embed_size).to(device)
        self.embed_action = nn.Linear(1, plm_embed_size).to(device)
        self.embed_state1 = nn.Linear(state_feature_dim, plm_embed_size).to(device)  # 状态编码器

        self.embed_ln = nn.LayerNorm(plm_embed_size).to(device)
        # =========== Multimodal Encoder (end) ===========
    
        # 将 action_head 设计为输出连续的 cwnd 值
        self.action_head = nn.Linear(plm_embed_size, 1).to(device)

        self.device = device
        self.device_out = device_out

        # 用于评估时的状态队列
        self.states_dq = deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length)
        self.returns_dq = deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length)
        self.actions_dq = deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length)

        self.residual = residual
        self.which_layer = which_layer  # 提供用于早停的 layer 参数
        self.modules_except_plm = nn.ModuleList([
            self.state_encoder, self.embed_timestep, self.embed_return, self.embed_action, self.embed_ln,
            self.embed_state1, self.action_head
        ])

    def forward(self, states, actions, returns, timesteps, attention_mask=None):
        # 确保 actions 和 states 是 float 类型
        actions = actions.to(self.device).float()
        returns = returns.to(self.device).float()
        timesteps = timesteps.to(self.device)

        # 嵌入 actions, returns 和 timesteps
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns)
        time_embeddings = self.embed_timestep(timesteps)

        action_embeddings += time_embeddings
        returns_embeddings += time_embeddings

        # 处理 states 并将其转化为嵌入
        states = states.to(self.device).float()
        states_features = self.state_encoder(states)
        states_embeddings1 = self.embed_state1(states_features) + time_embeddings

        # 拼接 returns, states, actions 嵌入
        stacked_inputs = []
        action_embed_positions = np.zeros(returns_embeddings.shape[1])
        for i in range(returns_embeddings.shape[1]):
            stacked_input = torch.cat((returns_embeddings[0, i:i + 1], states_embeddings1[0, i:i + 1], 
                                    action_embeddings[0, i:i + 1]), dim=0)
            stacked_inputs.append(stacked_input)
            action_embed_positions[i] = (i + 1) * (2 + 1)

        stacked_inputs = torch.cat(stacked_inputs, dim=0).unsqueeze(0)
        stacked_inputs = stacked_inputs[:, -self.plm_embed_size:, :]
        stacked_inputs_ln = self.embed_ln(stacked_inputs)

        if attention_mask is None:
            attention_mask = torch.ones((stacked_inputs_ln.shape[0], stacked_inputs_ln.shape[1]), dtype=torch.long, device=self.device)

        transformer_outputs = self.plm(
            inputs_embeds=stacked_inputs_ln,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        logits = transformer_outputs['last_hidden_state']
        if self.residual:
            logits = logits + stacked_inputs_ln

        logits_used = logits[:, action_embed_positions - 2]
        action_pred = self.action_head(logits_used)

        # 这里进行 squeeze，确保 action_pred 的形状变为 (1, 20)
        action_pred = action_pred.squeeze(-1)

        return action_pred


    

    def sample(self, state, target_return, timestep, **kwargs):
        """
        用于评估或测试的采样函数。
        """
        prev_stacked_inputs = []
        for i in range(len(self.states_dq)):
            prev_return_embeddings = self.returns_dq[i]
            prev_state_embeddings = self.states_dq[i]
            prev_action_embeddings = self.actions_dq[i]
            prev_stacked_inputs.append(torch.cat((prev_return_embeddings, prev_state_embeddings, prev_action_embeddings), dim=1))
        prev_stacked_inputs = torch.cat(prev_stacked_inputs, dim=1)

        target_return = torch.as_tensor(target_return, dtype=torch.float32, device=self.device).reshape(1, 1, 1)
        timestep = torch.as_tensor(timestep, dtype=torch.int32, device=self.device).reshape(1, 1)

        return_embeddings = self.embed_return(target_return)
        time_embeddings = self.embed_timestep(timestep)

        return_embeddings = return_embeddings + time_embeddings

        # 处理状态
        state = state.to(self.device)
        state_features = self.state_encoder(state)
        state_embeddings1 = self.embed_state1(state_features) + time_embeddings
        state_embeddings = state_embeddings1

        # 调试信息：打印嵌入层输出的形状
        #print("return_embeddings shape:", return_embeddings.shape)  # 输出形状应为 (batch_size, seq_len, embed_dim)
        #print("state_embeddings shape:", state_embeddings.shape)    # 输出形状应为 (batch_size, seq_len, feature_num, embed_dim)

        # 如果 state_embeddings 是 4D，调整为 3D
        if state_embeddings.dim() == 4:
            # 方法1: 使用 mean 来合并第三维度
            state_embeddings = state_embeddings.mean(dim=2)

            # 或者使用 view/reshape 展平第三维度
            # state_embeddings = state_embeddings.view(state_embeddings.size(0), state_embeddings.size(1), -1)

        # 再次打印以确认调整后的维度
        #print("Adjusted state_embeddings shape:", state_embeddings.shape)

        # 拼接 return_embeddings 和 state_embeddings
        stacked_inputs = torch.cat((return_embeddings, state_embeddings), dim=1)
        stacked_inputs = torch.cat((prev_stacked_inputs, stacked_inputs), dim=1)
        stacked_inputs = stacked_inputs[:, -self.plm_embed_size:, :]
        stacked_inputs_ln = self.embed_ln(stacked_inputs)

        attention_mask = torch.ones((stacked_inputs_ln.shape[0], stacked_inputs_ln.shape[1]), dtype=torch.long, device=self.device)

        transformer_outputs = self.plm(
            inputs_embeds=stacked_inputs_ln,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        logits = transformer_outputs['last_hidden_state']
        if self.residual:
            logits = logits + stacked_inputs_ln

        logits_used = logits[:, -1:]
        action_pred = self.action_head(logits_used)
        action_pred = action_pred.reshape(-1)
        bitrate, _ = self._sample(action_pred)

        action_tensor = torch.zeros(1, 1, 1, dtype=torch.float32, device=self.device)
        action_tensor[..., 0] = (bitrate + 1) / self.bitrate_levels
        action_embeddings = self.embed_action(action_tensor)

        self.returns_dq.append(return_embeddings)
        self.states_dq.append(state_embeddings) 
        self.actions_dq.append(action_embeddings)

        return bitrate

    
    def clear_dq(self):
        self.states_dq.clear()
        self.actions_dq.clear()
        self.returns_dq.clear()
        
        self.states_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))
        self.actions_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))
        self.returns_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))

    def _sample(self, logits):
        pi = F.softmax(logits, 0).cpu().numpy()
        idx = random.choices(np.arange(pi.size), pi)[0]
        lgprob = np.log(pi[idx])
        return idx, lgprob

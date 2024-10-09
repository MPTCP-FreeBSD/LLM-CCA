# import argparse
# import pandas as pd
# import numpy as np
# import os
# import pickle
# import torch  # 用于创建 Tensor

# class ExperiencePool:
#     def __init__(self):
#         self.states = []
#         self.actions = []
#         self.rewards = []
#         self.dones = []

#     def add(self, state, action, reward, done):
#         self.states.append(state)
#         self.actions.append(action)
#         self.rewards.append(reward)
#         self.dones.append(done)

# def calculate_reward(rtt, throughput):
#     # 确保RTT和吞吐量非零，避免除零错误
#     if rtt <= 0 or throughput <= 0:
#         return 0.1  # 当RTT或吞吐量为零时返回最小奖励值

#     # 使用吞吐量/RTT作为度量标准
#     performance_ratio = throughput / rtt

#     # 使用对数缩放来平滑奖励值
#     reward = np.log10(performance_ratio)

#     return reward

# def run(args):
#     df = pd.read_csv(args.csv_file)
#     exp_pool = ExperiencePool()

#     for index, row in df.iterrows():
#         rtt = float(row['rtt']) if pd.notna(row['rtt']) else 0
#         cwnd = float(row['cwnd']) if pd.notna(row['cwnd']) else 0
#         throughput = float(row['throughput']) if pd.notna(row['throughput']) else 0

#         # 将 state 转换为 Tensor，只包含 [rtt, cwnd, throughput]
#         state = torch.tensor([rtt, cwnd, throughput], dtype=torch.float32)

#         # 动作定义为 cwnd
#         action = cwnd

#         # 计算 reward
#         reward = calculate_reward(rtt, throughput)
#         done = index == len(df) - 1

#         # 将 state, action, reward, done 加入经验池
#         exp_pool.add(state, action, reward, done)

#     # 保存 ExperiencePool 实例
#     exp_pool_path = os.path.join(args.output_dir, 'prague_exp_pool.pkl')

#     with open(exp_pool_path, 'wb') as f:
#         pickle.dump(exp_pool, f)

#     print(f"经验池已保存至: {exp_pool_path}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='根据 prague 实验数据生成经验池')
#     parser.add_argument('--csv_file', type=str, default=r'C:\Users\10111\Desktop\SIT724\Runpod\experiment data\通过脚本或取得prague和cubic的数据\prague cwnd script\第三次实验\只跑prague\only_prague_output_rtt_cwnd_throughput.csv', help='CSV文件路径')
#     parser.add_argument('--output_dir', type=str, default=r'C:\Users\10111\Desktop\SIT724\Runpod\experiment data\通过脚本或取得prague和cubic的数据\prague cwnd script\第三次实验\只跑prague', help='生成经验池输出目录')

#     args = parser.parse_args()
#     run(args)


import argparse
import pandas as pd
import numpy as np
import os
import pickle
import torch  # 用于创建 Tensor

class ExperiencePool:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def add(self, state, action, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

def calculate_reward(rtt, throughput):
    # 确保RTT和吞吐量非零，避免除零错误
    if rtt <= 0 or throughput <= 0:
        return 0.1  # 当RTT或吞吐量为零时返回最小奖励值

    # 使用吞吐量/RTT作为度量标准
    performance_ratio = throughput / rtt

    # 使用对数缩放来平滑奖励值
    reward = np.log10(performance_ratio)

    return reward

def run(args):
    df = pd.read_csv(args.csv_file)
    exp_pool = ExperiencePool()

    # 找到最大和最小的 cwnd 值
    max_cwnd_value = df['cwnd'].max()
    min_cwnd_value = df['cwnd'].min()

    for index, row in df.iterrows():
        rtt = float(row['rtt']) if pd.notna(row['rtt']) else 0
        cwnd = float(row['cwnd']) if pd.notna(row['cwnd']) else 0
        throughput = float(row['throughput']) if pd.notna(row['throughput']) else 0

        # 将 state 转换为 Tensor，只包含 [rtt, cwnd, throughput]
        state = torch.tensor([rtt, cwnd, throughput], dtype=torch.float32)

        # 将 cwnd 缩放到 [1, 2, 3] 范围
        # 先进行归一化，再映射到 [1, 2, 3] 范围
        action_normalized = (cwnd - min_cwnd_value) / (max_cwnd_value - min_cwnd_value)
        action = int(action_normalized * 2) + 1  # 映射到 1, 2, 3

        # 计算 reward
        reward = calculate_reward(rtt, throughput)
        done = index == len(df) - 1

        # 将 state, action, reward, done 加入经验池
        exp_pool.add(state, action, reward, done)

    # 保存 ExperiencePool 实例
    exp_pool_path = os.path.join(args.output_dir, 'prague_exp_pool.pkl')

    with open(exp_pool_path, 'wb') as f:
        pickle.dump(exp_pool, f)

    print(f"经验池已保存至: {exp_pool_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='根据 prague 实验数据生成经验池')
    parser.add_argument('--csv_file', type=str, default=r'C:\Users\10111\Desktop\SIT724\Runpod\experiment data\通过脚本或取得prague和cubic的数据\prague cwnd script\第三次实验\只跑prague\only_prague_output_rtt_cwnd_throughput.csv', help='CSV文件路径')
    parser.add_argument('--output_dir', type=str, default=r'C:\Users\10111\Desktop\SIT724\Runpod\experiment data\通过脚本或取得prague和cubic的数据\prague cwnd script\第三次实验\只跑prague', help='生成经验池输出目录')

    args = parser.parse_args()
    run(args)

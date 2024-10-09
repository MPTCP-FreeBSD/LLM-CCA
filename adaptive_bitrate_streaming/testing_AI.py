# import torch
# import pickle
# import numpy as np
# from sklearn.metrics import mean_squared_error

# # 假设你有一个验证数据集 val_data.pkl
# def load_model(model_path):
#     # 加载你训练好的模型
#     model = torch.load(model_path)
#     model.eval()  # 设置为评估模式
#     return model

# def load_validation_data(validation_data_path):
#     with open(validation_data_path, 'rb') as f:
#         validation_data = pickle.load(f)
#     return validation_data

# # 添加逆归一化函数
# def inverse_normalize_action(action_normalized, min_cwnd_value, max_cwnd_value):
#     # 逆归一化公式
#     return action_normalized * (max_cwnd_value - min_cwnd_value) + min_cwnd_value

# def validate_model(model, validation_data, min_cwnd_value, max_cwnd_value):
#     predictions = []
#     ground_truth = []

#     for data in validation_data:
#         state = data['state']  # 提取状态 (cwnd, rtt, throughput)
#         true_action_normalized = data['action']  # 提取归一化的真实 action (cwnd)

#         # 将 state 转换为模型输入的格式
#         state_tensor = torch.tensor(state).float().unsqueeze(0)  # 适配模型输入
        
#         # 使用模型进行预测
#         predicted_action_normalized = model(state_tensor).item()

#         # 将预测值和真实值逆归一化
#         predicted_action = inverse_normalize_action(predicted_action_normalized, min_cwnd_value, max_cwnd_value)
#         true_action = inverse_normalize_action(true_action_normalized, min_cwnd_value, max_cwnd_value)

#         predictions.append(predicted_action)
#         ground_truth.append(true_action)

#     # 计算均方误差 (MSE)
#     mse = mean_squared_error(ground_truth, predictions)
#     print(f"Validation MSE: {mse}")

#     return predictions, ground_truth

# def visualize_results(predictions, ground_truth):
#     import matplotlib.pyplot as plt
#     plt.plot(predictions, label='Predicted cwnd')
#     plt.plot(ground_truth, label='True cwnd')
#     plt.legend()
#     plt.title("Predicted vs True cwnd")
#     plt.show()

# if __name__ == "__main__":
#     # 加载模型和验证数据集
#     model = load_model("path_to_your_model.pth")
#     validation_data = load_validation_data("path_to_your_validation_data.pkl")

#     # 这里需要传入验证数据的 min_cwnd_value 和 max_cwnd_value，确保与训练数据保持一致
#     min_cwnd_value = 159280  # 替换为实际最小 cwnd 值
#     max_cwnd_value = 786264  # 替换为实际最大 cwnd 值

#     # 运行验证
#     predictions, ground_truth = validate_model(model, validation_data, min_cwnd_value, max_cwnd_value)

#     # 可视化结果
#     visualize_results(predictions, ground_truth)


# import torch
# import pickle
# import numpy as np
# from sklearn.metrics import mean_squared_error

# # 加载模型
# def load_model(adapter_model_path, modules_except_plm_path, plm_model_path):
#     # 加载基础模型 (PLM) 并加载 adapter 和模块参数
#     model = torch.load(plm_model_path)  # 加载基础模型
#     model.plm.load_adapter(adapter_model_path, adapter_name='default')  # 加载 adapter 权重
#     model.modules_except_plm.load_state_dict(torch.load(modules_except_plm_path))  # 加载除 PLM 之外的模块权重
#     model.eval()  # 设置为评估模式
#     return model

# # 加载验证数据
# def load_validation_data(validation_data_path):
#     with open(validation_data_path, 'rb') as f:
#         validation_data = pickle.load(f)
#     return validation_data

# # 逆归一化函数
# def inverse_normalize_action(action_normalized, min_cwnd_value, max_cwnd_value):
#     return action_normalized * (max_cwnd_value - min_cwnd_value) + min_cwnd_value

# # 验证模型
# def validate_model(model, validation_data, min_cwnd_value, max_cwnd_value):
#     predictions = []
#     ground_truth = []

#     for data in validation_data:
#         state = data['state']  # 提取状态 (cwnd, rtt, throughput)
#         true_action_normalized = data['action']  # 提取归一化的真实 action (cwnd)

#         # 将 state 转换为模型输入的格式
#         state_tensor = torch.tensor(state).float().unsqueeze(0)  # 适配模型输入
        
#         # 使用模型进行预测
#         predicted_action_normalized = model(state_tensor).item()

#         # 逆归一化预测值和真实值
#         predicted_action = inverse_normalize_action(predicted_action_normalized, min_cwnd_value, max_cwnd_value)
#         true_action = inverse_normalize_action(true_action_normalized, min_cwnd_value, max_cwnd_value)

#         predictions.append(predicted_action)
#         ground_truth.append(true_action)

#     # 计算均方误差 (MSE)
#     mse = mean_squared_error(ground_truth, predictions)
#     print(f"Validation MSE: {mse}")

#     return predictions, ground_truth

# # 可视化结果
# def visualize_results(predictions, ground_truth):
#     import matplotlib.pyplot as plt
#     plt.plot(predictions, label='Predicted cwnd')
#     plt.plot(ground_truth, label='True cwnd')
#     plt.legend()
#     plt.title("Predicted vs True cwnd")
#     plt.show()

# if __name__ == "__main__":
#     # 加载模型和验证数据集
#     adapter_model_path = "path_to_adapter_model.bin"
#     modules_except_plm_path = "path_to_modules_except_plm.bin"
#     plm_model_path = "path_to_plm_model"  # 例如 llama 或 gpt2 的基础模型路径
    
#     model = load_model(adapter_model_path, modules_except_plm_path, plm_model_path)
#     validation_data = load_validation_data("path_to_your_validation_data.pkl")

#     # 这里需要传入验证数据的 min_cwnd_value 和 max_cwnd_value，确保与训练数据保持一致
#     min_cwnd_value = 159280  # 替换为实际最小 cwnd 值
#     max_cwnd_value = 786264  # 替换为实际最大 cwnd 值

#     # 运行验证
#     predictions, ground_truth = validate_model(model, validation_data, min_cwnd_value, max_cwnd_value)

#     # 可视化结果
#     visualize_results(predictions, ground_truth)



#nengpaole
# import os
# import torch
# import pickle
# import numpy as np
# from sklearn.metrics import mean_squared_error

# # 添加加载函数
# def load_model(adapter_model_path, plm_model_dir, modules_except_plm_path):
#     # 1. 加载 PLM 模型 (分片加载)
#     # PyTorch 会自动找到并加载多个分片文件
#     plm_model = torch.load(os.path.join(plm_model_dir, 'pytorch_model-00001-of-00002.bin'))
    
#     print("PLM model loaded successfully from split files.")

#     # 2. 加载 adapter_model
#     adapter_model = torch.load(adapter_model_path)

#     # 3. 加载 modules_except_plm
#     modules_except_plm = torch.load(modules_except_plm_path)

#     # 确保所有模型都处于评估模式
#     plm_model.eval()
#     adapter_model.eval()
#     modules_except_plm.eval()
    
#     return plm_model, adapter_model, modules_except_plm

# # 加载验证数据
# def load_validation_data(validation_data_path):
#     with open(validation_data_path, 'rb') as f:
#         validation_data = pickle.load(f)
#     return validation_data

# # 添加逆归一化函数
# def inverse_normalize_action(action_normalized, min_cwnd_value, max_cwnd_value):
#     # 逆归一化公式
#     return action_normalized * (max_cwnd_value - min_cwnd_value) + min_cwnd_value

# # 验证模型
# def validate_model(plm_model, adapter_model, modules_except_plm, validation_data, min_cwnd_value, max_cwnd_value):
#     predictions = []
#     ground_truth = []

#     for state, true_action_normalized in zip(validation_data.states, validation_data.actions):
#         # 提取状态 (cwnd, rtt, throughput)
#         # 提取归一化的真实 action (cwnd)
#         state_tensor = torch.tensor(state).float().unsqueeze(0)  # 适配模型输入

#         # 使用 PLM 模型进行预测
#         predicted_action_normalized = plm_model(state_tensor).item()

#         # 将预测值和真实值逆归一化
#         predicted_action = inverse_normalize_action(predicted_action_normalized, min_cwnd_value, max_cwnd_value)
#         true_action = inverse_normalize_action(true_action_normalized, min_cwnd_value, max_cwnd_value)

#         predictions.append(predicted_action)
#         ground_truth.append(true_action)

#     # 计算均方误差 (MSE)
#     mse = mean_squared_error(ground_truth, predictions)
#     print(f"Validation MSE: {mse}")

#     return predictions, ground_truth

# # 可视化结果
# def visualize_results(predictions, ground_truth):
#     import matplotlib.pyplot as plt
#     plt.plot(predictions, label='Predicted cwnd')
#     plt.plot(ground_truth, label='True cwnd')
#     plt.legend()
#     plt.title("Predicted vs True cwnd")
#     plt.show()

# if __name__ == "__main__":
#     # 设置文件路径
#     # adapter_model_path = "D:/AI_copy2.0/训练过的NetLLM存档/只跑prague 的/adaptive_bitrate_streaming/data/ft_plms/llama_base/._ss_None/rank_128_w_20_gamma_1.0_sfd_256_lr_0.0001_wd_0.0001_warm_2000_epochs_60_seed_100003/early_stop_-1_best_model/adapter_model.bin"
#     # plm_model_path = "D:/AI_copy2.0/训练过的NetLLM存档/测试文件/downloaded_plms/llama/base/pytorch_model.bin"  # 替换为实际的文件名
#     # modules_except_plm_path = "D:/AI_copy2.0/训练过的NetLLM存档/只跑prague 的/adaptive_bitrate_streaming/data/ft_plms/llama_base/._ss_None/rank_128_w_20_gamma_1.0_sfd_256_lr_0.0001_wd_0.0001_warm_2000_epochs_60_seed_100003/early_stop_-1_best_model/modules_except_plm.bin"
#     # 替换为实际路径
#     # 设置文件路径
#     plm_model_dir = "D:/AI_copy2.0/训练过的NetLLM存档/测试文件/downloaded_plms/llama/base/"
#     adapter_model_path = "D:/AI_copy2.0/训练过的NetLLM存档/只跑prague 的/adaptive_bitrate_streaming/data/ft_plms/llama_base/._ss_None/rank_128_w_20_gamma_1.0_sfd_256_lr_0.0001_wd_0.0001_warm_2000_epochs_60_seed_100003/early_stop_-1_best_model/adapter_model.bin"
#     modules_except_plm_path = "D:/AI_copy2.0/训练过的NetLLM存档/只跑prague 的/adaptive_bitrate_streaming/data/ft_plms/llama_base/._ss_None/rank_128_w_20_gamma_1.0_sfd_256_lr_0.0001_wd_0.0001_warm_2000_epochs_60_seed_100003/early_stop_-1_best_model/modules_except_plm.bin"




#     # 加载模型
#     plm_model, adapter_model, modules_except_plm = load_model(adapter_model_path, plm_model_dir, modules_except_plm_path)

#     # 加载验证数据
#     validation_data_path = "./prague_exp_pool.pkl"  # 使用你的训练 PKL 文件
#     validation_data = load_validation_data(validation_data_path)

#     # 这里需要传入验证数据的 min_cwnd_value 和 max_cwnd_value，确保与训练数据保持一致
#     min_cwnd_value = 159280  # 替换为实际最小 cwnd 值
#     max_cwnd_value = 786264  # 替换为实际最大 cwnd 值

#     # 运行验证
#     predictions, ground_truth = validate_model(plm_model, adapter_model, modules_except_plm, validation_data, min_cwnd_value, max_cwnd_value)

#     # 可视化结果
#     visualize_results(predictions, ground_truth)

#能跑了，需要上传到runpod，本地跑不了用的CPU
# import os
# import torch
# import pickle
# import numpy as np
# from transformers import AutoModelForCausalLM
# from sklearn.metrics import mean_squared_error

# # 初始化 PLM 模型（如 LLaMA）的类
# def load_plm_model(plm_model_dir):
#     # 假设使用 transformers 库来加载 LLaMA
#     config_path = os.path.join(plm_model_dir, "config.json")  # 确保有配置文件
#     plm_model = AutoModelForCausalLM.from_pretrained(plm_model_dir)
#     return plm_model

# # 加载 adapter 和 modules_except_plm 的权重
# def load_adapter_and_modules(adapter_model_path, modules_except_plm_path):
#     # adapter 和其他模块权重
#     adapter_model = torch.load(adapter_model_path, map_location=torch.device('cpu'))
#     modules_except_plm = torch.load(modules_except_plm_path, map_location=torch.device('cpu'))
#     return adapter_model, modules_except_plm

# # 加载所有模型
# def load_model(adapter_model_path, plm_model_dir, modules_except_plm_path):
#     # 1. 加载 PLM 模型 (LLaMA)
#     plm_model = load_plm_model(plm_model_dir)
#     print("PLM model loaded successfully from split files.")

#     # 2. 加载 adapter_model 和 modules_except_plm
#     adapter_model, modules_except_plm = load_adapter_and_modules(adapter_model_path, modules_except_plm_path)

#     # 确保所有模型都处于评估模式
#     plm_model.eval()
#     # 注意：如果 adapter_model 和 modules_except_plm 是 dict，可能需要手动将其应用到模型架构中

#     return plm_model, adapter_model, modules_except_plm

# # 加载验证数据
# def load_validation_data(validation_data_path):
#     with open(validation_data_path, 'rb') as f:
#         validation_data = pickle.load(f)
#     return validation_data

# # 添加逆归一化函数
# def inverse_normalize_action(action_normalized, min_cwnd_value, max_cwnd_value):
#     # 逆归一化公式
#     return action_normalized * (max_cwnd_value - min_cwnd_value) + min_cwnd_value

# # 验证模型
# def validate_model(plm_model, adapter_model, modules_except_plm, validation_data, min_cwnd_value, max_cwnd_value):
#     predictions = []
#     ground_truth = []

#     for state, true_action_normalized in zip(validation_data.states, validation_data.actions):
#         # 提取状态 (cwnd, rtt, throughput)
#         state_tensor = torch.tensor(state).float().unsqueeze(0)  # 适配模型输入

#         # 使用 PLM 模型进行预测
#         with torch.no_grad():
#             predicted_action_normalized = plm_model(state_tensor).item()

#         # 将预测值和真实值逆归一化
#         predicted_action = inverse_normalize_action(predicted_action_normalized, min_cwnd_value, max_cwnd_value)
#         true_action = inverse_normalize_action(true_action_normalized, min_cwnd_value, max_cwnd_value)

#         predictions.append(predicted_action)
#         ground_truth.append(true_action)

#     # 计算均方误差 (MSE)
#     mse = mean_squared_error(ground_truth, predictions)
#     print(f"Validation MSE: {mse}")

#     return predictions, ground_truth

# # 可视化结果
# def visualize_results(predictions, ground_truth):
#     import matplotlib.pyplot as plt
#     plt.plot(predictions, label='Predicted cwnd')
#     plt.plot(ground_truth, label='True cwnd')
#     plt.legend()
#     plt.title("Predicted vs True cwnd")
#     plt.show()

# if __name__ == "__main__":
#     # 设置文件路径
#     plm_model_dir = "D:/AI_copy2.0/训练过的NetLLM存档/测试文件/downloaded_plms/llama/base/"
#     adapter_model_path = "D:/AI_copy2.0/训练过的NetLLM存档/只跑prague的/adaptive_bitrate_streaming/data/ft_plms/llama_base/._ss_None/rank_128_w_20_gamma_1.0_sfd_256_lr_0.0001_wd_0.0001_warm_2000_epochs_60_seed_100003/early_stop_-1_best_model/adapter_model.bin"
#     modules_except_plm_path = "D:/AI_copy2.0/训练过的NetLLM存档/只跑prague的/adaptive_bitrate_streaming/data/ft_plms/llama_base/._ss_None/rank_128_w_20_gamma_1.0_sfd_256_lr_0.0001_wd_0.0001_warm_2000_epochs_60_seed_100003/early_stop_-1_best_model/modules_except_plm.bin"

#     # 加载模型
#     plm_model, adapter_model, modules_except_plm = load_model(adapter_model_path, plm_model_dir, modules_except_plm_path)

#     # 加载验证数据
#     validation_data_path = "./prague_exp_pool.pkl"  # 使用你的训练 PKL 文件
#     validation_data = load_validation_data(validation_data_path)

#     # 这里需要传入验证数据的 min_cwnd_value 和 max_cwnd_value，确保与训练数据保持一致
#     min_cwnd_value = 159280  # 替换为实际最小 cwnd 值
#     max_cwnd_value = 786264  # 替换为实际最大 cwnd 值

#     # 运行验证
#     predictions, ground_truth = validate_model(plm_model, adapter_model, modules_except_plm, validation_data, min_cwnd_value, max_cwnd_value)

#     # 可视化结果
#     visualize_results(predictions, ground_truth)


# import os
# import torch
# import pickle
# import numpy as np
# from transformers import AutoModelForCausalLM
# from sklearn.metrics import mean_squared_error

# # 初始化 PLM 模型（如 LLaMA）的类
# def load_plm_model(plm_model_dir, device):
#     plm_model = AutoModelForCausalLM.from_pretrained(plm_model_dir).to(device)

#     # 如果使用多块 GPU，自动分配
#     if torch.cuda.device_count() > 1:
#         print(f"Using {torch.cuda.device_count()} GPUs for PLM model.")
#         plm_model = torch.nn.DataParallel(plm_model)

#     return plm_model

# # 加载 adapter 和 modules_except_plm 的权重
# def load_adapter_and_modules(adapter_model_path, modules_except_plm_path, device):
#     adapter_model = torch.load(adapter_model_path, map_location=device)
#     modules_except_plm = torch.load(modules_except_plm_path, map_location=device)

#     # 使用多块 GPU
#     if torch.cuda.device_count() > 1:
#         print(f"Using {torch.cuda.device_count()} GPUs for adapter and modules_except_plm.")
#         adapter_model = torch.nn.DataParallel(adapter_model)
#         modules_except_plm = torch.nn.DataParallel(modules_except_plm)

#     return adapter_model, modules_except_plm

# # 加载所有模型
# def load_model(adapter_model_path, plm_model_dir, modules_except_plm_path, device):
#     plm_model = load_plm_model(plm_model_dir, device)
#     print("PLM model loaded successfully from split files.")

#     adapter_model, modules_except_plm = load_adapter_and_modules(adapter_model_path, modules_except_plm_path, device)

#     plm_model.eval()
#     adapter_model.eval()
#     modules_except_plm.eval()

#     return plm_model, adapter_model, modules_except_plm

# # 加载验证数据
# def load_validation_data(validation_data_path):
#     with open(validation_data_path, 'rb') as f:
#         validation_data = pickle.load(f)
#     return validation_data

# # 逆归一化函数
# def inverse_normalize_action(action_normalized, min_cwnd_value, max_cwnd_value):
#     return action_normalized * (max_cwnd_value - min_cwnd_value) + min_cwnd_value

# # 验证模型
# def validate_model(plm_model, adapter_model, modules_except_plm, validation_data, min_cwnd_value, max_cwnd_value, device):
#     predictions = []
#     ground_truth = []

#     for state, true_action_normalized in zip(validation_data.states, validation_data.actions):
#         state_tensor = torch.tensor(state).float().unsqueeze(0).to(device)

#         with torch.no_grad():
#             predicted_action_normalized = plm_model(state_tensor).item()

#         predicted_action = inverse_normalize_action(predicted_action_normalized, min_cwnd_value, max_cwnd_value)
#         true_action = inverse_normalize_action(true_action_normalized, min_cwnd_value, max_cwnd_value)

#         predictions.append(predicted_action)
#         ground_truth.append(true_action)

#     mse = mean_squared_error(ground_truth, predictions)
#     print(f"Validation MSE: {mse}")

#     return predictions, ground_truth

# # 修改后的可视化函数，打印预测值和真实值
# def visualize_results(predictions, ground_truth):
#     print("Predictions:", predictions)
#     print("Ground Truth:", ground_truth)

#     # 如果运行环境支持图像生成
#     try:
#         import matplotlib.pyplot as plt
#         plt.plot(predictions, label='Predicted cwnd')
#         plt.plot(ground_truth, label='True cwnd')
#         plt.legend()
#         plt.title("Predicted vs True cwnd")
#         plt.savefig("cwnd_comparison.png")
#         print("Image saved as 'cwnd_comparison.png'.")
#     except ImportError:
#         print("matplotlib not available, only printing data.")

# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # 文件路径
#     plm_model_dir = "/workspace/NetLLM/downloaded_plms/llama/base/"
#     adapter_model_path = "/workspace/NetLLM/adaptive_bitrate_streaming/data/ft_plms/llama_base/._ss_None/rank_128_w_20_gamma_1.0_sfd_256_lr_0.0001_wd_0.0001_warm_2000_epochs_60_seed_100003/early_stop_-1_best_model/adapter_model.bin"
#     modules_except_plm_path = "/workspace/NetLLM/adaptive_bitrate_streaming/data/ft_plms/llama_base/._ss_None/rank_128_w_20_gamma_1.0_sfd_256_lr_0.0001_wd_0.0001_warm_2000_epochs_60_seed_100003/early_stop_-1_best_model/modules_except_plm.bin"

#     # 加载模型
#     plm_model, adapter_model, modules_except_plm = load_model(adapter_model_path, plm_model_dir, modules_except_plm_path, device)

#     # 加载验证数据
#     validation_data_path = "./prague_exp_pool.pkl"
#     validation_data = load_validation_data(validation_data_path)

#     min_cwnd_value = 159280
#     max_cwnd_value = 786264

#     # 运行验证
#     predictions, ground_truth = validate_model(plm_model, adapter_model, modules_except_plm, validation_data, min_cwnd_value, max_cwnd_value, device)

#     # 可视化结果
#     visualize_results(predictions, ground_truth)


# import os
# import torch
# import pickle
# import numpy as np
# from transformers import AutoModelForCausalLM
# from sklearn.metrics import mean_squared_error

# # 定义 ExperiencePool 类
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

#     def __len__(self):
#         return len(self.states)

# # 初始化 PLM 模型（如 LLaMA）的类
# def load_plm_model(plm_model_dir, device):
#     config_path = os.path.join(plm_model_dir, "config.json")
#     plm_model = AutoModelForCausalLM.from_pretrained(plm_model_dir).to(device)

#     if torch.cuda.device_count() > 1:
#         print(f"Using {torch.cuda.device_count()} GPUs for PLM model.")
#         plm_model = torch.nn.DataParallel(plm_model)
    
#     return plm_model

# # 加载 adapter 和 modules_except_plm 的权重
# def load_adapter_and_modules(adapter_model_path, modules_except_plm_path, device):
#     adapter_model = torch.load(adapter_model_path, map_location=device)
#     modules_except_plm = torch.load(modules_except_plm_path, map_location=device)

#     if torch.cuda.device_count() > 1:
#         print(f"Using {torch.cuda.device_count()} GPUs for adapter and modules_except_plm.")
#         adapter_model = torch.nn.DataParallel(adapter_model)
#         modules_except_plm = torch.nn.DataParallel(modules_except_plm)

#     return adapter_model, modules_except_plm

# # 加载所有模型
# def load_model(adapter_model_path, plm_model_dir, modules_except_plm_path, device):
#     plm_model = load_plm_model(plm_model_dir, device)
#     print("PLM model loaded successfully from split files.")

#     adapter_model, modules_except_plm = load_adapter_and_modules(adapter_model_path, modules_except_plm_path, device)

#     plm_model.eval()
#     adapter_model.eval()
#     modules_except_plm.eval()

#     return plm_model, adapter_model, modules_except_plm

# # 加载验证数据
# def load_validation_data(validation_data_path):
#     with open(validation_data_path, 'rb') as f:
#         validation_data = pickle.load(f)
#     return validation_data

# # 添加逆归一化函数
# def inverse_normalize_action(action_normalized, min_cwnd_value, max_cwnd_value):
#     return action_normalized * (max_cwnd_value - min_cwnd_value) + min_cwnd_value

# # 验证模型
# def validate_model(plm_model, adapter_model, modules_except_plm, validation_data, min_cwnd_value, max_cwnd_value, device):
#     predictions = []
#     ground_truth = []

#     for state, true_action_normalized in zip(validation_data.states, validation_data.actions):
#         state_tensor = torch.tensor(state).float().unsqueeze(0).to(device)

#         with torch.no_grad():
#             predicted_action_normalized = plm_model(state_tensor).item()

#         predicted_action = inverse_normalize_action(predicted_action_normalized, min_cwnd_value, max_cwnd_value)
#         true_action = inverse_normalize_action(true_action_normalized, min_cwnd_value, max_cwnd_value)

#         predictions.append(predicted_action)
#         ground_truth.append(true_action)

#     mse = mean_squared_error(ground_truth, predictions)
#     print(f"Validation MSE: {mse}")

#     return predictions, ground_truth

# # 可视化结果
# def visualize_results(predictions, ground_truth):
#     import matplotlib.pyplot as plt
#     plt.plot(predictions, label='Predicted cwnd')
#     plt.plot(ground_truth, label='True cwnd')
#     plt.legend()
#     plt.title("Predicted vs True cwnd")
#     plt.show()

# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     plm_model_dir = "/workspace/NetLLM/downloaded_plms/llama/base/"
#     adapter_model_path = "/workspace/NetLLM/adaptive_bitrate_streaming/data/ft_plms/llama_base/._ss_None/rank_128_w_20_gamma_1.0_sfd_256_lr_0.0001_wd_0.0001_warm_2000_epochs_60_seed_100003/early_stop_-1_best_model/adapter_model.bin"
#     modules_except_plm_path = "/workspace/NetLLM/adaptive_bitrate_streaming/data/ft_plms/llama_base/._ss_None/rank_128_w_20_gamma_1.0_sfd_256_lr_0.0001_wd_0.0001_warm_2000_epochs_60_seed_100003/early_stop_-1_best_model/modules_except_plm.bin"

#     plm_model, adapter_model, modules_except_plm = load_model(adapter_model_path, plm_model_dir, modules_except_plm_path, device)

#     validation_data_path = "./prague_exp_pool.pkl"
#     validation_data = load_validation_data(validation_data_path)

#     min_cwnd_value = 159280
#     max_cwnd_value = 786264

#     predictions, ground_truth = validate_model(plm_model, adapter_model, modules_except_plm, validation_data, min_cwnd_value, max_cwnd_value, device)

#     visualize_results(predictions, ground_truth)



import os
import torch
import pickle
import numpy as np
from transformers import AutoModelForCausalLM
from sklearn.metrics import mean_squared_error

# 定义 ExperiencePool 类
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

    def __len__(self):
        return len(self.states)

# 初始化 PLM 模型（如 LLaMA）的类
def load_plm_model(plm_model_dir, device):
    plm_model = AutoModelForCausalLM.from_pretrained(plm_model_dir).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for PLM model.")
        plm_model = torch.nn.DataParallel(plm_model)

    return plm_model

# 加载 adapter 和 modules_except_plm 的权重
def load_adapter_and_modules(adapter_model_path, modules_except_plm_path, device):
    adapter_model = torch.load(adapter_model_path, map_location=device)
    modules_except_plm = torch.load(modules_except_plm_path, map_location=device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for adapter and modules_except_plm.")
        adapter_model = torch.nn.DataParallel(adapter_model)
        modules_except_plm = torch.nn.DataParallel(modules_except_plm)

    return adapter_model, modules_except_plm

# 加载所有模型
def load_model(adapter_model_path, plm_model_dir, modules_except_plm_path, device):
    plm_model = load_plm_model(plm_model_dir, device)
    print("PLM model loaded successfully from split files.")

    adapter_model, modules_except_plm = load_adapter_and_modules(adapter_model_path, modules_except_plm_path, device)

    plm_model.eval()
    adapter_model.eval()
    modules_except_plm.eval()

    return plm_model, adapter_model, modules_except_plm

# 加载验证数据
def load_validation_data(validation_data_path):
    with open(validation_data_path, 'rb') as f:
        validation_data = pickle.load(f)
    return validation_data

# 添加逆归一化函数
def inverse_normalize_action(action_normalized, min_cwnd_value, max_cwnd_value):
    return action_normalized * (max_cwnd_value - min_cwnd_value) + min_cwnd_value

# 验证模型
def validate_model(plm_model, adapter_model, modules_except_plm, validation_data, min_cwnd_value, max_cwnd_value, device):
    predictions = []
    ground_truth = []

    for state, true_action_normalized in zip(validation_data.states, validation_data.actions):
        # 检查输入是否为有效的浮点数类型，防止无效数据
        if torch.isnan(state).any() or torch.isinf(state).any():  # 检查输入状态是否无效
            print(f"Invalid input state detected: {state}, skipping this example.")
            continue

        # 将输入张量转换为整数类型
        state_tensor = torch.tensor(state).unsqueeze(0).to(device)

        try:
            # 强制将state_tensor转换为LongTensor类型用于embedding操作
            state_tensor = state_tensor.long()
            with torch.no_grad():
                predicted_action_normalized = plm_model(state_tensor).item()

            predicted_action = inverse_normalize_action(predicted_action_normalized, min_cwnd_value, max_cwnd_value)
            true_action = inverse_normalize_action(true_action_normalized, min_cwnd_value, max_cwnd_value)

            predictions.append(predicted_action)
            ground_truth.append(true_action)

        except RuntimeError as e:
            # 捕捉推理阶段的 CUDA 错误
            print(f"Error during prediction: {e}, skipping this example.")
            continue

    if len(predictions) == 0 or len(ground_truth) == 0:  # 检查是否有有效样本
        print("No valid predictions or ground truth available for MSE calculation.")
        return predictions, ground_truth

    mse = mean_squared_error(ground_truth, predictions)
    print(f"Validation MSE: {mse}")

    return predictions, ground_truth





# 可视化结果
def visualize_results(predictions, ground_truth):
    import matplotlib.pyplot as plt
    plt.plot(predictions, label='Predicted cwnd')
    plt.plot(ground_truth, label='True cwnd')
    plt.legend()
    plt.title("Predicted vs True cwnd")
    plt.show()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 调试时可以设置CUDA_LAUNCH_BLOCKING以获取更精确的错误
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    plm_model_dir = "/workspace/NetLLM/downloaded_plms/llama/base/"
    adapter_model_path = "/workspace/NetLLM/adaptive_bitrate_streaming/data/ft_plms/llama_base/._ss_None/rank_128_w_20_gamma_1.0_sfd_256_lr_0.0001_wd_0.0001_warm_2000_epochs_60_seed_100003/early_stop_-1_best_model/adapter_model.bin"
    modules_except_plm_path = "/workspace/NetLLM/adaptive_bitrate_streaming/data/ft_plms/llama_base/._ss_None/rank_128_w_20_gamma_1.0_sfd_256_lr_0.0001_wd_0.0001_warm_2000_epochs_60_seed_100003/early_stop_-1_best_model/modules_except_plm.bin"

    plm_model, adapter_model, modules_except_plm = load_model(adapter_model_path, plm_model_dir, modules_except_plm_path, device)

    validation_data_path = "./prague_exp_pool.pkl"
    validation_data = load_validation_data(validation_data_path)

    min_cwnd_value = 159280
    max_cwnd_value = 786264

    predictions, ground_truth = validate_model(plm_model, adapter_model, modules_except_plm, validation_data, min_cwnd_value, max_cwnd_value, device)

    visualize_results(predictions, ground_truth)

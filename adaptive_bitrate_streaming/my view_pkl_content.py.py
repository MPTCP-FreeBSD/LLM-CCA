import pickle

# 加载 .pkl 文件
with open('cwnd_rtt_experience_pool.pkl', 'rb') as f:
    data = pickle.load(f)

# 打印文件内容
print("States:", data['states'])
print("Actions:", data['actions'])
print("Rewards:", data['rewards'])
print("Dones:", data['dones'])

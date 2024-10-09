import pickle
from plm_special.data.exp_pool import ExperiencePool  # 导入 ExperiencePool 类


#这里是pkl的路径
#Path of the loaded PKL
pkl_path = 'C:\\Users\\10111\\Desktop\\SIT724\\Runpod\experiment data\\通过脚本或取得prague和cubic的数据\\prague cwnd script\\第三次实验\\只跑prague\\prague_exp_pool.pkl'
#这里是生成的txt的路径
#Path of output txt
txt_path = 'C:\\Users\\10111\\Desktop\\SIT724\\Runpod\experiment data\\通过脚本或取得prague和cubic的数据\\prague cwnd script\\第三次实验\\只跑prague\\prague_exp_pool.txt'

# 打开 .pkl 文件并加载内容
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

# 检查数据类型并解码 ExperiencePool 对象
if isinstance(data, ExperiencePool):
    with open(txt_path, 'w') as txt_file:
        txt_file.write("ExperiencePool Data:\n")
        
        # 遍历并写入经验池中的数据
        for i in range(len(data.states)):
            txt_file.write(f"State {i}: {data.states[i]}\n")
            txt_file.write(f"Action {i}: {data.actions[i]}\n")
            txt_file.write(f"Reward {i}: {data.rewards[i]}\n")
            txt_file.write(f"Done {i}: {data.dones[i]}\n\n")

        txt_file.write(f"\nTotal number of items: {len(data.states)}\n")
else:
    # 如果数据不是 ExperiencePool 对象，直接写入数据
    with open(txt_path, 'w') as txt_file:
        txt_file.write(str(data))
        txt_file.write("\n\nTotal number of items: {}\n".format(len(data)))

print(f"Data has been written to {txt_path}")


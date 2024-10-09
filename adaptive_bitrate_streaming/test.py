# #show_pkl.py
 
# import pickle
# path='exp_pool.pkl'   #path='/root/……/aus_openface.pkl'   pkl文件所在路径
	   
# f=open(path,'rb')
# data=pickle.load(f)
 
# print(data)
# print(len(data))


import pickle
from plm_special.data.exp_pool import ExperiencePool  # 导入 ExperiencePool 类

# 指定 .pkl 文件路径和输出的 .txt 文件路径

#官方的
#pkl_path = 'exp_pool.pkl'
#txt_path = 'exp_pool_contents.txt'

#pkl_path = 'cwnd_rtt_experience_pool.pkl'
#txt_path = 'cwnd_rtt_experience_pool.txt'
#这里是pkl的路径
pkl_path = 'C:\\Users\\10111\\Desktop\\SIT724\\Runpod\experiment data\\通过脚本或取得prague和cubic的数据\\prague cwnd script\\第三次实验\\只跑prague\\prague_exp_pool.pkl'
# #这里是生成的txt的路径
txt_path = 'C:\\Users\\10111\\Desktop\\SIT724\\Runpod\experiment data\\通过脚本或取得prague和cubic的数据\\prague cwnd script\\第三次实验\\只跑prague\\prague_exp_pool.txt'

# pkl_path = 'C:\\Users\\10111\\Desktop\\SIT724\\Runpod\experiment data\\通过脚本或取得prague和cubic的数据\\prague cwnd script\\第三次实验\\prague和cubic同时跑的数据\\competing_prague_output_rtt_cwnd_throughput.pkl'
# #这里是生成的txt的路径
# txt_path = 'C:\\Users\\10111\\Desktop\\SIT724\\Runpod\experiment data\\通过脚本或取得prague和cubic的数据\\prague cwnd script\\第三次实验\\prague和cubic同时跑的数据\\competing_prague_output_rtt_cwnd_throughput.txt'

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


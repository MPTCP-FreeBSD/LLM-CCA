import pandas as pd
import matplotlib.pyplot as plt
import os

# 读取 CSV 文件
file_path = r'C:\Users\10111\Desktop\SIT724\reproduce experiment\L4S experiment\Dualpi2\ECN enabled\1 client prague & cubic\ECN_enable_sameclient_prague_cubic_retransmission.csv'
data = pd.read_csv(file_path)

# 只选择前150个数据点
data = data.head(150)

# 绘制图像
plt.figure(figsize=(12, 6))

# 绘制 Client1 BBR 的重传数据
plt.plot(data['Interval start'], data['prague-retransmission'], linestyle='-', color='b', label='prague-retransmission')

# 绘制 Client2 Cubic 的重传数据
plt.plot(data['Interval start'], data['cubic-retransmission'], linestyle='-', color='r', label='prague-retransmission')

# 设置图表标题和标签
#plt.title('Retransmissions Over Time')
plt.xlabel('Interval Start (s)')
plt.ylabel('Retransmissions (packets)')

# 设置 x 轴刻度
plt.xticks(ticks=range(0, 151, 10))  # 每10秒一个刻度

# 添加图例
plt.legend()

# 显示图表
plt.tight_layout()

# 保存图表到当前目录
output_dir = os.path.dirname(file_path)
output_path = os.path.join(output_dir, 'retransmission.png')
plt.savefig(output_path)

# 显示图表
plt.show()

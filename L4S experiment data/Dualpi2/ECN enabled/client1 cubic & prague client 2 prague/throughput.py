import pandas as pd
import matplotlib.pyplot as plt
import os

# 读取 CSV 文件
file_path = r'C:\Users\10111\Desktop\SIT724\reproduce experiment\L4S experiment\Dualpi2\ECN enabled\client1 cubic & prague client 2 prague\ECN_enable_client1_cubic_prague_client2_prague_throughput.csv'
data = pd.read_csv(file_path)

# 只选择前150个数据点
data = data.head(150)

# 绘制图像
plt.figure(figsize=(12, 6))

# 绘制 Client1 Cubic 的吞吐量数据
plt.plot(data['Interval start'], data['client1-cubic-throughput'], linestyle='-', color='b', label='Client1 Cubic Throughput')

# 绘制 Client1 Prague 的吞吐量数据
plt.plot(data['Interval start'], data['client1-prague-throughput'], linestyle='-', color='r', label='Client1 Prague Throughput')

# 绘制 Client2 Prague 的吞吐量数据
plt.plot(data['Interval start'], data['client2-prague-throughput'], linestyle='-', color='g', label='Client2 Prague Throughput')

# 设置图表标题和标签
#plt.title('Throughput Over Time')
plt.xlabel('Interval Start (s)')
plt.ylabel('Throughput (Bps)')

# 设置 x 轴刻度
plt.xticks(ticks=range(0, 151, 10))  # 每10秒一个刻度

# 设置 y 轴数值格式为普通数字
plt.ticklabel_format(style='plain', axis='y')

# 添加图例
plt.legend()

# 显示图表
plt.tight_layout()

# 保存图表到当前目录
output_dir = os.path.dirname(file_path)
output_path = os.path.join(output_dir, 'throughput.png')
plt.savefig(output_path)

# 显示图表
plt.show()
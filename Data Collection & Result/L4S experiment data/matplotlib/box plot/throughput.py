import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 定义文件夹路径和文件名称
base_dir = r'C:\Users\10111\Desktop\test'
files = ['1 flow throughput.csv', '2 flow throughput.csv', '4 flow throughput.csv']

# 用于存储所有数据的列表
data_list = []
flow_labels = ['1 Flow', '2 Flows', '4 Flows']

# 读取每个文件中的 CSV 数据
for file, flow_label in zip(files, flow_labels):
    file_path = os.path.join(base_dir, file)
    if os.path.exists(file_path):
        print(f"Reading file: {file_path}")
        df = pd.read_csv(file_path)
        df = df.head(150)  # 只取前150个interval的数据
        # 添加流量标签
        df['Condition'] = f'{flow_label}'
        data_list.append(df)
    else:
        print(f"File not found: {file_path}")

# 检查是否有数据被读取
if not data_list:
    print("No data was read. Please check the file paths and ensure the files exist.")
else:
    # 合并所有数据到一个DataFrame
    combined_df = pd.concat(data_list, ignore_index=True)

# 重塑数据以便于绘图
combined_df_melted = combined_df.melt(id_vars=['Interval start', 'Condition'], 
                                      value_vars=['prague-throughput', 'cubic-throughput'],
                                      var_name='Algorithm', value_name='Throughput')

# 绘制box plot
plt.figure(figsize=(16, 10))
# showfliers=False 关闭异常值显示
sns.boxplot(x='Condition', y='Throughput', hue='Algorithm', data=combined_df_melted, showfliers=False, hue_order=['prague-throughput', 'cubic-throughput'])
plt.title('Throughput Comparison Under Different Conditions')
plt.xlabel('Condition')
plt.ylabel('Throughput')
plt.xticks(rotation=0)
plt.legend(title='Algorithm')
plt.ticklabel_format(style='plain', axis='y')  # 禁用 y 轴上的科学计数法
plt.tight_layout()
plt.show()

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 定义文件夹路径和文件名称
base_dir = r'C:\Users\10111\Desktop\SIT724\reproduce experiment\L4S experiment\Dualpi2\ECN disabled'
files = ['1 flow retransmission.csv', '2 flow retransmission.csv', '4 flow retransmission.csv']

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

# 打印列名以确保数据列正确
print("Columns in combined dataframe:", combined_df.columns)

# 确保 'Condition' 列为字符串类型
combined_df['Condition'] = combined_df['Condition'].astype(str)

# 确保 'Retransmission' 列为数值类型
combined_df['prague-retransmission'] = pd.to_numeric(combined_df['prague-retransmission'], errors='coerce')
combined_df['cubic-retransmission'] = pd.to_numeric(combined_df['cubic-retransmission'], errors='coerce')

# 检查是否有非数值数据
non_numeric_rows = combined_df[combined_df[['prague-retransmission', 'cubic-retransmission']].isna().any(axis=1)]
print(non_numeric_rows)

# 删除包含非数值 'Retransmission' 值的行
combined_df = combined_df.dropna(subset=['prague-retransmission', 'cubic-retransmission'])

# 重塑数据以便于绘图
combined_df_melted = combined_df.melt(id_vars=['Interval start', 'Condition'], 
                                      value_vars=['prague-retransmission', 'cubic-retransmission'],
                                      var_name='Algorithm', value_name='Retransmission')

# 确保 'Algorithm' 列为字符串类型
combined_df_melted['Algorithm'] = combined_df_melted['Algorithm'].astype(str)

# 绘制box plot
plt.figure(figsize=(16, 10))
sns.boxplot(x='Condition', y='Retransmission', hue='Algorithm', data=combined_df_melted, showfliers=False, hue_order=['cubic-retransmission', 'prague-retransmission'])
plt.title('Retransmission Comparison Under Different Conditions')
plt.xlabel('Condition')
plt.ylabel('Retransmission (Packets)')
plt.xticks(rotation=0)
plt.legend(title='Algorithm')
plt.subplots_adjust(bottom=0.21)  # 设置图像底部边距
plt.show()

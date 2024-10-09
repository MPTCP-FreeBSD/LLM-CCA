# 根据 json 文件来的更准确，用 ss -ti 是错的
import json
import csv

# 读取 JSON 文件
with open('iperf3_prague_output.json', 'r') as json_file:
    iperf3_data = json.load(json_file)

# 提取 RTT, CWND 和 throughput 数据
data = []
for interval in iperf3_data['intervals']:
    for stream in interval['streams']:
        second = interval['sum']['start']  # 每秒的开始时间
        rtt = stream.get('rtt', 0) / 1000000  # 将 rtt 从毫秒转换为秒
        cwnd = stream.get('snd_cwnd', 0)   # cwnd 以字节为单位
        throughput = interval['sum'].get('bits_per_second', 0)  # throughput 以 bps 为单位
        
        # 仅当 throughput 不为 0 时，才将数据添加到列表中
        # if throughput > 0:
        data.append({
            'second': second,
            'rtt': rtt,
            'cwnd': cwnd,
            'throughput': throughput
        })

# 将数据写入 CSV 文件
with open('prague_output_rtt_cwnd_throughput.csv', 'w', newline='') as csvfile:
    fieldnames = ['second', 'rtt', 'cwnd', 'throughput']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(data)

print("数据已成功写入csv")

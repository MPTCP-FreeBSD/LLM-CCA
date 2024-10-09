# 输出 JSON 文件路径
JSON_OUTPUT_FILE="iperf3_bbr_output.json"

# 服务器 IP 地址和端口
SERVER_IP="192.168.4.14"
SERVER_PORT="3002"

# 启动 `iperf3` 并将输出保存为 JSON 文件
iperf3 -c "$SERVER_IP" -p "$SERVER_PORT" -C bbr -tinf --json > "$JSON_OUTPUT_FILE"


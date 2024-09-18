ip_address=$(python3 -c "import socket;print(f'{socket.gethostbyname(socket.gethostname())}')")

echo $ip_address

mkdir -p /tmp/rayserve-demo; cd /tmp/rayserve-demo

ray metrics launch-prometheus

export RAY_GRAFANA_HOST=http://${ip_address}:3000

ray start --head --dashboard-host 0.0.0.0 --metrics-export-port 8080

/usr/share/grafana/bin/grafana-server --homepath /usr/share/grafana --config /tmp/ray/session_latest/metrics/grafana/grafana.ini web >grafana.stdout.log 2>&1 &

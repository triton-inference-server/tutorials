
ip_address=$(hostname -I | awk '{print $1}')

echo $ip_address

mkdir -p /tmp/rayserve-demo; cd /tmp/rayserve-demo

ray metrics launch-prometheus

export RAY_GRAFANA_HOST=http://${ip_address}:3000

ray start --head --dashboard-host 0.0.0.0 --metrics-export-port 8080 --disable-usage-stats

/usr/share/grafana/bin/grafana-server --homepath /usr/share/grafana --config /tmp/ray/session_latest/metrics/grafana/grafana.ini web >grafana.stdout.log 2>&1 &

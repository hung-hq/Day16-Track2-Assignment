#!/bin/bash
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "Starting user_data setup for CPU benchmark node"

# Update and install runtime dependencies for ML benchmark
dnf update -y
dnf install -y python3 python3-pip

python3 -m pip install --upgrade pip
python3 -m pip install lightgbm scikit-learn pandas numpy kaggle

# Working directory for benchmark assets
mkdir -p /home/ec2-user/ml-benchmark
chown -R ec2-user:ec2-user /home/ec2-user/ml-benchmark

cat << 'EOF' > /opt/health_server.py
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
  def do_GET(self):
    if self.path == "/health":
      self.send_response(200)
      self.send_header("Content-Type", "application/json")
      self.end_headers()
      self.wfile.write(b'{"status":"ok","mode":"cpu-benchmark"}')
      return

    self.send_response(404)
    self.end_headers()


if __name__ == "__main__":
  HTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
EOF

cat << 'EOF' > /etc/systemd/system/cpu-health.service
[Unit]
Description=CPU Benchmark Health Endpoint
After=network.target

[Service]
ExecStart=/usr/bin/python3 /opt/health_server.py
Restart=always
User=root

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable cpu-health
systemctl start cpu-health

echo "CPU benchmark node initialized"
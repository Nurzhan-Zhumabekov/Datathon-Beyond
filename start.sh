#!/bin/zsh
# Starts the service + a public HTTPS address and keeps the Mac awake.
# Uses ngrok (fixed free domain of the account, same URL after every restart) if it is configured,
# otherwise falls back to a Cloudflare quick tunnel (URL changes on every start).
# usage: ./start.sh            (Ctrl+C stops everything)
cd "$(dirname "$0")"
mkdir -p data

caffeinate -ims -w $$ &

.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --log-level warning > data/server.log 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; pkill -P $$ 2>/dev/null; exit" INT TERM EXIT

until curl -s http://127.0.0.1:8000/health > /dev/null; do sleep 1; done
echo "Service is up on http://127.0.0.1:8000"
echo "API key: see .env (API_KEY=...)"

if command -v ngrok > /dev/null && ngrok config check > /dev/null 2>&1; then
  ngrok http 8000 --log stdout --log-format logfmt > data/ngrok.log 2>&1 &
  for i in {1..30}; do
    url=$(curl -s http://127.0.0.1:4040/api/tunnels | python3 -c "import sys,json; t=json.load(sys.stdin).get('tunnels',[]); print(t[0]['public_url'] if t else '')" 2>/dev/null)
    [[ -n "$url" ]] && break
    sleep 1
  done
  echo "$url" > data/public_url.txt
  echo "PUBLIC URL (ngrok, fixed): $url"
  wait
else
  cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8000 2>&1 | while read -r line; do
    echo "$line" >> data/tunnel.log
    url=$(echo "$line" | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com')
    if [[ -n "$url" ]]; then
      echo "$url" > data/public_url.txt
      echo "PUBLIC URL (cloudflare, changes on restart): $url"
    fi
  done
fi

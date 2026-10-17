#!/bin/zsh
# Starts the service and a public HTTPS tunnel, and keeps the Mac awake.
# usage: ./start.sh            (Ctrl+C stops everything)
cd "$(dirname "$0")"
mkdir -p data

# Keep the Mac awake while the service runs (display may sleep; system must not).
caffeinate -imsu -w $$ &

# API + web UI on localhost:8000
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --log-level warning > data/server.log 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

until curl -s http://127.0.0.1:8000/health > /dev/null; do sleep 1; done
echo "Service is up on http://127.0.0.1:8000"
echo "API key: $(grep API_KEY .env | cut -d= -f2)"

# Free Cloudflare quick tunnel (no account). The public URL is printed below and saved to data/public_url.txt
cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8000 2>&1 | while read -r line; do
  echo "$line" >> data/tunnel.log
  url=$(echo "$line" | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com')
  if [[ -n "$url" ]]; then
    echo "$url" > data/public_url.txt
    echo "PUBLIC URL: $url"
  fi
done

#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/Lai-Yanjun/BTC_15m_Prediction.git"
BRANCH="main"
APP_DIR="${HOME}/apps/predicta-live"
SERVICE_NAME="predicta-live"
SERVICE_USER="$(id -un)"
PROXY_URL=""

usage() {
  cat <<'EOF'
Usage:
  bash deploy_raspi.sh [--proxy http://127.0.0.1:7897] [--app-dir /home/pi/apps/predicta-live] [--service-user pi]

What it does:
  1) Install system deps
  2) Clone/update repo
  3) Create venv and install requirements
  4) Create .env if missing
  5) Install/refresh systemd service
  6) Enable + restart service
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --proxy)
      PROXY_URL="${2:-}"
      shift 2
      ;;
    --app-dir)
      APP_DIR="${2:-}"
      shift 2
      ;;
    --service-user)
      SERVICE_USER="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

echo "[1/6] Installing system dependencies..."
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip jq

echo "[2/6] Cloning/updating repository..."
mkdir -p "$(dirname "$APP_DIR")"
if [[ ! -d "$APP_DIR/.git" ]]; then
  git clone -b "$BRANCH" "$REPO_URL" "$APP_DIR"
else
  cd "$APP_DIR"
  git fetch origin
  git checkout "$BRANCH"
  git reset --hard "origin/${BRANCH}"
fi

cd "$APP_DIR"

echo "[3/6] Building python environment..."
if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "[4/6] Ensuring env file exists..."
if [[ ! -f ".env" ]]; then
  cp .env.example .env
  echo "Created .env from template. Please edit: ${APP_DIR}/.env"
fi

echo "[5/6] Writing systemd service..."
EXEC_START="${APP_DIR}/.venv/bin/python run_live_model.py"
if [[ -n "$PROXY_URL" ]]; then
  EXEC_START="${EXEC_START} --proxy ${PROXY_URL}"
fi

sudo tee "/etc/systemd/system/${SERVICE_NAME}.service" >/dev/null <<EOF
[Unit]
Description=Predicta 15m Live Trader
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
WorkingDirectory=${APP_DIR}
Environment=PYTHONUNBUFFERED=1
ExecStart=${EXEC_START}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "[6/6] Enabling and starting service..."
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"

echo
echo "Done."
echo "Next:"
echo "  1) Edit ${APP_DIR}/.env and ${APP_DIR}/trade_config.yaml"
echo "  2) Restart service: sudo systemctl restart ${SERVICE_NAME}"
echo "  3) Check status:    sudo systemctl status ${SERVICE_NAME}"
echo "  4) Tail logs:       journalctl -u ${SERVICE_NAME} -f"

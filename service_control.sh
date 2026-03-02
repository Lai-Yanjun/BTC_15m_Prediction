#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${1:-predicta-live}"
ACTION="${2:-status}"

case "$ACTION" in
  start)
    sudo systemctl start "$SERVICE_NAME"
    ;;
  stop)
    sudo systemctl stop "$SERVICE_NAME"
    ;;
  restart)
    sudo systemctl restart "$SERVICE_NAME"
    ;;
  status)
    sudo systemctl status "$SERVICE_NAME"
    ;;
  enable)
    sudo systemctl enable "$SERVICE_NAME"
    ;;
  disable)
    sudo systemctl disable "$SERVICE_NAME"
    ;;
  logs)
    journalctl -u "$SERVICE_NAME" -f
    ;;
  *)
    cat <<'EOF'
Usage:
  bash service_control.sh [service_name] [action]

Examples:
  bash service_control.sh predicta-live status
  bash service_control.sh predicta-live stop
  bash service_control.sh predicta-live logs
EOF
    exit 2
    ;;
esac

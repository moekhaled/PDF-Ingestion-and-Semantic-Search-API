#!/usr/bin/env bash
set -euo pipefail

ACTION=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --action)
      ACTION="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$ACTION" ]]; then
  echo "Usage: ./orchestrate.sh --action start|terminate"
  exit 1
fi

case "$ACTION" in
  start)
    docker compose up --build -d
    ;;
  terminate)
    docker compose down -v
    ;;
  *)
    echo "Invalid action: $ACTION"
    exit 1
    ;;
esac

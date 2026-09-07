#!/bin/sh
set -e

uid="${HOST_UID:-1000}"
gid="${HOST_GID:-1000}"

mkdir -p /data/projects /data/weights
chown -R "${uid}:${gid}" /data/projects /data/weights

echo "TEM instance segmentation — open http://localhost:8000"

if command -v gosu >/dev/null 2>&1; then
  exec gosu "${uid}:${gid}" uvicorn app.main:app --host 0.0.0.0 --port 8000
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000

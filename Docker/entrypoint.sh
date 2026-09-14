#!/bin/sh
set -e

uid="${HOST_UID:-1000}"
gid="${HOST_GID:-1000}"

export PATH="/opt/conda/bin:${PATH}"
# The image HOME is often /root. After gosu that directory is not writable.
export HOME=/tmp
export PYTHONPATH="/app"
# Ultralytics AMP checks download a tiny probe checkpoint (not the training model).
export YOLO_CONFIG_DIR=/data/weights/ultralytics/config

mkdir -p /data/projects /data/weights/ultralytics "$YOLO_CONFIG_DIR"
# CWD is /app (not writable). Some Ultralytics versions mkdir a relative "weights/".
if [ ! -e /app/weights ]; then
  ln -s /data/weights/ultralytics /app/weights
fi
chown -R "${uid}:${gid}" /data/projects /data/weights

echo "TEM instance segmentation — open http://localhost:8000"

if command -v gosu >/dev/null 2>&1; then
  exec gosu "${uid}:${gid}" uvicorn app.main:app --host 0.0.0.0 --port 8000
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000

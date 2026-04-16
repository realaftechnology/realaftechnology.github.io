#!/bin/bash

# Seed volume db from bundled db.sqlite on first boot
if [ -d /data ] && [ ! -f /data/db.sqlite ] && [ -f db.sqlite ]; then
  echo "==> Seeding /data/db.sqlite from bundled db..."
  cp db.sqlite /data/db.sqlite
fi

echo "==> Running ingest (new episodes only)..."
python3 ingest.py || echo "⚠️  Ingest had errors but continuing..."

echo "==> Starting server..."
exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120

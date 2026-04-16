#!/bin/bash
set -e

echo "==> Running ingest (new episodes only)..."
python ingest.py

echo "==> Starting server..."
exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120

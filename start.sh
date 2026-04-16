#!/bin/bash

echo "==> Running ingest (new episodes only)..."
python3 ingest.py || echo "⚠️  Ingest had errors but continuing..."

echo "==> Starting server..."
exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120

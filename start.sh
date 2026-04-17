#!/bin/bash

# Use Railway persistent volume for the database
export DB_PATH=/data/db.sqlite

echo "==> Checking database on volume..."
if [ ! -f "$DB_PATH" ]; then
  # Volume is empty — seed from bundled db if available, otherwise build fresh
  BUNDLED="$(dirname "$0")/db.sqlite"
  if [ -f "$BUNDLED" ]; then
    echo "==> Seeding volume from bundled db.sqlite..."
    cp "$BUNDLED" "$DB_PATH"
    echo "==> Seed complete."
  else
    echo "==> No bundled db found — will build fresh from transcripts."
  fi
fi

echo "==> Running ingest (new episodes only)..."
python3 ingest.py || echo "⚠️  Ingest had errors but continuing..."

echo "==> Starting server..."
exec gunicorn app:app --bind 0.0.0.0:$PORT --worker-class gthread --workers 2 --threads 4 --timeout 120

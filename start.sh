#!/bin/bash

# Use Railway persistent volume if it's mounted and writable, otherwise use local db
if [ -d "/data" ] && touch /data/.write_test 2>/dev/null; then
  rm -f /data/.write_test
  export DB_PATH=/data/db.sqlite
  echo "==> Volume at /data is available — using /data/db.sqlite"

  if [ ! -f "$DB_PATH" ]; then
    BUNDLED="$(dirname "$0")/db.sqlite"
    if [ -f "$BUNDLED" ]; then
      echo "==> Seeding volume from bundled db.sqlite..."
      cp "$BUNDLED" "$DB_PATH"
      echo "==> Seed complete."
    else
      echo "==> No bundled db — will build fresh from transcripts."
    fi
  fi
else
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  export DB_PATH="$SCRIPT_DIR/db.sqlite"
  echo "==> Volume not available — using local db at $DB_PATH"
fi

echo "==> Running ingest (new episodes only)..."
python3 ingest.py || echo "⚠️  Ingest had errors but continuing..."

echo "==> Starting server..."
exec gunicorn app:app --bind 0.0.0.0:$PORT --worker-class gthread --workers 2 --threads 4 --timeout 3600

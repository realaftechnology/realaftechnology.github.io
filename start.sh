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

# Ingest runs in the background so the web server can start immediately and
# pass Railway's health check. When there's nothing new, ingest's hash-check
# finishes in seconds; when there's a big batch of new episodes (e.g. the
# initial Andygram import, which embeds ~2k posts via the OpenAI API), it can
# run for 30+ minutes — blocking gunicorn startup causes the deploy to be
# marked unhealthy.
#
# Concurrency note: SQLite handles the concurrent read-while-write cleanly
# at file-level locking. Newly-ingested episodes become searchable the
# moment ingest commits them.
echo "==> Running ingest in background (logs: /tmp/ingest.log)..."
mkdir -p /tmp
nohup python3 ingest.py > /tmp/ingest.log 2>&1 &
echo "==> Ingest PID: $!"

echo "==> Starting server..."
exec gunicorn app:app --bind 0.0.0.0:$PORT --worker-class gthread --workers 2 --threads 4 --timeout 3600

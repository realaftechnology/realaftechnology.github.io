# AFBrain

## Local Setup

1. Install: `pip3 install flask gunicorn python-docx`
2. Set vars: `export OPENAI_API_KEY="..." AFBRAIN_PASSWORD="..." SECRET_KEY="..."`
3. Drop .docx files into transcripts/
4. Run: `python3 ingest.py`
5. Run: `python3 app.py` → http://localhost:5000

## Deploy to Railway

1. Push this folder to a GitHub repo (add db.sqlite to .gitignore)
2. railway.app → New Project → Deploy from GitHub
3. Set these environment variables in Railway:
   - OPENAI_API_KEY
   - AFBRAIN_PASSWORD
   - SECRET_KEY (any random string)
   - DB_PATH = /data/db.sqlite
4. Add a Volume mounted at /data
5. Upload db.sqlite to the volume via Railway dashboard

## Adding episodes
1. Export from Trint as .docx
2. Drop into transcripts/
3. python3 ingest.py
4. Push updated db.sqlite to Railway volume

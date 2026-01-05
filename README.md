# CSV to DB Uploader

Simple Flask app to upload CSV files and import them into a MySQL database.

**Project layout**
- `app.py`: Flask app that accepts CSV uploads, creates tables from CSV headers, and inserts rows into MySQL.
- `templates/upload.html`: Upload form used by the web UI.
- `uploads/`: Default upload directory (created automatically).
- `etl_log.txt`: Log file for info messages.

**Requirements**
- Python 3.8+
- See `requirements.txt` for required Python packages.

**Environment / LLM config**
- Configure Ollama endpoint and model via environment variables (optional):

   - `OLLAMA_URL` — URL to Ollama generate API (default: `http://localhost:11434/api/generate`)
   - `LLM_MODEL` — model name to use (default: `phi3`)

   Example (PowerShell):

   ```powershell
   $env:OLLAMA_URL = "http://localhost:11434/api/generate"
   $env:LLM_MODEL = "phi3"
   ```

**Setup**
1. Create a virtual environment and activate it:

   ```powershell
   python -m venv .venv; .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Configure the database connection in `app.py` by editing the `DB_CONFIG` dictionary. Do NOT commit credentials to version control.

**Run (development)**

```powershell
python app.py
```

Open `http://127.0.0.1:5000/` in your browser and upload CSV files. Each CSV will create a table named after the file (without extension). Column names are normalized to lowercase with spaces, dashes, and slashes replaced by underscores.

**Notes & Tips**
- The app uses `mysql-connector-python` to connect to MySQL. Ensure your MySQL server is reachable from the machine running the app.
- The app optionally calls an Ollama-compatible LLM if `OLLAMA_URL` is reachable. If the LLM is unavailable the server falls back to a simple rule-based PDF parser.
- The current implementation uses simple type inference (INT → BIGINT, float → DOUBLE, else TEXT). Adjust the logic in `create_table` in `app.py` if you need different types.
- For production, disable `debug=True` in `app.run`, add error handling, and use a proper WSGI server (e.g., Gunicorn) behind a reverse proxy.


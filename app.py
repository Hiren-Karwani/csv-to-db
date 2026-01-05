import os
import re
import uuid
import json
from urllib.parse import urlparse, unquote
import mysql.connector
from flask import Flask, render_template, request, send_file, send_from_directory, jsonify
from datetime import datetime
import logging
from PIL import Image
import pytesseract
import cv2
import requests
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import pandas as pd

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

logging.basicConfig(level=logging.INFO)

# LLM config
OLLAMA_URL = os.environ.get("OLLAMA_URL")
OLLAMA_MODELS = ["mistral", "phi3"]
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

# ----------------- FILESS / MYSQL CONFIGURATION -----------------
DB_CONFIG = {
    'user': 'csv_to_db_outsidedo',
    'password': os.environ.get('12345678', '12345678'), 
    'host': 'zja4ea.h.filess.io',
    'port': '61032',
    'database': 'csv_to_db_outsidedo',
    # CHANGED: Set to False to prevent crashing on "Table already exists" warnings
    'raise_on_warnings': False 
}

def get_db_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        logging.error(f"Error connecting to Filess MySQL: {err}")
        return None

def init_db():
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            # This will now safely issue a warning if table exists, without crashing
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    filename VARCHAR(512),
                    doc_type VARCHAR(255),
                    data LONGTEXT,
                    raw_text LONGTEXT,
                    created_at DATETIME
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )
            conn.commit()
            cur.close()
            conn.close()
            logging.info('Initialized MySQL documents table on Filess')
        except Exception as e:
            # We catch it just in case, but raise_on_warnings: False prevents the main crash
            logging.warning(f'Initialization note (safe to ignore if table exists): {e}')
    else:
        logging.error("Could not initialize DB: Connection failed.")

def save_document_to_db(filename, doc_type, data_dict, raw_text):
    """Save document to MySQL. For CSVs, save each row as separate entry."""
    conn = get_db_connection()
    if not conn:
        logging.error('Could not connect to database')
        return False
    
    try:
        cur = conn.cursor()
        # For CSV data that's a list of rows, save each row separately
        if isinstance(data_dict, list):
            for i, row in enumerate(data_dict):
                unique_id = f"{filename}_{i}"
                data_str = json.dumps(row) if isinstance(row, dict) else json.dumps({"row": row})
                cur.execute(
                    "INSERT INTO documents (filename, doc_type, data, raw_text, created_at) VALUES (%s, %s, %s, %s, NOW())",
                    (unique_id, doc_type, data_str, json.dumps(row))
                )
        else:
            # Single document (PDF, image, or aggregated data)
            data_str = json.dumps(data_dict) if not isinstance(data_dict, str) else data_dict
            cur.execute(
                "INSERT INTO documents (filename, doc_type, data, raw_text, created_at) VALUES (%s, %s, %s, %s, NOW())",
                (filename, doc_type, data_str, raw_text)
            )
        
        conn.commit()
        cur.close()
        conn.close()
        logging.info(f'Saved {filename} to database')
        return True
    except Exception:
        logging.exception(f'Failed to save {filename} to database')
        if conn:
            conn.close()
        return False

# Initialize DB on startup
init_db()

# ---------------- LLM FUNCTIONS ----------------

def call_ollama(prompt, model):
    if not OLLAMA_URL:
        return None
    try:
        payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}}
        r = requests.post(OLLAMA_URL, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data.get("response") or data.get("text") or data.get("output") or (data.get('result') if isinstance(data.get('result'), str) else None)
    except Exception:
        logging.exception(f"Ollama call failed for model {model}")
        return None


def call_openai(prompt, model=OPENAI_MODEL):
    if not OPENAI_API_KEY:
        return None
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that answers concisely and returns output as JSON when requested."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 512
        }
        r = requests.post(url, headers=headers, json=body, timeout=30)
        r.raise_for_status()
        j = r.json()
        if "choices" in j and len(j["choices"]) > 0:
            return j["choices"][0]["message"]["content"]
        return None
    except Exception:
        return None


def council_call(user_question, context=""):
    prompt = (
        "Please answer the user question using the provided CONTEXT. "
        "Return a JSON object with keys: answer (string) and confidence (number between 0 and 1). "
        "If you cannot answer, return answer as an empty string and confidence 0."
        f"\nCONTEXT:\n{context}\nQUESTION: {user_question}\nJSON:" 
    )

    responders = []
    if OLLAMA_URL:
        for m in OLLAMA_MODELS:
            raw = call_ollama(prompt, model=m)
            responders.append((f"ollama-{m}", raw))

    if OPENAI_API_KEY:
        raw = call_openai(prompt)
        responders.append(("openai", raw))

    details = []
    for provider, raw in responders:
        if not raw:
            details.append({"provider": provider, "raw": None, "answer": "", "confidence": 0.0})
            continue
        ans = ""
        conf = 0.0
        try:
            start = raw.find('{')
            end = raw.rfind('}')
            if start != -1 and end != -1:
                snippet = raw[start:end+1]
                parsed = json.loads(snippet)
                ans = str(parsed.get('answer', '')) if parsed.get('answer') is not None else ''
                conf = float(parsed.get('confidence', 0.0)) if parsed.get('confidence') is not None else 0.0
            else:
                ans = raw.strip()
                conf = 0.5
        except Exception:
            ans = raw.strip()
            conf = 0.5

        details.append({"provider": provider, "raw": raw, "answer": ans, "confidence": conf})

    agg = {}
    for d in details:
        a = d['answer'].strip()
        if not a:
            continue
        key = a
        agg.setdefault(key, 0.0)
        agg[key] += d.get('confidence', 0.0) + 0.2 

    if not agg:
        best = max(details, key=lambda x: x.get('confidence', 0.0)) if details else None
        if best:
            return best['answer'] or (best['raw'] or ''), details
        return "", details

    best_answer = max(agg.items(), key=lambda x: x[1])[0]
    return best_answer, details


# ---------------- OCR & TEXT EXTRACTION ----------------

def preprocess_image(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return thresh

def extract_text_from_image(path):
    processed = preprocess_image(path)
    return pytesseract.image_to_string(processed)

def extract_text_from_pdf(path):
    text = ""
    try:
        reader = PdfReader(path)
        for p in reader.pages:
            t = p.extract_text()
            if t:
                text += t
    except Exception:
        pass

    if text.strip():
        return text

    images = convert_from_path(path)
    for img in images:
        temp = f"temp_{uuid.uuid4()}.png"
        img.save(temp)
        text += extract_text_from_image(temp)
        os.remove(temp)

    return text


def extract_structured_data_from_text(text, doc_type):
    if not text:
        return {}
    
    if doc_type == 'Receipt / Bill':
        return extract_receipt_fields(text)
    
    # For resumes, extract key fields
    if 'resume' in doc_type.lower() or 'cv' in doc_type.lower():
        return {
            "name": extract_name(text),
            "email": extract_email(text),
            "phone": extract_phone(text),
            "years_experience": extract_years_experience(text),
            "skills": extract_skills(text),
            "preview_lines": [l.strip() for l in text.splitlines() if l.strip()][:20]
        }
    
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    preview_lines = lines[:20]
    return {"preview_lines": preview_lines}

def extract_name(text):
    """Extract name from resume - typically first 1-2 capitalized lines"""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        return lines[0]
    return ""

def extract_email(text):
    """Extract email address from text"""
    match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    return match.group(0) if match else ""

def extract_phone(text):
    """Extract phone number from text"""
    match = re.search(r'\b(?:\+?1[-.]?)?(?:\(?[0-9]{3}\)?[-.]?)?[0-9]{3}[-.]?[0-9]{4}\b', text)
    return match.group(0) if match else ""

def extract_years_experience(text):
    """Extract years of experience from resume text"""
    # Look for patterns like "5 years", "10+ years", etc.
    match = re.search(r'(\d+)\+?\s*(?:years?|yrs)(?:\s+(?:of\s+)?experience)?', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    # Also check for "exp:" or "experience:"
    match = re.search(r'(?:exp|experience):\s*(\d+)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0

def extract_skills(text):
    """Extract skills from resume"""
    skills = []
    # Look for common skill keywords
    skill_keywords = ['python', 'java', 'javascript', 'sql', 'c++', 'c#', 'go', 'rust', 'php',
                     'react', 'vue', 'angular', 'node.js', 'django', 'flask', 'spring',
                     'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'ci/cd',
                     'machine learning', 'ai', 'data science', 'analytics', 'excel',
                     'communication', 'leadership', 'project management']
    
    text_lower = text.lower()
    for skill in skill_keywords:
        if skill in text_lower:
            skills.append(skill.title())
    
    return list(set(skills))[:10]  # Return unique skills, max 10

# ---------------- CLASSIFICATION ----------------

def classify_document(text):
    t = text.lower()
    if "income tax" in t or "itr" in t or "pan" in t:
        return "Income Tax Return"
    if any(x in t for x in ["total", "gst", "amount", "invoice"]):
        return "Receipt / Bill"
    if "semester" in t or "examination" in t:
        return "Academic Document"
    return "General Document"

# ---------------- DATA EXTRACTION ----------------

def extract_receipt_fields(text):
    return {
        "Vendor": re.search(r"([A-Z][A-Za-z &]{3,})", text).group(1) if re.search(r"([A-Z][A-Za-z &]{3,})", text) else "",
        "Date": re.search(r"\d{2}[/-]\d{2}[/-]\d{4}", text).group(0) if re.search(r"\d{2}[/-]\d{2}[/-]\d{4}", text) else "",
        "Total Amount": re.search(r"(₹|\$)?\s?\d+(\.\d{2})", text).group(0) if re.search(r"(₹|\$)?\s?\d+(\.\d{2})", text) else ""
    }

# ---------------- RESULT PDF ----------------

def create_result_pdf(data, filename):
    """No-op: PDFs are no longer generated locally. All data is stored in database."""
    return None

# ---------------- ROUTES ----------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        name = file.filename
        path = os.path.join(UPLOAD_FOLDER, name)
        file.save(path)

        # Logic to process and save ONE file (for the index route upload)
        if name.lower().endswith((".png", ".jpg", ".jpeg")):
            text = extract_text_from_image(path)
            category = classify_document(text)
            structured = {}
            if category == 'Receipt / Bill':
                structured = extract_structured_data_from_text(text, category)
            save_document_to_db(name, category, structured, text)
            
        elif name.lower().endswith(".csv"):
            df = pd.read_csv(path)
            text = df.to_string()
            data_dict = {"rows": len(df), "columns": df.columns.tolist()}
            category = "CSV Data"
            save_document_to_db(name, category, data_dict, text)
            
        else:
            text = extract_text_from_pdf(path)
            category = classify_document(text)
            structured = extract_structured_data_from_text(text, category)
            save_document_to_db(name, category, structured, text)

        result = {"Document Type": category}
        # Create PDF result for immediate download
        pdf_name = f"result_{uuid.uuid4()}.pdf"
        result_pdf = create_result_pdf(result, pdf_name)
        return send_file(result_pdf, as_attachment=True)

    return render_template("uploads.html")

@app.route("/preview/<name>")
def preview(name):
    path = os.path.join(UPLOAD_FOLDER, name)

    if name.lower().endswith((".png",".jpg",".jpeg")):
        text = extract_text_from_image(path)
    elif name.lower().endswith(".csv"):
        df = pd.read_csv(path)
        return df.head(10).to_html(classes="preview-table", index=False, border=0)
    else:
        text = extract_text_from_pdf(path)

    if name.lower().endswith('.pdf'):
        doc_type = classify_document(text)
        structured = extract_structured_data_from_text(text, doc_type)
        if not structured:
            return "<p>No preview available.</p>"

        if 'preview_lines' in structured:
            rows = ''.join(f"<tr><td>{line}</td></tr>" for line in structured['preview_lines'])
            return f"<div><strong>Preview Lines:</strong><table class=\"preview-table\"><tbody>{rows}</tbody></table></div>"

        rows = ''.join(f"<tr><th style=\"width:200px; text-align:left;\">{k}</th><td>{(v if v is not None else '')}</td></tr>" for k, v in structured.items())
        return f"<div><strong>Extracted Data:</strong><table class=\"preview-table\"><tbody>{rows}</tbody></table></div>"

    return f"<pre>{text}</pre>"


@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files')
    if not files:
        return "<p>No files uploaded.</p>", 200

    out = "<h3>Upload Summary</h3><ul>"
    for f in files:
        fname = f.filename
        path = os.path.join(UPLOAD_FOLDER, fname)
        try:
            f.save(path)
            
            # --- CSV HANDLING ---
            if fname.lower().endswith('.csv'):
                df = pd.read_csv(path)
                # Convert each row to a dict and save as separate document
                rows_list = df.to_dict(orient='records')
                raw_text = df.to_string()
                
                # Save each row as a separate resume entry
                save_document_to_db(fname, "Resume", rows_list, raw_text)
                out += f"<li>CSV {fname}: {len(df)} rows processed and saved to DB</li>"

            # --- IMAGE HANDLING ---
            elif fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                text = extract_text_from_image(path)
                category = classify_document(text)
                
                structured = extract_structured_data_from_text(text, category)
                
                save_document_to_db(fname, category, structured, text)
                out += f"<li>Image {fname}: Classified as {category} (Saved to DB)</li>"

            # --- PDF HANDLING ---
            elif fname.lower().endswith('.pdf'):
                text = extract_text_from_pdf(path)
                category = classify_document(text)
                structured = extract_structured_data_from_text(text, category)
                
                save_document_to_db(fname, category, structured, text)
                
                out += f"<li>PDF {fname}: {category} (Saved to DB) — <a href='/preview/{fname}'>Preview</a></li>"
            
            else:
                out += f"<li>{fname}: Uploaded (unsupported format for DB analysis)</li>"
                
        except Exception as e:
            out += f"<li>{fname}: Error - {e}</li>"

    out += "</ul>"
    return out, 200


@app.route('/results/<filename>')
def results_file(filename):
    """Results are no longer stored locally. All data is in the database."""
    return "Results not available. Data is stored in database.", 404


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/dashboard')
def dashboard():
    stats = {"total": 0, "by_type": {}}
    files = []
    
    # Fetch from MySQL database
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("SELECT filename, doc_type FROM documents ORDER BY created_at DESC LIMIT 100")
            rows = cur.fetchall()
            
            by_type = {}
            for filename, doc_type in rows:
                files.append(filename)
                by_type[doc_type] = by_type.get(doc_type, 0) + 1
            
            stats["total"] = len(files)
            stats["by_type"] = by_type
            
            cur.close()
            conn.close()
        except Exception:
            logging.exception('Failed to fetch dashboard stats from DB')
    
    return render_template('dashboard.html', stats=stats, files=files)


@app.route('/search')
def search():
    skill = request.args.get('skill', '').strip().lower()
    results = []
    seen_filenames = set()

    # 1. Fetch ALL documents from DB
    try:
        docs = get_all_documents(limit=1000)
        for d in docs:
            fn = d.get('filename')
            if fn: seen_filenames.add(fn)
            
            doc_type = (d.get('doc_type') or '').lower()
            if 'receipt' in doc_type or 'bill' in doc_type:
                continue

            data = d.get('data') or {}
            if isinstance(data, str):
                try: data = json.loads(data)
                except: data = {}
            
            name = data.get('name') or fn
            email = data.get('email') or ''
            skills = data.get('skills') or ''
            
            experience = data.get('experience') or data.get('years_experience') or ''
            if not experience:
                raw = d.get('raw_text') or ''
                m = re.search(r"(\d+)\+?\s+years", raw.lower())
                experience = f"{m.group(1)} years" if m else ''
            
            results.append({
                'name': name,
                'email': email,
                'skills': skills,
                'experience': experience,
                'source': 'db',
                'identifier': fn
            })
    except Exception:
        logging.exception('Failed to load documents from DB')

    # 2. Check local Uploads folder for missing files (Fallback)
    try:
        for f in os.listdir(UPLOAD_FOLDER):
            if f not in seen_filenames and f.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
                results.append({
                    'name': f, 
                    'email': '', 
                    'skills': 'Unprocessed (Not in DB)', 
                    'experience': '', 
                    'source': 'folder', 
                    'identifier': f
                })
    except Exception:
        pass

    ranking = []
    if skill:
        filtered = [
            r for r in results 
            if skill in (str(r.get('skills') or '')).lower() 
            or skill in (str(r.get('name') or '')).lower()
        ]
        ranking = rank_resumes(skill)
        results = filtered

    return render_template('search.html', results=results, error=None, ranking=ranking)


def get_context_from_db(user_query):
    q = user_query.strip()
    if not q:
        return ""

    rows = []
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            pattern = f"%{q}%"
            cur.execute("SELECT filename, doc_type, data, raw_text FROM documents WHERE raw_text LIKE %s OR data LIKE %s LIMIT 10", (pattern, pattern))
            for fn, dt, data_json, raw in cur.fetchall():
                rows.append((fn, dt, data_json, raw))
            cur.close()
            conn.close()
        except Exception:
            logging.exception('MySQL search failed')
            return ""
    
    if not rows:
        return ""

    entries = []
    for fn, dt, data_json, raw in rows:
        snippet = ''
        try:
            dd = json.loads(data_json) if data_json else {}
            if isinstance(dd, dict):
                if 'preview_lines' in dd and dd['preview_lines']:
                    snippet = dd['preview_lines'][0]
                else:
                    for k, v in dd.items():
                        if v:
                            snippet = str(v)
                            break
        except Exception:
            snippet = ''

        if not snippet and raw:
            snippet = (raw[:200] + '...') if len(raw) > 200 else raw

        entries.append(f"{fn} ({dt}) — {snippet}")

    return "\n".join(entries)


def count_documents_by_doc_type_like(substr):
    substr = substr.lower()
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            pattern = f"%{substr}%"
            cur.execute("SELECT COUNT(*) FROM documents WHERE LOWER(doc_type) LIKE %s", (pattern,))
            count = cur.fetchone()[0]
            cur.close()
            conn.close()
            return int(count)
        except Exception:
            logging.exception('MySQL count failed')
            return 0
    return 0


def get_documents_by_doc_type_like(substr):
    substr = substr.lower()
    docs = []
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            pattern = f"%{substr}%"
            cur.execute("SELECT filename, doc_type, data, raw_text FROM documents WHERE LOWER(doc_type) LIKE %s LIMIT 500", (pattern,))
            for fn, dt, data_json, raw in cur.fetchall():
                try:
                    data_val = json.loads(data_json) if data_json else {}
                except Exception:
                    data_val = data_json
                docs.append({'filename': fn, 'doc_type': dt, 'data': data_val, 'raw_text': raw})
            cur.close()
            conn.close()
            return docs
        except Exception:
            logging.exception('MySQL docs fetch failed')
            return []
    return []


def get_all_documents(limit=1000):
    docs = []
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("SELECT filename, doc_type, data, raw_text FROM documents LIMIT %s", (limit,))
            for fn, dt, data_json, raw in cur.fetchall():
                try:
                    data_val = json.loads(data_json) if data_json else {}
                except Exception:
                    data_val = data_json
                docs.append({'filename': fn, 'doc_type': dt, 'data': data_val, 'raw_text': raw})
            cur.close()
            conn.close()
            return docs
        except Exception:
            logging.exception('MySQL fetch all documents failed')
            return []
    return []


def rank_resumes(skill_query):
    if isinstance(skill_query, str):
        skills = [s.strip().lower() for s in re.split(r'[,\s]+', skill_query) if s.strip()]
    elif isinstance(skill_query, (list, tuple)):
        skills = [s.strip().lower() for s in skill_query if isinstance(s, str) and s.strip()]
    else:
        skills = []

    docs = get_documents_by_doc_type_like('resume')
    ranking = []
    for d in docs:
        data = d.get('data') or {}
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                data = {}
        name = (data.get('name') or d.get('filename') or 'Unknown')
        candidate_skills = []
        s = data.get('skills')
        if isinstance(s, list):
            candidate_skills = [x.strip().lower() for x in s if isinstance(x, str)]
        elif isinstance(s, str):
            candidate_skills = [x.strip().lower() for x in re.split(r'[;,]', s) if x.strip()]

        score = 0
        for skill in skills:
            if any(skill in cs for cs in candidate_skills):
                score += 1

        ranking.append((name, score))

    ranking.sort(key=lambda x: x[1], reverse=True)
    return ranking


def summarize_finances():
    docs = get_documents_by_doc_type_like('receipt') + get_documents_by_doc_type_like('bill')
    total = 0.0
    vendors = {}
    for d in docs:
        data = d.get('data') or {}
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                data = {}
        amount_raw = data.get('Total Amount') or data.get('total') or ''
        try:
            amt = float(re.sub(r"[^0-9.]", "", str(amount_raw)) or 0)
        except Exception:
            amt = 0.0
        vendor = data.get('Vendor') or data.get('vendor') or 'Unknown'
        total += amt
        vendors[vendor] = vendors.get(vendor, 0.0) + amt

    return total, vendors


def build_context(question):
    words = [w.strip().lower() for w in re.split(r"\s+", question) if w.strip()]
    if not words:
        return ""
    docs = get_all_documents(limit=500)
    context_pieces = []
    for d in docs:
        raw = (d.get('raw_text') or '')
        if not raw:
            continue
        low = raw.lower()
        if any(w in low for w in words):
            fn = d.get('filename') or 'unknown'
            dt = d.get('doc_type') or 'unknown'
            snippet = raw[:4000]
            piece = f"[{fn} - {dt}]\n{snippet}"
            context_pieces.append(piece)

    joined = "\n\n".join(context_pieces)
    return joined[:10000]


def ai_analyze_resume(filename, data, raw_text):
    """
    Detailed ATS analysis with 7 scoring categories and 5-8 specific recommendations.
    Categories: Contact Info (10 pts), Summary (5 pts), Skills (15 pts), Experience (20 pts),
                Education (8 pts), Formatting (10 pts), Keywords (7 pts) = 75 total -> normalized to 10
    """
    
    # Try LLM first
    prompt = (
        "You are an expert ATS specialist. Analyze this resume and score it 0-10 on ATS-friendliness. "
        "Provide 5-8 SPECIFIC, ACTIONABLE recommendations (each starting with emoji like '❌ CRITICAL:' or '✅ GOOD:' or '⚠️ IMPROVE:'). "
        "Also provide a sample ATS-friendly format. "
        "Return JSON: {\"score\": num, \"improvements\": [\"❌ item1\", \"✅ item2\", ...], \"sample\": \"formatted resume text\"}\n\n"
        f"FILENAME: {filename}\nDATA: {json.dumps(data)}\nRAW TEXT:\n{raw_text[:5000]}\n\nJSON:"
    )

    responders = []
    if OLLAMA_URL:
        for m in OLLAMA_MODELS:
            raw = call_ollama(prompt, model=m)
            responders.append((f'ollama-{m}', raw))

    if OPENAI_API_KEY:
        raw = call_openai(prompt)
        responders.append(('openai', raw))

    for provider, raw in responders:
        if not raw:
            continue
        try:
            start = raw.find('{')
            end = raw.rfind('}')
            if start != -1 and end != -1:
                snippet = raw[start:end+1]
                parsed = json.loads(snippet)
                score = float(parsed.get('score', 0))
                improvements = parsed.get('improvements', []) or []
                sample = parsed.get('sample', '') or parsed.get('resume', '')
                if improvements and isinstance(improvements[0], str):
                    return {
                        'score': min(max(round(score * 10 / 100), 0), 10) if score > 10 else min(max(round(score), 0), 10),
                        'improvements': improvements[:8],
                        'sample': sample,
                        'provider': provider
                    }
        except Exception:
            continue

    # Heuristic fallback with 7-category scoring
    try:
        scores = {}
        improvements = []
        
        # 1. CONTACT INFO (10 pts max)
        contact_score = 0
        email = data.get('email') or re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", raw_text)
        phone = data.get('phone') or re.search(r"\b(?:\+?1[-.]?)?(?:\(?[0-9]{3}\)?[-.]?)?[0-9]{3}[-.]?[0-9]{4}\b", raw_text)
        
        if email:
            contact_score += 5
        else:
            improvements.append("❌ CRITICAL: Add a professional email address at the top of resume")
        
        if phone:
            contact_score += 3
        else:
            improvements.append("⚠️ IMPROVE: Add phone number for easy contact")
        
        linkedin = re.search(r"linkedin\.com", raw_text, re.IGNORECASE)
        if linkedin:
            contact_score += 2
        else:
            improvements.append("⚠️ IMPROVE: Add LinkedIn profile URL for ATS systems")
        
        scores['Contact Info'] = min(10, contact_score)
        
        # 2. PROFESSIONAL SUMMARY (5 pts max)
        summary_score = 0
        summary_keywords = ['professional', 'skilled', 'experienced', 'expertise', 'proven', 'results']
        summary_text = raw_text[:1000]
        if any(kw in summary_text.lower() for kw in summary_keywords):
            summary_score = 5
        else:
            improvements.append("⚠️ IMPROVE: Add a professional summary highlighting key competencies")
        scores['Summary'] = summary_score
        
        # 3. SKILLS (15 pts max)
        skills = data.get('skills') or []
        if isinstance(skills, str):
            skills = [s.strip() for s in re.split(r'[,;]', skills) if s.strip()]
        skills_score = min(15, len(skills) * 2)
        if skills_score < 10:
            improvements.append("⚠️ IMPROVE: List at least 5-10 relevant technical and soft skills")
        scores['Skills'] = skills_score
        
        # 4. EXPERIENCE (20 pts max)
        exp_score = 0
        years = data.get('years_experience') or 0
        if isinstance(years, str):
            m = re.search(r"(\d+)", years)
            years = int(m.group(1)) if m else 0
        
        if years >= 5:
            exp_score = 20
        elif years >= 2:
            exp_score = 15
        elif years > 0:
            exp_score = 10
        else:
            improvements.append("❌ CRITICAL: Clearly state years of professional experience (e.g., '5+ years')")
        
        # Check for job descriptions with metrics
        job_keywords = ['responsibility', 'achievement', 'led', 'managed', 'developed', 'improved']
        if any(kw in raw_text.lower() for kw in job_keywords):
            exp_score = min(20, exp_score + 5)
        else:
            improvements.append("⚠️ IMPROVE: Use action verbs (led, managed, developed) and quantify achievements")
        
        scores['Experience'] = exp_score
        
        # 5. EDUCATION (8 pts max)
        edu_score = 0
        edu_keywords = ['bachelor', 'master', 'phd', 'b.s', 'b.a', 'm.s', 'm.b.a', 'diploma', 'university', 'college']
        if any(kw in raw_text.lower() for kw in edu_keywords):
            edu_score = 8
        else:
            improvements.append("⚠️ IMPROVE: Include formal education (degree, institution, graduation year)")
        scores['Education'] = edu_score
        
        # 6. FORMATTING (10 pts max)
        format_score = 10  # Assume good formatting if we got this far
        if len(raw_text) > 5000:
            format_score = 5
            improvements.append("⚠️ IMPROVE: Keep resume to 1-2 pages; use clear section headers and bullet points")
        if '\n' not in raw_text[:500]:
            format_score = max(0, format_score - 3)
            improvements.append("⚠️ IMPROVE: Use clear line breaks and section headers for ATS parsing")
        scores['Formatting'] = format_score
        
        # 7. KEYWORDS (7 pts max)
        keyword_score = 0
        tech_keywords = ['python', 'java', 'sql', 'aws', 'project', 'team', 'communication', 'problem-solving']
        matched = sum(1 for kw in tech_keywords if kw in raw_text.lower())
        keyword_score = min(7, matched)
        if keyword_score < 4:
            improvements.append("⚠️ IMPROVE: Include industry-specific keywords matching job descriptions")
        scores['Keywords'] = keyword_score
        
        # Calculate total score (0-10 scale)
        total = sum(scores.values())
        max_score = 75
        final_score = min(10, round((total / max_score) * 10, 1))
        
        # Ensure at least 5-8 recommendations
        while len(improvements) < 5 and len(improvements) < 8:
            if 'formatting' not in str(improvements).lower():
                improvements.append("✅ GOOD: Resume is well-structured and easy to parse")
            else:
                improvements.append("✅ GOOD: Relevant experience and skills highlighted")
            if len(improvements) >= 8:
                break
        
        sample = generate_ats_sample(data, skills)
        return {
            'score': int(final_score),
            'improvements': improvements[:8],
            'sample': sample,
            'provider': 'heuristic',
            'scores': scores
        }
    except Exception as e:
        logging.exception(f"ATS analysis error: {e}")
        return {
            'score': 0,
            'improvements': ["Error analyzing resume"],
            'sample': '',
            'provider': 'heuristic-error'
        }

def generate_ats_sample(data, skills):
    """Generate a sample ATS-friendly resume format"""
    name = data.get('name', 'Your Name')
    email = data.get('email', 'your.email@example.com')
    phone = data.get('phone', '(123) 456-7890')
    
    sample = f"""
{name}
{email} | {phone} | linkedin.com/in/yourprofile

PROFESSIONAL SUMMARY
Results-driven professional with proven expertise in {', '.join(skills[:3]) if skills else 'relevant skills'}. 
Demonstrated ability to drive successful projects and lead teams.

SKILLS
{', '.join(skills[:8]) if skills else 'Technical and soft skills'}

PROFESSIONAL EXPERIENCE
[Company Name] | [Job Title] | [Dates]
- Achievement or responsibility with quantifiable impact
- Additional responsibility demonstrating expertise
- Success metric or contribution

EDUCATION
[Degree] | [University] | [Graduation Year]

CERTIFICATIONS & AWARDS
[Relevant certifications and professional achievements]
"""
    return sample.strip()

def count_documents_by_extension(ext):
    try:
        files = os.listdir(UPLOAD_FOLDER)
        return sum(1 for f in files if f.lower().endswith(ext))
    except:
        return 0



@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    user_query = (data.get('message') or '').strip()
    uq = user_query.lower()

    if "how many" in uq:
        if any(x in uq for x in ["pdf", "pdfs"]):
            count = count_documents_by_extension('.pdf')
            return {"response": f"There are {count} PDF file(s) in the uploads folder."}
        if any(x in uq for x in ["csv", "csvs"]):
            count = count_documents_by_extension('.csv')
            return {"response": f"There are {count} CSV file(s) in the uploads folder."}
        if any(x in uq for x in ["jpg", "jpeg", "png", "images"]):
            c = count_documents_by_extension('.png') + count_documents_by_extension('.jpg') + count_documents_by_extension('.jpeg')
            return {"response": f"There are {c} image file(s) in the uploads folder."}
        if "file" in uq or "files" in uq:
            try:
                files = os.listdir(UPLOAD_FOLDER)
                return {"response": f"There are {len(files)} file(s) in the uploads folder."}
            except Exception:
                return {"response": "Could not access the uploads folder."}

    if any(p in uq for p in ["list files", "show files", "what files", "recent uploads", "show uploads"]):
        try:
            files = sorted(os.listdir(UPLOAD_FOLDER), reverse=True)[:50]
            if not files:
                return {"response": "No files found in the uploads folder."}
            return {"response": "Files:\n" + "\n".join(files)}
        except Exception:
            return {"response": "Could not list uploads folder."}

    if "resume" in uq and ("how many" in uq or "count" in uq):
        count = count_documents_by_doc_type_like('resume')
        return {"response": f"There are {count} resume(s) uploaded."}

    if ("any resume" in uq) or ("is there" in uq and "resume" in uq):
        count = count_documents_by_doc_type_like('resume')
        return {"response": "Yes, at least one resume is uploaded." if count else "No resumes have been uploaded yet."}

    if "ats" in uq and ("rank" in uq or "compare" in uq or "score" in uq or "best" in uq):
        docs = get_documents_by_doc_type_like('resume')
        if not docs:
            # Fallback: check CSVs if no PDF/Image resumes found
            try:
                for f in os.listdir(UPLOAD_FOLDER):
                    if f.lower().endswith('.csv'):
                         docs.append({'filename': f, 'data': {'name': f}, 'raw_text': ''})
            except: pass
            
        if not docs:
            return {"response": "No resumes found in the database to compare."}
        
        ranked_list = []
        processing_docs = docs[:10]
        
        for d in processing_docs:
            fn = d.get('filename')
            data_val = d.get('data') or {}
            raw = d.get('raw_text') or ''
            
            if isinstance(data_val, str):
                try: data_val = json.loads(data_val)
                except: data_val = {}
                
            analysis = ai_analyze_resume(fn, data_val, raw)
            score = analysis.get('score', 0)
            
            name = data_val.get('name') or fn
            ranked_list.append((name, score))
            
        ranked_list.sort(key=lambda x: x[1], reverse=True)
        
        response_text = "Here is the ATS Score Ranking for your uploaded resumes:\n\n"
        for i, (name, score) in enumerate(ranked_list, 1):
            medal = "🥇" if i==1 else ("🥈" if i==2 else ("🥉" if i==3 else f"{i}."))
            response_text += f"{medal} {name}: {score}/10\n"
            
        response_text += "\nTip: Click 'View Analysis' on the Search page for detailed improvements."
        return {"response": response_text}

    m = re.search(r"(what|which)\s+skills\s+does\s+([\w\-\.' ]{2,100})\s+have\??", user_query, flags=re.I)
    if m:
        name = m.group(2).strip()
        docs = get_documents_by_doc_type_like('resume')
        skills_found = []
        for d in docs:
            data_val = d.get('data') or {}
            if isinstance(data_val, str):
                try:
                    data_val = json.loads(data_val)
                except Exception:
                    data_val = {}
            try:
                if name.lower() in (data_val.get('name', '') or '').lower() or name.lower() in (d.get('filename') or '').lower():
                    skills = data_val.get('skills', [])
                    if isinstance(skills, list):
                        skills_found.extend(skills)
                    elif isinstance(skills, str) and skills:
                        skills_found.extend([s.strip() for s in re.split(r"[,;\\|]", skills) if s.strip()])
            except Exception:
                pass

        if skills_found:
            return {"response": f"{name.title()}'s skills: {', '.join(sorted(set(skills_found)))}"}
        return {"response": f"No skills found for {name.title()}."}

    m2 = re.search(r"(?:rank|who is best|best candidate|top candidates) (?:for )?(.+)", uq)
    if m2 and any(ch.isalpha() for ch in m2.group(1)):
        skill_query = m2.group(1)
        ranking = rank_resumes(skill_query)
        if not ranking:
            return {"response": "No resumes available to rank."}
        lines = [f"{name}: {score}" for name, score in ranking[:10]]
        return {"response": "Top candidates:\n" + "\n".join(lines)}

    if any(p in uq for p in ["summarize finances", "summarize receipts", "total spent", "expense summary"]):
        total, vendors = summarize_finances()
        top_vendors = sorted(vendors.items(), key=lambda x: x[1], reverse=True)[:10]
        vendor_lines = [f"{v}: {amt:.2f}" for v, amt in top_vendors]
        resp = f"Total from receipts: {total:.2f}\nTop vendors:\n" + "\n".join(vendor_lines) if total else "No receipt data found."
        return {"response": resp}

    m3 = re.search(r"(?:preview|show preview|show details|details for|open)\s+([\w\-\./ ]+\.[a-zA-Z0-9]{1,5})", user_query, flags=re.I)
    if m3:
        fname = m3.group(1).strip()
        path = os.path.join(UPLOAD_FOLDER, fname)
        if os.path.exists(path):
            snippet = ''
            try:
                if fname.lower().endswith('.csv'):
                    df = pd.read_csv(path)
                    snippet = df.head(5).to_string(index=False)
                elif fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    snippet = extract_text_from_image(path)[:800]
                else:
                    snippet = extract_text_from_pdf(path)[:800]
            except Exception:
                snippet = ''
            url = f"/preview/{fname}"
            return {"response": f"Preview for {fname}: {url}\n\n{snippet}"}
        
        docs = get_all_documents(limit=200)
        for d in docs:
            if (d.get('filename') or '').lower() == fname.lower():
                data = d.get('data') or {}
                raw = (d.get('raw_text') or '')[:800]
                return {"response": f"Document {fname} found in DB.\nData: {json.dumps(data)[:1000]}\n\nSnippet:\n{raw}"}
        return {"response": f"File {fname} not found in uploads or DB."}

    if "itr" in uq and "year" in uq:
        docs = get_documents_by_doc_type_like('income')
        years = set()
        for d in docs:
            text = d.get('raw_text') or ''
            matches = re.findall(r"(20\d{2})", text)
            years.update(matches)
        if years:
            return {"response": f"ITR document(s) found from year(s): {', '.join(sorted(years))}"}
        return {"response": "ITR documents found but year could not be detected."}

    context = get_context_from_db(user_query)
    if context:
        if any(w in uq for w in ["why", "explain", "recommend", "who", "which", "suggest"]):
            answer, details = council_call(user_query, context=build_context(user_query))
            return {"response": answer or ("I found these relevant documents:\n" + context), "details": details}
        return {"response": "I found these relevant documents:\n" + context}

    ctx = build_context(user_query)
    if ctx:
        answer, details = council_call(user_query, context=ctx)
        return {"response": answer or "I couldn't produce a confident answer.", "details": details}

    return {"response": "I couldn't find matching documents or a direct answer. Try asking about files, resumes, receipts, or give more details."}


@app.route('/resume_preview')
def resume_preview():
    name = request.args.get('name', '').strip()
    if not name:
        return jsonify({'error': 'missing name'}), 400

    try:
        docs = get_documents_by_doc_type_like('resume')
        for d in docs:
            fn = (d.get('filename') or '')
            data = d.get('data') or {}
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except Exception:
                    data = {}
            candidate_name = (data.get('name') or '')
            if fn.lower() == name.lower() or candidate_name.lower() == name.lower():
                analysis = ai_analyze_resume(fn, data, d.get('raw_text') or '')
                return jsonify({'found_in': 'db', 'filename': fn, 'data': data, 'analysis': analysis})
    except Exception:
        logging.exception('Error searching DB resumes')

    try:
        for f in os.listdir(UPLOAD_FOLDER):
            if not f.lower().endswith('.csv'):
                continue
            df = pd.read_csv(os.path.join(UPLOAD_FOLDER, f))
            name_col = None
            for c in df.columns:
                if 'name' in c.lower():
                    name_col = c; break
            if not name_col:
                continue
            for _, row in df.iterrows():
                if name.lower() == str(row.get(name_col,'')).strip().lower():
                    data = {c: str(row.get(c,'')) for c in df.columns}
                    raw = json.dumps(data)
                    analysis = ai_analyze_resume(f, data, raw)
                    return jsonify({'found_in': 'csv', 'filename': f, 'data': data, 'analysis': analysis})
    except Exception:
        logging.exception('Error searching CSV resumes')

    # Fallback: check raw files in upload folder
    path = os.path.join(UPLOAD_FOLDER, name)
    if os.path.exists(path):
        text = ""
        if name.lower().endswith(('.png','.jpg','.jpeg')):
             text = extract_text_from_image(path)
        elif name.lower().endswith('.pdf'):
             text = extract_text_from_pdf(path)
        
        data = {'name': name, 'skills': '', 'email': ''} 
        analysis = ai_analyze_resume(name, data, text)
        return jsonify({'found_in': 'folder_fallback', 'filename': name, 'data': data, 'analysis': analysis})

    return jsonify({'error': 'not found'}), 404

if __name__ == "__main__":
    app.run(debug=True)
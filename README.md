# 📄 DataHub: Intelligent Document Processor & ATS Analyzer

**DataHub** is a Flask-based application designed to bridge the gap between static documents and actionable data. It functions as a versatile document processor that supports **CSV-to-Database ETL**, **Resume Parsing with ATS Scoring**, and an **AI-powered Chatbot** to interact with your uploaded files.

Built with Python, it leverages OCR (Tesseract), PDF processing, and Large Language Models (Ollama/OpenAI) to extract, analyze, and query information.

---

## 🚀 Key Features

### 1. 📂 Universal File Upload & ETL
* **CSV Processing:** Automatically reads CSV uploads, infers data types, and inserts them into a MySQL database.
* **Document Storage:** Handles PDF and Image uploads (PNG/JPG), storing metadata and raw text content for search and retrieval.

### 2. 🤖 Resume ATS Analyzer
* **Smart Extraction:** Automatically extracts key fields like **Name**, **Email**, **Phone**, **Skills**, and **Experience** from Resumes (PDF/Image) using Regex and OCR.
* **ATS Scoring:** Provides an automated "ATS Score" (0-10) based on heuristic analysis (keyword density, formatting, contact info).
* **Detailed Feedback:** Generates actionable improvements and sample formats to help candidates optimize their CVs.
* **Comparison Tool:** Rank and compare multiple candidates based on specific skills (e.g., "Python", "SQL").

### 3. 💬 AI Chat with Documents
* **Context-Aware Q&A:** Chat with your documents! The system retrieves relevant text chunks from your uploads to answer questions like *"How many resumes list Python?"* or *"Summarize the expenses in the uploaded receipts."*
* **LLM Integration:** Supports local **Ollama** models (Mistral, Phi3) and **OpenAI** (GPT-3.5/4) for high-quality reasoning.

### 4. 📊 Dashboard & Search
* **Visual Stats:** View charts of document distribution (PDF vs CSV vs Images).
* **Advanced Search:** Filter database records by specific skills or keywords.

---

## 🛠️ Tech Stack

* **Backend:** Flask (Python)
* **Database:** MySQL (via `mysql-connector-python`)
* **Data Processing:** Pandas, NumPy
* **OCR & Parsing:**
    * `pytesseract` (Tesseract OCR Engine)
    * `pdf2image` & `Poppler`
    * `PyPDF2`
    * `OpenCV` (`opencv-python`)
* **AI/LLM:** OpenAI API, Ollama (Local LLM)

---

## ⚙️ Prerequisites

Before installing the Python packages, ensure you have the following system-level tools installed. These are **required** for parsing PDFs and Images.

### 1. Python 3.8+ & MySQL
Ensure Python and a running MySQL server are available.

### 2. Tesseract OCR (For Image Text Extraction)
* **Windows:** Download the [UB-Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki). Add the installation path (e.g., `C:\Program Files\Tesseract-OCR`) to your System PATH.
* **Linux:** `sudo apt-get install tesseract-ocr`
* **macOS:** `brew install tesseract`

### 3. Poppler (For PDF to Image conversion)
* **Windows:** Download the [latest binary](https://github.com/oschwartz10612/poppler-windows/releases/), extract it, and add the `bin` folder to your System PATH.
* **Linux:** `sudo apt-get install poppler-utils`
* **macOS:** `brew install poppler`

---

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/datahub-ats.git](https://github.com/your-username/datahub-ats.git)
    cd datahub-ats
    ```

2.  **Set up Virtual Environment**
    ```powershell
    # Windows (PowerShell)
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

    # Linux/macOS
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🔧 Configuration

### 1. Database Setup
Open `app.py` and configure the `DB_CONFIG` dictionary with your MySQL credentials.
*Note: For production, it is recommended to use environment variables instead of hardcoding credentials.*

```python
# app.py
DB_CONFIG = {
    'user': 'your_db_user',      # e.g. root
    'password': 'your_password', # e.g. 12345678
    'host': 'your_host',         # e.g. localhost
    'port': '3306',
    'database': 'your_db_name',
    'raise_on_warnings': False
}
2. LLM Configuration (Optional)
To enable the Chatbot and advanced AI features, set the following environment variables.

For Local LLM (Ollama):

PowerShell

# Windows PowerShell
$env:OLLAMA_URL = "http://localhost:11434/api/generate"
$env:LLM_MODEL = "mistral" # or phi3
For OpenAI:

PowerShell

$env:OPENAI_API_KEY = "sk-your-openai-api-key"
▶️ Usage
Run the Application

Bash

python app.py
Access the Web UI Open your browser and navigate to http://127.0.0.1:5000/.

Features Guide:

Upload: Go to the Home page to upload CSVs, PDFs (Resumes/Receipts), or Images.

Dashboard: View file statistics and download processed files.

Search Resumes: Go to the "Search" tab. Enter a skill (e.g., "Python") to rank candidates. Click "Analyze" to see the ATS score and improvements.

Ask AI: Use the chatbot widget in the bottom right corner to ask questions about your data (e.g., "Compare ATS scores of all candidates").

📂 Project Structure
Plaintext

csv-to-db/
├── app.py                 # Main Flask application logic
├── requirements.txt       # Python dependencies
├── etl_log.txt            # Log file for operations
├── uploads/               # Directory for uploaded raw files
├── results/               # Directory for generated reports/PDFs
├── templates/             # HTML Templates
│   ├── uploads.html       # Home/Upload page
│   ├── dashboard.html     # Analytics dashboard
│   └── search.html        # Resume search & ATS interface
└── static/
    ├── styles.css         # CSS Styling
    └── script.js          # JavaScript for Chat & UI
📝 License
This project is open-source and available under the MIT License.
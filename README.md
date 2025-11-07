# Smart Study Companion

This repository contains a Streamlit app that predicts likely exam topics by analyzing past-year question papers and an optional syllabus using TF-IDF and trend weighting.

## Files in this package
- `app.py` - Streamlit application (main file)
- `requirements.txt` - Python dependencies
- `sample_docs/` - small set of example .txt files to demo the app
- `README.md` - this file

## How to run locally
1. Create a Python environment and install dependencies:
```bash
pip install -r requirements.txt
```

2. Run Streamlit:
```bash
streamlit run app.py
```

3. Open the URL shown by Streamlit (usually http://localhost:8501). Upload your past question papers (PDF or TXT) and an optional syllabus, then press **Analyze**.

## Deploy to Streamlit Cloud
1. Create a new GitHub repo with these files.
2. In Streamlit Cloud, link your GitHub repo and deploy (Streamlit will install `requirements.txt`).

## Notes
- For better results, upload at least 3-5 past papers and include year information in filenames (e.g., `paper_2019.pdf`).
- If PDFs extract poorly, convert to text before upload or use OCR.
- This package is prepared for ANURAG SAINI THE BAKU — customize the UI or scoring logic as desired.
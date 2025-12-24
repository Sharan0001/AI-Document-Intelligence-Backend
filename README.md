# AI Document Intelligence Backend

An AI-powered backend service for analyzing invoices and contracts.

## Features
- Automatic document type detection (invoice / contract)
- OCR fallback for scanned PDFs
- Field extraction (dates, amounts, parties)
- Risk assessment with severity and score
- Clause detection (rule-based + semantic)
- Explainability endpoints
- SQLite persistence
- Metrics and error analysis APIs

## Tech Stack
- FastAPI
- Python
- SQLite
- Sentence Transformers
- OCR (Tesseract)

## Running Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload

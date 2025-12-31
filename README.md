📄 AI Document Intelligence Platform

An end-to-end AI system for **automated invoice and contract analysis**, delivering structured extraction, risk assessment, clause intelligence, and explainability through a modern full-stack architecture.

This project transforms unstructured business documents into **actionable intelligence**.

---

🧩 Why This Project Matters

Organizations deal with large volumes of documents such as:

- Invoices  
- Vendor contracts  
- Legal agreements  

Manual review is:

- Time-consuming  
- Error-prone  
- Inconsistent  
- Hard to scale  

Missed clauses, unclear payment terms, or risky contract conditions can lead to **financial loss, legal exposure, and operational risk**.

This system enables:

- Automated document understanding  
- Consistent risk detection  
- Explainable AI decisions  
- Centralized document intelligence  

Designed for **finance teams, legal reviewers, auditors, and compliance workflows**.

---

💡 Key Features

📑 Automatic Document Understanding

- Auto-detects document type:
  - Invoice
  - Contract
- Supports both:
  - Digital PDFs
  - Scanned PDFs (OCR fallback)

---

🧾 Structured Field Extraction

**Invoices**
- Invoice number
- Vendor
- Invoice date
- Due date
- Amount
- Currency

**Contracts**
- Effective date
- End date
- Governing law
- Payment terms
- Key contractual metadata

---

⚠️ Risk Assessment & Scoring

- Document-level risk score
- Severity classification:
  - Low
  - Medium
  - High
- Rule-based + semantic risk detection
- Deterministic and explainable logic

---

📚 Clause Intelligence

- Clause detection using:
  - Rule-based patterns
  - Semantic similarity (Sentence Transformers)
- Clause condensation into core types:
  - Termination
  - Payment
  - Liability
  - Governing law
- Highlighted risky clauses

---

🧠 Explainability (“Why is this risky?”)

- Clear explanation of triggered risk flags
- Evidence-based reasoning
- UI-level mapping for:
  - Human-readable titles
  - Evidence fields
  - Clause focus

No black-box decisions.

---

🗂️ Persistent Document Store

- SQLite-backed storage
- View previously analyzed documents
- Reload full document analysis by ID
- Delete individual or all documents

---

📊 System Insights & Monitoring

- Metrics dashboard:
  - Document counts
  - Risk distribution
- Error analysis endpoint
- Operational visibility into the system

---

🛠️ Tech Stack

| Layer        | Technology |
|-------------|------------|
| Frontend    | React + Vite |
| Backend     | FastAPI |
| AI / NLP    | Sentence Transformers |
| OCR         | Tesseract |
| Database    | SQLite |
| Deployment  | Vercel (Frontend), Hugging Face Spaces (Backend) |
| Language    | Python, TypeScript |

---

🚀 How It Works

1. User uploads a PDF document  
2. Backend auto-classifies document type  
3. OCR is applied if needed  
4. Fields and clauses are extracted  
5. Risk rules and semantic analysis run  
6. Explainability logic generates insights  
7. Results are persisted and visualized  

---

📐 Architecture

```

PDF Input
↓
Document Classifier
↓
OCR (if needed)
↓
Field & Clause Extraction
↓
Risk Assessment Engine
↓
Explainability Layer
↓
FastAPI Backend
↓
React Frontend (Dashboard)

````

Frontend and backend are **fully decoupled** and communicate via a clean API.

---

📦 Installation (Local Development)

1️⃣ Clone repositories

```bash
git clone https://github.com/Sharan0001/AI-Document-Intelligence-Backend.git
git clone https://github.com/Sharan0001/AI-Document-Intelligence-Frontend.git
````

---

2️⃣ Backend setup

```bash
cd AI-Document-Intelligence-Backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend runs at:

```
http://localhost:8000
```

---

3️⃣ Frontend setup

```bash
cd AI-Document-Intelligence-Frontend
npm install
npm run dev
```

Frontend runs at:

```
http://localhost:5173
```

---

🌐 Deployment

* **Backend**: Hugging Face Spaces (Docker + FastAPI)
* **Frontend**: Vercel

The same API contract is used locally and in production.

---

✨ Example Output

* Risk severity: **High**
* Risk score: **78**
* Flag detected: *Missing payment terms*
* Clause highlighted: *PAYMENT*
* Explainability: *“Payment obligations are unclear”*

---

🧾 Business Value

This system enables organizations to:

* Reduce document review time
* Improve risk detection consistency
* Standardize contract & invoice analysis
* Support audit and compliance workflows
* Build explainable AI into decision processes

Potential extensions:

* Enterprise document pipelines
* Contract lifecycle management
* Finance automation tools
* Compliance dashboards

---

🎯 What This Project Demonstrates

* Full-stack AI system design
* Applied NLP & document intelligence
* Explainable AI principles
* API-first architecture
* Production-grade deployment
* Debugging and infra decision-making

Not just an ML model — a **deployable document intelligence product**.

---

🔮 Roadmap

* Batch document uploads
* Export reports (PDF / JSON)
* Authentication & role-based access
* Advanced analytics dashboards
* Async processing for large workloads

---

👤 Author

**Sharan**
B.Tech Artificial Intelligence & Data Science

Built as a production-style system integrating:

**AI → Risk Intelligence → Explainability → Product UI**

Connect:

* LinkedIn: [https://www.linkedin.com/in/sharan-v-188065257](https://www.linkedin.com/in/sharan-v-188065257)
* GitHub: [https://github.com/Sharan0001](https://github.com/Sharan0001)

---

⭐ Support

If you find this project useful, consider starring the repository to support continued development.

---

📢 Final Thoughts

This project reflects:

* Strong engineering fundamentals
* Practical AI deployment experience
* Product-oriented thinking
* Real-world problem solving

A solid demonstration of **AI applied beyond notebooks**.

```

from dateutil import parser as date_parser
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pdfplumber
import io
import os
from datetime import datetime, date
from typing import Optional, Dict, Any, List
import re
import sqlite3
import json
import hashlib
import numpy as np
from collections import defaultdict

# -------------------------------------------------------------------
# Optional OCR imports (graceful fallback if not installed)
# -------------------------------------------------------------------
try:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

OCR_AVAILABLE = (
    os.getenv("OCR_ENABLED", "1") == "1"
) and OCR_AVAILABLE

# -------------------------------------------------------------------
# Optional semantic embeddings (sentence-transformers)
# -------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer  # type: ignore

    EMBEDDING_MODEL_AVAILABLE = True
except ImportError:
    EMBEDDING_MODEL_AVAILABLE = False

EMB_MODEL = None  # lazy-loaded SentenceTransformer
CLAUSE_PROTOTYPE_EMBEDDINGS: Dict[str, np.ndarray] = {}

# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------
app = FastAPI(
    title="AI Document Intelligence – Invoice/Contract Analyzer",
    description="API-first backend for extracting key fields & risks from invoices and contracts.",
    version="0.1.0"
)

ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*").split(",")

# Allow local dev / frontend later
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# SQLite – simple document store (with content_hash + summary + clauses)
# -------------------------------------------------------------------
DB_PATH = os.getenv("DB_PATH", "documents.db")

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.row_factory = sqlite3.Row

with conn:
    # Base table (for fresh DBs)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            file_name TEXT,
            extracted_fields TEXT NOT NULL,
            raw_preview TEXT NOT NULL,
            risk_assessment TEXT,
            content_hash TEXT,
            summary TEXT,
            clauses TEXT
        )
        """
    )
    # For older DBs that might be missing columns, try adding them safely
    for col in ["content_hash", "summary", "clauses"]:
        try:
            conn.execute(f"ALTER TABLE documents ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError:
            # Column already exists – ignore
            pass

    # Unique index for dedupe on content_hash
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_content_hash "
        "ON documents(content_hash)"
    )


def compute_doc_hash(doc_type: str, raw_text: str) -> str:
    """
    Stable content hash so we can detect duplicate uploads.
    Uses doc_type + full extracted raw_text.
    """
    h = hashlib.sha256()
    h.update(doc_type.encode("utf-8"))
    h.update(b"::")
    h.update(raw_text.encode("utf-8", errors="ignore"))
    return h.hexdigest()


# -------------------------------------------------------------------
# Pydantic Schemas
# -------------------------------------------------------------------

class InvoiceFields(BaseModel):
    invoice_number: Optional[str] = None
    vendor_name: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    total_amount: Optional[float] = None
    currency: Optional[str] = None
    raw_text_sample: Optional[str] = None


class ContractFields(BaseModel):
    party_a: Optional[str] = None
    party_b: Optional[str] = None
    effective_date: Optional[str] = None
    end_date: Optional[str] = None
    governing_law: Optional[str] = None
    payment_terms: Optional[str] = None
    raw_text_sample: Optional[str] = None


class RiskAssessment(BaseModel):
    flags: List[str]
    severity: str  # "low" | "medium" | "high"
    overall_score: float  # 0.0–1.0
    notes: Optional[str] = None


class Clause(BaseModel):
    type: str
    confidence: float
    text: str       # FULL clause text
    preview: str    # SHORT preview for UI


class ExtractionResponse(BaseModel):
    doc_type: str
    extracted_fields: Dict[str, Any]
    raw_preview: str
    risk_assessment: Optional[RiskAssessment] = None
    document_id: Optional[int] = None
    summary: Optional[str] = None  # human-readable summary
    clauses: Optional[List[Clause]] = None  # detected contract clauses (if any)


class DocumentSummary(BaseModel):
    id: int
    doc_type: str
    created_at: str
    file_name: Optional[str] = None
    risk_severity: Optional[str] = None
    risk_flags: List[str] = []
    overall_score: Optional[float] = None
    summary: Optional[str] = None  # quick glance summary


# -------------------------------------------------------------------
# DB helpers
# -------------------------------------------------------------------

def save_document(
    doc_type: str,
    file_name: Optional[str],
    extracted_fields: Dict[str, Any],
    raw_preview: str,
    risk_assessment: Optional["RiskAssessment"],
    content_hash: Optional[str],
    summary: Optional[str],
    clauses: Optional[List[Clause]],
) -> int:
    """
    Persist a document + analysis and return its ID.

    Dedup strategy:
      1) If content_hash matches → reuse that row.
      2) Else if (doc_type, raw_preview) matches → reuse that row.
      3) Else insert new row.
    """
    with conn:
        # 1) Hash-based dedupe
        if content_hash:
            existing = conn.execute(
                "SELECT id FROM documents WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
            if existing:
                return existing["id"]

        # 2) Raw-preview-based dedupe (also works for old rows without hash)
        existing2 = conn.execute(
            "SELECT id FROM documents WHERE doc_type = ? AND raw_preview = ?",
            (doc_type, raw_preview),
        ).fetchone()
        if existing2:
            return existing2["id"]

        # 3) Insert new row
        cur = conn.execute(
            """
            INSERT INTO documents
            (doc_type, created_at, file_name, extracted_fields, raw_preview,
             risk_assessment, content_hash, summary, clauses)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_type,
                datetime.utcnow().isoformat(),
                file_name,
                json.dumps(extracted_fields),
                raw_preview,
                json.dumps(risk_assessment.dict()) if risk_assessment else None,
                content_hash,
                summary,
                json.dumps([c.dict() for c in clauses]) if clauses else None,
            ),
        )
        return cur.lastrowid


def load_document(doc_id: int) -> Optional[ExtractionResponse]:
    """Fetch a stored document by ID and map back to ExtractionResponse."""
    row = conn.execute(
        "SELECT * FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()

    if not row:
        return None

    extracted_fields = json.loads(row["extracted_fields"])
    risk_raw = row["risk_assessment"]
    risk_obj = None
    if risk_raw:
        risk_obj = RiskAssessment(**json.loads(risk_raw))

    clauses_raw = row["clauses"]
    clauses: Optional[List[Clause]] = None
    if clauses_raw:
        try:
            parsed = json.loads(clauses_raw)
            clauses = [Clause(**c) for c in parsed]
        except Exception:
            clauses = None

    return ExtractionResponse(
        doc_type=row["doc_type"],
        extracted_fields=extracted_fields,
        raw_preview=row["raw_preview"],
        risk_assessment=risk_obj,
        document_id=row["id"],
        summary=row["summary"],
        clauses=clauses,
    )


# -------------------------------------------------------------------
# Utility: PDF to text (now with OCR fallback)
# -------------------------------------------------------------------

def pdf_to_text_basic(file_bytes: bytes) -> str:
    """
    Existing simple text extractor:
    - Works great for digital/text-based PDFs.
    - Returns empty string for image-only / scanned PDFs.
    """
    text_chunks = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_chunks.append(page_text)
    return "\n".join(text_chunks)


def is_mostly_empty(text: str, min_len: int = 30) -> bool:
    """
    Heuristic: treat the PDF as 'no text' if the extracted
    text is very short or almost whitespace.
    """
    if not text:
        return True
    stripped = text.strip()
    return len(stripped) < min_len


def ocr_pdf_bytes(file_bytes: bytes) -> str:
    """
    OCR fallback for scanned PDFs.

    Uses:
      - pdfplumber to open pages
      - page.to_image(...).original → PIL.Image
      - pytesseract.image_to_string for OCR

    If OCR is not available or rendering fails, returns "".
    """
    if not OCR_AVAILABLE:
        return ""

    try:
        texts: List[str] = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                try:
                    page_image = page.to_image(resolution=300).original
                except Exception:
                    continue

                try:
                    page_text = pytesseract.image_to_string(page_image)
                except Exception:
                    page_text = ""
                texts.append(page_text)

        return "\n".join(texts)
    except Exception:
        return ""


def pdf_to_text(file_bytes: bytes) -> str:
    """
    Smart PDF-to-text pipeline:

    1) Try native text extraction (pdfplumber).
    2) If result looks empty/too short → try OCR on each page.
    3) If OCR fails or not available → return whatever we have.
    """
    base_text = pdf_to_text_basic(file_bytes)

    if not is_mostly_empty(base_text):
        return base_text

    ocr_text = ocr_pdf_bytes(file_bytes)
    if not is_mostly_empty(ocr_text):
        return ocr_text

    return base_text


# -------------------------------------------------------------------
# Date parsing + whitespace normalization
# -------------------------------------------------------------------

def parse_possible_date(text: str) -> Optional[str]:
    """
    Try to parse a human-readable date string into ISO format,
    biasing towards any explicit 4-digit year present.
    """
    try:
        year_match = re.search(r"(19|20)\d{2}", text)
        if year_match:
            year = int(year_match.group(0))
            dt = date_parser.parse(text, fuzzy=True, default=datetime(year, 1, 1))
        else:
            dt = date_parser.parse(text, fuzzy=True)

        # Year sanity bounds: reject dates outside reasonable range
        current_year = datetime.utcnow().year
        parsed_year = dt.year
        if parsed_year < 1990 or parsed_year > (current_year + 5):
            return None

        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def normalize_whitespace(value: Optional[str]) -> Optional[str]:
    """
    Collapse internal whitespace (including newlines) to single spaces
    and strip ends. Keeps None as None.
    """
    if value is None:
        return None
    return re.sub(r"\s+", " ", value).strip()


# -------------------------------------------------------------------
# Tiny doc_type classifier (heuristic)
# -------------------------------------------------------------------

def classify_doc_type_from_text(raw_text: str) -> str:
    """
    Robust heuristic doc type classifier.

    Priority:
    1) Strong contract signals (override everything)
    2) Strong invoice signals
    3) Structural fallback
    """
    text = raw_text.lower()

    # --- STRONG CONTRACT SIGNALS ---
    contract_strong = [
        "this agreement",
        "agreement is made",
        "whereas",
        "witnesseth",
        "in witness whereof",
        "governing law",
        "term and termination",
        "limitation of liability",
        "confidentiality",
    ]

    if any(k in text for k in contract_strong):
        return "contract"

    # --- STRONG INVOICE SIGNALS ---
    invoice_strong = [
        "invoice no",
        "invoice number",
        "bill to",
        "ship to",
        "balance due",
        "total due",
        "amount due",
    ]

    if any(k in text for k in invoice_strong):
        return "invoice"

    # --- STRUCTURAL FALLBACKS ---
    # Long documents are almost always contracts
    if len(raw_text) > 4000:
        return "contract"

    # Numeric density heuristic (invoices are number-heavy)
    digit_count = sum(c.isdigit() for c in raw_text)
    if digit_count > 0:
        ratio = digit_count / max(len(raw_text), 1)
        if ratio > 0.12:
            return "invoice"

    # Default safe assumption
    return "contract"


# -------------------------------------------------------------------
# Summary builder
# -------------------------------------------------------------------

def build_summary(doc_type: str, fields: Dict[str, Any], risk: Optional[RiskAssessment]) -> str:
    """
    Create a short human-readable summary such as:
      - "Invoice 2967 – SuperStore INVOICE – $6916.80 – medium risk"
      - "Contract between HOF Village, LLC and National Football Museum, Inc. – effective 2018-12-19 – medium risk"
    """
    severity = risk.severity if risk else "unknown"

    if doc_type == "invoice":
        inv_no = fields.get("invoice_number") or "Unknown #"
        vendor = fields.get("vendor_name") or "Unknown vendor"
        amount = fields.get("total_amount")
        currency = fields.get("currency") or ""

        if isinstance(amount, (int, float)):
            amount_str = f"{amount:.2f}"
        else:
            amount_str = "unknown"

        inv_no = normalize_whitespace(str(inv_no))
        vendor = normalize_whitespace(str(vendor))

        return f"Invoice {inv_no} – {vendor} – {currency}{amount_str} – {severity} risk"

    # contract
    party_a = fields.get("party_a") or "Party A"
    party_b = fields.get("party_b") or "Party B"
    effective_date = fields.get("effective_date") or "N/A"

    party_a = normalize_whitespace(str(party_a))
    party_b = normalize_whitespace(str(party_b))

    return (
        f"Contract between {party_a} and {party_b} – "
        f"effective {effective_date} – {severity} risk"
    )


# -------------------------------------------------------------------
# Rule-based clause classification (fast, no transformers)
# -------------------------------------------------------------------

def split_into_sections(raw_text: str) -> List[str]:
    """
    Split contract text into logical sections:

    1. Prefer blank-line separated paragraphs.
    2. Additionally start a new section when we see typical headings like:
       - 'SECTION 1.', 'ARTICLE I', '1. TERM', '2. PAYMENT', etc.
    """
    lines = raw_text.splitlines()

    sections: List[str] = []
    current: List[str] = []

    heading_pattern = re.compile(
        r"""^\s*(
            section\s+\d+(\.\d+)*   |   # SECTION 1, section 2.1 etc
            article\s+[ivx]+        |   # ARTICLE I, ARTICLE IV
            \d+(\.\d+)*\s+          |   # 1 Term, 2.1 Payment etc
            [A-Z][A-Z0-9\s]{4,}$        # FULLY UPPER headings like 'TERMINATION'
        )""",
        re.IGNORECASE | re.VERBOSE,
    )

    for line in lines:
        stripped = line.strip()

        # New logical block if we hit a heading AND we have some accumulated text
        if heading_pattern.match(stripped) and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("\n".join(current).strip())

    # If everything ended up as one big blob, also try splitting by double newlines
    if len(sections) == 1:
        paras = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
        if len(paras) > 1:
            sections = paras

    return [s for s in sections if s.strip()]


# Simple keyword lexicon for clause detection
CLAUSE_KEYWORDS = {
    "confidentiality": [
        "confidentiality", "confidential information",
        "non-disclosure", "nondisclosure",
        "keep confidential", "proprietary information"
    ],
    "termination": [
        "term and termination", "termination", "terminate",
        "termination for convenience", "termination for cause",
        "early termination"
    ],
    "indemnity": [
        "indemnify", "indemnification", "indemnity", "hold harmless"
    ],
    "payment": [
        "payment terms", "payment shall", "payment will be",
        "fees", "fee schedule", "compensation",
        "royalty", "royalties", "invoice", "invoices", "billing"
    ],
    "governing law": [
        "governing law", "shall be governed by the laws of",
        "laws of the state of", "laws of the", "jurisdiction", "venue"
    ],
    "scope of work": [
        "scope of work", "statement of work", "sow",
        "services to be provided", "description of services",
        "service description"
        # NOTE: we intentionally do NOT include bare "services"
        # because that appears in many headings and causes spammy matches.
    ],
    "intellectual property": [
        "intellectual property", "ip rights",
        "ownership of intellectual property", "background ip",
        "foreground ip", "copyright",
        "trademark", "licensor", "licensee", "ownership of"
    ],
    "liability": [
        "limitation of liability", "limited liability",
        "aggregate liability", "liability of either party",
        "consequential damages", "indirect damages", "special damages"
    ],
    "force majeure": [
        "force majeure", "acts of god", "act of god",
        "beyond its reasonable control"
    ],
}


def split_into_clause_candidates(text: str, max_len: int = 900) -> List[str]:
    """
    Split contract text into clause-sized chunks:

    1) First split on blank lines (\\n\\n) – natural paragraphs.
    2) If we still get very long blocks, further split by sentence boundaries.
    """
    # First pass: split on blank lines
    raw_blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]

    candidates: List[str] = []

    for block in raw_blocks:
        if len(block) <= max_len:
            candidates.append(block)
            continue

        # Second pass: split long blocks into sentence-ish chunks
        sentences = re.split(r'(?<=[\.\?\!])\s+', block)
        current = ""
        for s in sentences:
            if not s.strip():
                continue
            if len(current) + len(s) + 1 <= max_len:
                current = (current + " " + s).strip()
            else:
                if current:
                    candidates.append(current.strip())
                current = s.strip()
        if current:
            candidates.append(current.strip())

    # Fallback if everything was whitespace
    if not candidates and text.strip():
        candidates = [text.strip()]

    return candidates


def classify_contract_clauses_rule_based(
    raw_text: str,
    max_paragraphs: int = 80
) -> List[Clause]:
    paragraphs = split_into_clause_candidates(raw_text)
    if not paragraphs:
        return []

    paragraphs = paragraphs[:max_paragraphs]

    clauses: List[Clause] = []
    seen = set()

    for p in paragraphs:
        p_lower = p.lower()

        for label, keywords in CLAUSE_KEYWORDS.items():
            if not any(kw in p_lower for kw in keywords):
                continue

            full_text = p.strip()
            preview = full_text[:300] + "..." if len(full_text) > 300 else full_text

            key = (label, preview)
            if key in seen:
                continue
            seen.add(key)

            clauses.append(
                Clause(
                    type=label.upper().replace(" ", "_"),
                    confidence=0.95,
                    text=full_text,
                    preview=preview,
                )
            )

    return clauses

# -------------------------------------------------------------------
# Semantic clause classification (sentence-transformers)
# -------------------------------------------------------------------

# Prototype sentences for each clause type (used to compute similarity)
CLAUSE_PROTOTYPES = {
    "confidentiality": [
        "Each party agrees to keep all confidential information secret and not disclose it to any third party.",
        "The parties shall maintain the confidentiality of proprietary information."
    ],
    "termination": [
        "This agreement may be terminated by either party upon written notice.",
        "The term of this agreement shall continue until terminated as provided herein."
    ],
    "indemnity": [
        "The supplier shall indemnify and hold harmless the customer from any claims or damages.",
        "Each party agrees to indemnify the other against losses arising from third party claims."
    ],
    "payment": [
        "Payment shall be due within thirty days after receipt of an invoice.",
        "The customer agrees to pay the fees specified in the order form."
    ],
    "governing law": [
        "This agreement shall be governed by the laws of the State of Ohio.",
        "The contract is governed by the laws of the jurisdiction specified below."
    ],
    "scope of work": [
        "The services to be provided by the contractor are described in the scope of work.",
        "This statement of work describes the services to be performed under this agreement."
    ],
    "intellectual property": [
        "All intellectual property rights shall remain the property of the licensor.",
        "The parties acknowledge that no intellectual property is transferred under this agreement."
    ],
    "liability": [
        "In no event shall either party be liable for consequential or incidental damages.",
        "The total liability of either party shall be limited to the fees paid under this agreement."
    ],
    "force majeure": [
        "Neither party shall be liable for delays caused by events of force majeure.",
        "Performance shall be excused due to acts of God or other events beyond the party's reasonable control."
    ],
}


def get_embedding_model():
    """Lazy-load the sentence-transformers model, if available."""
    global EMB_MODEL
    if not EMBEDDING_MODEL_AVAILABLE:
        return None
    if EMB_MODEL is None:
        EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return EMB_MODEL


def ensure_clause_prototype_embeddings(model) -> None:
    """
    Pre-compute embeddings for prototype sentences for each clause label.
    """
    global CLAUSE_PROTOTYPE_EMBEDDINGS
    if CLAUSE_PROTOTYPE_EMBEDDINGS:
        return

    for label, sentences in CLAUSE_PROTOTYPES.items():
        embs = model.encode(sentences, normalize_embeddings=True)
        CLAUSE_PROTOTYPE_EMBEDDINGS[label] = np.array(embs)


def classify_contract_clauses_semantic(
    raw_text: str,
    max_paragraphs: int = 80,
    similarity_threshold: float = 0.6,
) -> List[Clause]:
    model = get_embedding_model()
    if model is None:
        return []

    paragraphs = split_into_clause_candidates(raw_text)
    if not paragraphs:
        return []

    paragraphs = paragraphs[:max_paragraphs]
    para_embs = model.encode(paragraphs, normalize_embeddings=True)
    para_embs = np.array(para_embs)

    ensure_clause_prototype_embeddings(model)

    clauses: List[Clause] = []
    seen_previews = set()

    for i, p in enumerate(paragraphs):
        vec = para_embs[i]

        best_label = None
        best_sim = -1.0

        for label, proto_embs in CLAUSE_PROTOTYPE_EMBEDDINGS.items():
            sims = proto_embs @ vec
            max_sim = float(np.max(sims))
            if max_sim > best_sim:
                best_sim = max_sim
                best_label = label

        if best_label is None or best_sim < similarity_threshold:
            continue

        full_text = p.strip()
        preview = full_text[:300] + "..." if len(full_text) > 300 else full_text

        if preview in seen_previews:
            continue
        seen_previews.add(preview)

        clauses.append(
            Clause(
                type=best_label.upper().replace(" ", "_"),
                confidence=best_sim,
                text=full_text,
                preview=preview,
            )
        )

    return clauses

# -------------------------------------------------------------------
# Clause post-processing (core filter + rounding)
# -------------------------------------------------------------------

# Core clause types you care about for the final JSON
CORE_CLAUSE_TYPES = {
    "CONFIDENTIALITY",
    "TERMINATION",
    "PAYMENT",
    "INDEMNITY",
    "GOVERNING_LAW",
}


def condense_clauses(
    clauses: List[Clause],
    max_per_type: int = 1
) -> List[Clause]:
    """
    Post-process detected clauses:

    - Group by clause.type (e.g. TERMINATION, INDEMNITY).
    - Optionally keep only a core subset of types (CORE_CLAUSE_TYPES).
    - For each type, keep only the top `max_per_type` by confidence.
    - Preserve overall document order for the kept clauses.
    - Round confidence scores for nicer JSON.
    """
    if not clauses:
        return clauses

    # Remember original positions to preserve doc order later
    indexed = list(enumerate(clauses))  # (idx_in_doc, Clause)

    buckets: Dict[str, List[tuple[int, Clause]]] = defaultdict(list)
    for idx, c in indexed:
        buckets[c.type].append((idx, c))

    selected: List[tuple[int, Clause]] = []

    for ctype, bucket in buckets.items():
        # If we're filtering to core types, skip others
        if CORE_CLAUSE_TYPES and ctype not in CORE_CLAUSE_TYPES:
            continue

        # Highest-confidence first
        bucket_sorted = sorted(bucket, key=lambda x: x[1].confidence, reverse=True)
        # Take at most N per type
        top_n = bucket_sorted[:max_per_type]
        selected.extend(top_n)

    # Re-sort by original position so output respects document flow
    selected_sorted = sorted(selected, key=lambda x: x[0])

    final = [c for _, c in selected_sorted]
    # Round confidence scores
    for c in final:
        try:
            c.confidence = round(float(c.confidence), 3)
        except Exception:
            # If for some reason confidence is not numeric, ignore
            pass

    return final


def classify_contract_clauses(
    raw_text: str,
    max_paragraphs: int = 80,
    max_per_type: int = 1,   # tweak this if you want only 1 per label
) -> List[Clause]:
    """
    Unified entry point:

    1) Try semantic clause detection (if sentence-transformers is installed).
    2) If not available or returns nothing, fall back to rule-based detection.
    3) Post-process to keep only the top N clauses per core type.
    """
    semantic_clauses = classify_contract_clauses_semantic(
        raw_text,
        max_paragraphs=max_paragraphs,
    )

    base_clauses = semantic_clauses
    if not base_clauses:
        base_clauses = classify_contract_clauses_rule_based(
            raw_text,
            max_paragraphs=max_paragraphs,
        )

    # 🔑 This is the important post-processing step
    return condense_clauses(base_clauses, max_per_type=max_per_type)


# -------------------------------------------------------------------
# Invoice extraction
# -------------------------------------------------------------------

def extract_invoice_fields(raw_text: str) -> InvoiceFields:
    """
    Basic regex + heuristic-based extraction.
    """
    # Invoice number pattern
    invoice_no = None
    m = re.search(
        r"(Invoice\s*No\.?|Invoice\s*#|#)\s*[:\-]?\s*([A-Za-z0-9\-\/]+)",
        raw_text,
        re.IGNORECASE,
    )
    if m:
        invoice_no = m.group(2).strip()

    # --- Date extraction ---
    invoice_date = None
    due_date = None

    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

    for line in lines:
        lower = line.lower()

        if ("invoice date" in lower or lower.startswith("date:")) and invoice_date is None:
            _, _, tail = line.partition(":")
            parsed = parse_possible_date(tail or line)
            if parsed:
                invoice_date = parsed

        if ("due date" in lower or "balance due" in lower) and due_date is None:
            _, _, tail = line.partition(":")
            parsed = parse_possible_date(tail or line)
            if parsed:
                due_date = parsed

    # Fallback: if still nothing, use the first 2 parseable dates in the whole text
    # ONLY consider lines containing date-related keywords
    if invoice_date is None or due_date is None:
        all_dates: List[str] = []
        date_keywords = ["date", "due", "effective", "end", "expire", "termination"]
        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in date_keywords):
                parsed = parse_possible_date(line)
                if parsed:
                    all_dates.append(parsed)

        if invoice_date is None and all_dates:
            invoice_date = all_dates[0]
        if due_date is None and len(all_dates) > 1:
            due_date = all_dates[1]

    # --- Amount (very naive – pick largest number with currency) ---
    amount_pattern = r"(INR|Rs\.?|₹|\$|EUR)\s*([\d,]+(?:\.\d{2})?)"
    amounts = re.findall(amount_pattern, raw_text)
    total_amount: Optional[float] = None
    currency: Optional[str] = None
    if amounts:
        def parse_amt(t):
            cur, amt = t
            return cur, float(amt.replace(",", ""))
        parsed_amounts = [parse_amt(t) for t in amounts]
        currency, total_amount = max(parsed_amounts, key=lambda x: x[1])

    vendor_name = None
    if lines:
        vendor_name = lines[0][:80]

    invoice_no = normalize_whitespace(invoice_no)
    vendor_name = normalize_whitespace(vendor_name)
    currency = normalize_whitespace(currency)

    sample = "\n".join(lines[:10])

    return InvoiceFields(
        invoice_number=invoice_no,
        vendor_name=vendor_name,
        invoice_date=invoice_date,
        due_date=due_date,
        total_amount=total_amount,
        currency=currency,
        raw_text_sample=sample,
    )


# -------------------------------------------------------------------
# Risk / Data-quality assessment for invoices (with overall score)
# -------------------------------------------------------------------

def assess_invoice_risk(fields: InvoiceFields, raw_text: str) -> RiskAssessment:
    flags: List[str] = []
    severity = "low"

    if not fields.invoice_number:
        flags.append("missing_invoice_number")

    if fields.total_amount is None:
        flags.append("missing_total_amount")

    if not fields.vendor_name:
        flags.append("missing_vendor")

    if not fields.currency:
        flags.append("missing_currency")

    if fields.invoice_date and fields.due_date:
        if fields.invoice_date == fields.due_date:
            flags.append("due_date_same_as_invoice_date")

        try:
            due_dt = datetime.fromisoformat(fields.due_date).date()
            today = datetime.utcnow().date()
            if due_dt < today:
                flags.append("overdue_invoice")
        except Exception:
            pass

    if fields.total_amount is not None:
        amt = fields.total_amount
        if amt < 10:
            flags.append("amount_too_low")
        elif amt > 1_000_000:
            flags.append("amount_too_high")

    money_pattern = r"(Total|Balance Due)\s*[: ]\s*\$?([0-9,]+\.\d{2})"
    matches = re.findall(money_pattern, raw_text, flags=re.IGNORECASE)

    if fields.total_amount is not None and matches:
        _, last_amount_str = matches[-1]
        try:
            final_amount = float(last_amount_str.replace(",", ""))
            if abs(final_amount - fields.total_amount) > 1e-2:
                flags.append("total_mismatch_with_balance_due")
        except ValueError:
            pass

    if any(f.startswith("missing") for f in flags):
        severity = "high"
    elif any(
        f in [
            "total_mismatch_with_balance_due",
            "due_date_same_as_invoice_date",
            "amount_too_high",
            "overdue_invoice",
        ]
        for f in flags
    ):
        severity = "medium"
    else:
        severity = "low"

    if not flags:
        overall_score = 0.1
    elif severity == "low":
        overall_score = min(1.0, 0.3 + 0.05 * len(flags))
    elif severity == "medium":
        overall_score = min(1.0, 0.6 + 0.08 * len(flags))
    else:
        overall_score = min(1.0, 0.8 + 0.05 * len(flags))

    # 🔢 Round overall_score for nicer JSON
    try:
        overall_score = round(float(overall_score), 3)
    except Exception:
        pass

    return RiskAssessment(
        flags=flags,
        severity=severity,
        overall_score=overall_score,
        notes="auto-generated audit & data-quality hints",
    )


# -------------------------------------------------------------------
# Contract extraction
# -------------------------------------------------------------------

def clean_party_name(name: Optional[str]) -> Optional[str]:
    """
    Small cleanup for extracted party names, e.g.:

      'between Reynolds Group Holdings Inc.' -> 'Reynolds Group Holdings Inc.'
      'among the Integrity Short Term Government Fund' -> 'the Integrity Short Term Government Fund'
    """
    if not name:
        return name
    s = name.strip()
    # Strip leading 'between' / 'among'
    s = re.sub(r"^(between|among)\s+", "", s, flags=re.IGNORECASE)
    # Strip leading punctuation after that
    s = re.sub(r"^[,;:\-]+\s*", "", s)
    return s or None


def extract_contract_fields(raw_text: str) -> ContractFields:
    """
    Heuristic-based extraction for contracts.
    """
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

    party_a: Optional[str] = None
    party_b: Optional[str] = None

    undersigned_pair = re.search(
        r"the undersigned,\s*(.*?),\s*\(.*?has agreed that\s*(.*?),\s*\(",
        raw_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if undersigned_pair:
        party_a = undersigned_pair.group(1).strip()
        party_b = undersigned_pair.group(2).strip()
    else:
        intro_match = re.search(
            r"made and entered into.*?by\s+(.*?)(?:\nWITNESSETH|\nWHEREAS|NOW, THEREFORE|$)",
            raw_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if intro_match:
            party_block = intro_match.group(1)
        else:
            party_block = " ".join(lines[:40])

        party_block = re.sub(r"\s+", " ", party_block)

        entity_pattern = re.compile(
            r"[A-Z][A-Za-z0-9&\.\-\s]+?\b(?:Inc\.?|LLC|Ltd\.?|Corp\.?|Corporation|Fund)",
            flags=re.IGNORECASE,
        )
        candidates = entity_pattern.findall(party_block)

        seen = set()
        parties: List[str] = []
        for c in candidates:
            name = c.strip()
            lower_name = name.lower()
            if lower_name not in seen:
                seen.add(lower_name)
                parties.append(name)

        if parties:
            party_a = parties[0]
            if len(parties) > 1:
                suffix_keywords = ["llc", "inc", "ltd", "corp", "corporation", "plc", "gmbh", "fund"]
                preferred_b = None
                for name in parties[1:]:
                    lower = name.lower()
                    if any(sfx in lower for sfx in suffix_keywords):
                        preferred_b = name
                        break
                party_b = preferred_b or parties[1]

    effective_date: Optional[str] = None
    end_date: Optional[str] = None

    eff_match = re.search(
        r"made and entered into as of\s+(.+?)(?:,?\s+by|\.)",
        raw_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if eff_match:
        effective_date = parse_possible_date(eff_match.group(1))

    if effective_date is None:
        date_keywords = ["date", "due", "effective", "end", "expire", "termination"]
        for line in lines[:10]:
            line_lower = line.lower()
            if any(kw in line_lower for kw in date_keywords):
                p = parse_possible_date(line)
                if p:
                    effective_date = p
                    break

    if effective_date:
        try:
            eff_dt: Optional[date] = datetime.fromisoformat(effective_date).date()
        except Exception:
            eff_dt = None
    else:
        eff_dt = None

    candidate_end_dates: List[date] = []
    term_keywords = [
        "term of this agreement",
        "term of the agreement",
        "term shall",
        "shall terminate",
        "termination",
        "expires",
        "expiration",
        "expire on",
        "end on",
        "end of this agreement",
        "continue until",
        "continue in effect until",
        "through",
    ]

    for line in lines:
        low = line.lower()
        if any(k in low for k in term_keywords):
            p = parse_possible_date(line)
            if p:
                try:
                    d = datetime.fromisoformat(p).date()
                    if eff_dt is None or d >= eff_dt:
                        candidate_end_dates.append(d)
                except Exception:
                    pass

    if candidate_end_dates:
        end_date = max(candidate_end_dates).isoformat()
    else:
        end_date = None

    # Ordering validation: if end_date < effective_date, set end_date = None
    if effective_date and end_date:
        try:
            eff_dt = datetime.fromisoformat(effective_date).date()
            end_dt = datetime.fromisoformat(end_date).date()
            if end_dt < eff_dt:
                end_date = None
        except Exception:
            pass

    governing_law = None
    gov_match = re.search(
        r"laws?\s+of\s+the\s+([A-Za-z\s]+?)(?:,|\.)",
        raw_text,
        flags=re.IGNORECASE,
    )
    if gov_match:
        governing_law = "the " + gov_match.group(1).strip()

    payment_terms = None
    pay_match = re.search(
    r"(Payment[\s\S]{0,1200}?\.)(?=\s+[A-Z])",
    raw_text,
    flags=re.IGNORECASE,
    )
    if pay_match:
        payment_terms = pay_match.group(1).strip()

    if not payment_terms:
        for line in lines:
            if "payment" in line.lower():
                payment_terms = line.strip()
                break

    # Clean + normalize parties
    party_a = normalize_whitespace(clean_party_name(party_a))
    party_b = normalize_whitespace(clean_party_name(party_b))
    governing_law = normalize_whitespace(governing_law)
    payment_terms = normalize_whitespace(payment_terms)

    sample = "\n".join(lines[:10])

    return ContractFields(
        party_a=party_a,
        party_b=party_b,
        effective_date=effective_date,
        end_date=end_date,
        governing_law=governing_law,
        payment_terms=payment_terms,
        raw_text_sample=sample,
    )


# -------------------------------------------------------------------
# Risk / Data-quality assessment for contracts (with overall score)
# -------------------------------------------------------------------

def assess_contract_risk(fields: ContractFields, raw_text: str) -> RiskAssessment:
    flags: List[str] = []
    severity = "low"

    if not fields.party_a or not fields.party_b:
        flags.append("missing_parties")

    if not fields.governing_law:
        flags.append("missing_governing_law")

    if not fields.payment_terms:
        flags.append("missing_payment_terms")

    if not fields.effective_date:
        flags.append("missing_effective_date")

    today = datetime.utcnow().date()
    eff_dt = None
    end_dt = None

    if fields.effective_date:
        try:
            eff_dt = datetime.fromisoformat(fields.effective_date).date()
        except Exception:
            pass

    if fields.end_date:
        try:
            end_dt = datetime.fromisoformat(fields.end_date).date()
        except Exception:
            pass

    if end_dt:
        if end_dt < today:
            flags.append("contract_expired")
        if eff_dt and (end_dt - eff_dt).days > 365 * 5:
            flags.append("long_term_contract")

    if "contract_expired" in flags or "missing_governing_law" in flags:
        severity = "high"
    elif (
        "missing_parties" in flags
        or "missing_payment_terms" in flags
        or "missing_effective_date" in flags
        or "long_term_contract" in flags
    ):
        severity = "medium"
    else:
        severity = "low"

    if not flags:
        overall_score = 0.1
    elif severity == "low":
        overall_score = min(1.0, 0.3 + 0.05 * len(flags))
    elif severity == "medium":
        overall_score = min(1.0, 0.6 + 0.08 * len(flags))
    else:
        overall_score = min(1.0, 0.8 + 0.05 * len(flags))

    # 🔢 Round overall_score for nicer JSON
    try:
        overall_score = round(float(overall_score), 3)
    except Exception:
        pass

    return RiskAssessment(
        flags=flags,
        severity=severity,
        overall_score=overall_score,
        notes="auto-generated contract risk & data-quality hints",
    )


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "time": datetime.utcnow().isoformat(),
        "message": "AI Document Intelligence API is up."
    }


@app.post("/extract/invoice", response_model=ExtractionResponse)
async def extract_invoice(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf"]:
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    file_bytes = await file.read()

    try:
        raw_text = pdf_to_text(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {e}")

    if is_mostly_empty(raw_text):
        if not OCR_AVAILABLE:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No text could be extracted from this PDF. "
                    "It looks like a scanned/image-only PDF and OCR is not configured "
                    "(pytesseract / Tesseract not available)."
                ),
            )
        else:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No text could be extracted from this PDF even after OCR. "
                    "Check that Tesseract is installed and on PATH, or try a clearer scan."
                ),
            )

    invoice_fields = extract_invoice_fields(raw_text)
    risk = assess_invoice_risk(invoice_fields, raw_text)
    raw_preview = raw_text[:500] if raw_text else ""

    summary = build_summary("invoice", invoice_fields.dict(), risk)
    clauses: List[Clause] = []  # invoices don't have contract clauses

    content_hash = compute_doc_hash("invoice", raw_text)

    doc_id = save_document(
        doc_type="invoice",
        file_name=file.filename,
        extracted_fields=invoice_fields.dict(),
        raw_preview=raw_preview,
        risk_assessment=risk,
        content_hash=content_hash,
        summary=summary,
        clauses=clauses,
    )

    return ExtractionResponse(
        doc_type="invoice",
        extracted_fields=invoice_fields.dict(),
        raw_preview=raw_preview,
        risk_assessment=risk,
        document_id=doc_id,
        summary=summary,
        clauses=clauses,
    )


@app.post("/extract/contract", response_model=ExtractionResponse)
async def extract_contract(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf"]:
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    file_bytes = await file.read()

    try:
        raw_text = pdf_to_text(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {e}")

    if is_mostly_empty(raw_text):
        if not OCR_AVAILABLE:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No text could be extracted from this PDF. "
                    "It looks like a scanned/image-only PDF and OCR is not configured "
                    "(pytesseract / Tesseract not available)."
                ),
            )
        else:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No text could be extracted from this PDF even after OCR. "
                    "Check that Tesseract is installed and on PATH, or try a clearer scan."
                ),
            )

    contract_fields = extract_contract_fields(raw_text)
    risk = assess_contract_risk(contract_fields, raw_text)
    raw_preview = raw_text[:500] if raw_text else ""

    # Semantic + rule-based clause classification
    clauses = classify_contract_clauses(raw_text)

    summary = build_summary("contract", contract_fields.dict(), risk)
    content_hash = compute_doc_hash("contract", raw_text)

    doc_id = save_document(
        doc_type="contract",
        file_name=file.filename,
        extracted_fields=contract_fields.dict(),
        raw_preview=raw_preview,
        risk_assessment=risk,
        content_hash=content_hash,
        summary=summary,
        clauses=clauses,
    )

    return ExtractionResponse(
        doc_type="contract",
        extracted_fields=contract_fields.dict(),
        raw_preview=raw_preview,
        risk_assessment=risk,
        document_id=doc_id,
        summary=summary,
        clauses=clauses,
    )


# -------------------------------------------------------------------
# NEW: Auto doc_type detection
# -------------------------------------------------------------------

@app.post("/extract/auto", response_model=ExtractionResponse)
async def extract_auto(file: UploadFile = File(...)):
    """
    Auto-detect doc_type (invoice vs contract) from content,
    then run the appropriate extractor + risk model + clause classifier (for contracts).
    """
    if file.content_type not in ["application/pdf"]:
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    file_bytes = await file.read()

    try:
        raw_text = pdf_to_text(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {e}")

    if is_mostly_empty(raw_text):
        if not OCR_AVAILABLE:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No text could be extracted from this PDF. "
                    "It looks like a scanned/image-only PDF and OCR is not configured "
                    "(pytesseract / Tesseract not available)."
                ),
            )
        else:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No text could be extracted from this PDF even after OCR. "
                    "Check that Tesseract is installed and on PATH, or try a clearer scan."
                ),
            )

    doc_type = classify_doc_type_from_text(raw_text)
    raw_preview = raw_text[:500] if raw_text else ""

    if doc_type == "invoice":
        fields_obj = extract_invoice_fields(raw_text)
        risk = assess_invoice_risk(fields_obj, raw_text)
        clauses: List[Clause] = []
    else:
        doc_type = "contract"
        fields_obj = extract_contract_fields(raw_text)
        risk = assess_contract_risk(fields_obj, raw_text)
        clauses = classify_contract_clauses(raw_text)

    fields_dict = fields_obj.dict()
    summary = build_summary(doc_type, fields_dict, risk)
    content_hash = compute_doc_hash(doc_type, raw_text)

    doc_id = save_document(
        doc_type=doc_type,
        file_name=file.filename,
        extracted_fields=fields_dict,
        raw_preview=raw_preview,
        risk_assessment=risk,
        content_hash=content_hash,
        summary=summary,
        clauses=clauses,
    )

    return ExtractionResponse(
        doc_type=doc_type,
        extracted_fields=fields_dict,
        raw_preview=raw_preview,
        risk_assessment=risk,
        document_id=doc_id,
        summary=summary,
        clauses=clauses,
    )


# -------------------------------------------------------------------
# Analyze endpoints – accept full ExtractionResponse
# -------------------------------------------------------------------

@app.post("/analyze/invoice", response_model=RiskAssessment)
async def analyze_invoice(payload: ExtractionResponse):
    """
    Recompute invoice risk from a previous extraction result.
    You can paste the JSON you got from /extract/invoice directly.
    """
    try:
        fields = InvoiceFields(**payload.extracted_fields)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid invoice fields: {e}")

    raw_text = payload.raw_preview or (fields.raw_text_sample or "")
    return assess_invoice_risk(fields, raw_text)


@app.post("/analyze/contract", response_model=RiskAssessment)
async def analyze_contract(payload: ExtractionResponse):
    """
    Recompute contract risk from a previous extraction result.
    You can paste the JSON you got from /extract/contract directly.
    """
    try:
        fields = ContractFields(**payload.extracted_fields)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid contract fields: {e}")

    raw_text = payload.raw_preview or (fields.raw_text_sample or "")
    return assess_contract_risk(fields, raw_text)


# -------------------------------------------------------------------
# Fetch + delete stored documents
# -------------------------------------------------------------------

@app.get("/documents/{doc_id}", response_model=ExtractionResponse)
def get_document(doc_id: int):
    doc = load_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int):
    with conn:
        cur = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Document not found.")
    return {"status": "deleted", "id": doc_id}


@app.delete("/documents")
def delete_all_documents():
    with conn:
        conn.execute("DELETE FROM documents")
    return {"status": "deleted_all"}


# -------------------------------------------------------------------
# List stored documents with filters
# -------------------------------------------------------------------

@app.get("/documents", response_model=List[DocumentSummary])
def list_documents(
    doc_type: Optional[str] = None,
    severity: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
):
    """
    List stored documents (summary view) with optional filters:
      - doc_type: "invoice" | "contract"
      - severity: "low" | "medium" | "high"
      - from_date, to_date: YYYY-MM-DD (filter on created_at)
    """
    from_dt: Optional[date] = None
    to_dt: Optional[date] = None

    if from_date:
        try:
            from_dt = datetime.fromisoformat(from_date).date()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid from_date (expected YYYY-MM-DD).")

    if to_date:
        try:
            to_dt = datetime.fromisoformat(to_date).date()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid to_date (expected YYYY-MM-DD).")

    rows = conn.execute(
        "SELECT * FROM documents ORDER BY datetime(created_at) DESC"
    ).fetchall()

    results: List[DocumentSummary] = []
    for row in rows:
        if doc_type and row["doc_type"] != doc_type:
            continue

        created_raw = row["created_at"]
        try:
            created_dt = datetime.fromisoformat(created_raw).date()
        except Exception:
            created_dt = None

        if from_dt and created_dt and created_dt < from_dt:
            continue
        if to_dt and created_dt and created_dt > to_dt:
            continue

        risk_raw = row["risk_assessment"]
        risk_severity = None
        risk_flags: List[str] = []
        overall_score: Optional[float] = None

        if risk_raw:
            try:
                r = json.loads(risk_raw)
                risk_severity = r.get("severity")
                risk_flags = r.get("flags", []) or []
                overall_score = r.get("overall_score")
                if overall_score is not None:
                    try:
                        overall_score = round(float(overall_score), 3)
                    except Exception:
                        overall_score = None
            except Exception:
                pass

        if severity and risk_severity != severity:
            continue

        results.append(
            DocumentSummary(
                id=row["id"],
                doc_type=row["doc_type"],
                created_at=row["created_at"],
                file_name=row["file_name"],
                risk_severity=risk_severity,
                risk_flags=risk_flags,
                overall_score=overall_score,
                summary=row["summary"],
            )
        )

    return results

@app.get("/metrics")
def system_metrics():
    """
    Aggregate metrics over all stored documents.
    Interview angle: 'How did you evaluate your system?'
    """
    rows = conn.execute("SELECT * FROM documents").fetchall()
    if not rows:
        return {"message": "No documents available"}

    total = len(rows)
    by_type = defaultdict(int)
    severity_count = defaultdict(int)
    avg_scores = defaultdict(list)

    for r in rows:
        by_type[r["doc_type"]] += 1
        if r["risk_assessment"]:
            risk = json.loads(r["risk_assessment"])
            severity_count[risk["severity"]] += 1
            avg_scores[risk["severity"]].append(risk["overall_score"])

    avg_scores_final = {
        k: round(sum(v) / len(v), 3) for k, v in avg_scores.items() if v
    }

    return {
        "total_documents": total,
        "documents_by_type": dict(by_type),
        "risk_severity_distribution": dict(severity_count),
        "average_risk_score_by_severity": avg_scores_final
    }

@app.get("/explain/{doc_id}")
def explain_document(doc_id: int):
    """
    Explain why a document got its risk score.
    Interview angle: 'Is your AI explainable?'
    """
    doc = load_document(doc_id)
    if not doc or not doc.risk_assessment:
        raise HTTPException(status_code=404, detail="Document or risk data not found")

    risk = doc.risk_assessment

    explanations = []
    for flag in risk.flags:
        explanations.append(f"Flag triggered: {flag.replace('_', ' ')}")

    return {
        "document_id": doc_id,
        "doc_type": doc.doc_type,
        "severity": risk.severity,
        "overall_score": risk.overall_score,
        "explanations": explanations
    }


@app.get("/error-analysis")
def error_analysis():
    """
    Identify recurring missing fields & weak signals.
    Interview angle: 'What are your model limitations?'
    """
    rows = conn.execute("SELECT * FROM documents").fetchall()
    flag_frequency = defaultdict(int)

    for r in rows:
        if r["risk_assessment"]:
            risk = json.loads(r["risk_assessment"])
            for f in risk.get("flags", []):
                flag_frequency[f] += 1

    sorted_flags = sorted(
        flag_frequency.items(), key=lambda x: x[1], reverse=True
    )

    return {
        "most_common_issues": [
            {"flag": f, "count": c} for f, c in sorted_flags
        ]
    }


@app.get("/stats")
def system_stats():
    """
    Operational stats for system monitoring.
    Interview angle: 'How does your system scale?'
    """
    rows = conn.execute("SELECT created_at FROM documents").fetchall()

    dates = []
    for r in rows:
        try:
            dates.append(datetime.fromisoformat(r["created_at"]).date())
        except Exception:
            pass

    per_day = defaultdict(int)
    for d in dates:
        per_day[str(d)] += 1

    return {
        "documents_processed": len(rows),
        "documents_per_day": dict(per_day)
    }



import io
import os
import re
from typing import List, Dict, Any
from pypdf import PdfReader

AUTHORITY_WEIGHTS = {
    "REGULATORY_BINDING": 4,
    "BOARD_RESOLUTION": 3,
    "EXECUTED_CONTRACT": 2,
    "INTERNAL_MEMO": 1,
}


def extract_effective_date(text: str) -> str:
    m = re.search(r"Effective Date:\s*(\d{4}-\d{2}-\d{2})", text)
    if m:
        return m.group(1)
    m_iso = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    if m_iso:
        return m_iso.group(1)
    m_yr = re.search(r"\b(20\d{2})\b", text)
    return f"{m_yr.group(1)}-01-01" if m_yr else "1970-01-01"


def extract_authority(text: str) -> tuple:
    for label, weight in AUTHORITY_WEIGHTS.items():
        if label in text:
            return label, weight
    lower = text.lower()
    if "article " in lower or "regulatory" in lower or "ema" in lower:
        return "REGULATORY_BINDING", 4
    if "board resolution" in lower:
        return "BOARD_RESOLUTION", 3
    if "agreement" in lower or "amendment" in lower or "contract" in lower:
        return "EXECUTED_CONTRACT", 2
    return "INTERNAL_MEMO", 1


def chunk_text(text: str, chunk_words: int = 95, overlap: int = 20) -> List[str]:
    words = text.split()
    if len(words) <= chunk_words:
        return [" ".join(words)] if words else []
    chunks = []
    step = max(1, chunk_words - overlap)
    for start in range(0, len(words), step):
        part = " ".join(words[start : start + chunk_words])
        if part:
            chunks.append(part)
        if start + chunk_words >= len(words):
            break
    return chunks


def parse_document_bytes(filename: str, raw_bytes: bytes) -> List[Dict[str, Any]]:
    ext = os.path.splitext(filename)[1].lower()
    pages_text: List[tuple] = []

    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(raw_bytes))
        for page_num, page in enumerate(reader.pages, 1):
            txt = (page.extract_text() or "").strip()
            if txt:
                pages_text.append((page_num, txt))
    else:
        txt = raw_bytes.decode("utf-8", errors="ignore").strip()
        if txt:
            pages_text.append((1, txt))

    full_text = "\n".join(t for _, t in pages_text)
    eff_date = extract_effective_date(full_text)
    auth_label, auth_tier = extract_authority(full_text)
    doc_id = os.path.splitext(filename)[0]

    chunks = []
    chunk_idx = 0
    for page_num, p_text in pages_text:
        for seg in chunk_text(p_text):
            chunk_idx += 1
            chunks.append({
                "chunk_id": f"{doc_id}::p{page_num}::c{chunk_idx}",
                "doc_id": doc_id,
                "filename": filename,
                "page": page_num,
                "effective_date": eff_date,
                "authority_label": auth_label,
                "authority_tier": auth_tier,
                "content": seg,
            })
    return chunks


def load_directory(docs_dir: str) -> List[Dict[str, Any]]:
    all_chunks = []
    if not os.path.exists(docs_dir):
        return all_chunks
    for fname in sorted(os.listdir(docs_dir)):
        if fname.lower().endswith((".pdf", ".md", ".txt")):
            fpath = os.path.join(docs_dir, fname)
            with open(fpath, "rb") as f:
                all_chunks.extend(parse_document_bytes(fname, f.read()))
    return all_chunks

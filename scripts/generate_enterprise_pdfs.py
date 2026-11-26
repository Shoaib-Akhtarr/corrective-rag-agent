import os
from fpdf import FPDF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(ROOT, "documents")
os.makedirs(DOCS_DIR, exist_ok=True)

PDF_DOCS = [
    (
        "01_Apex_Sovereign_Cloud_Policy_2025_Q1.pdf",
        "Apex Cloud Infrastructure — Sovereign Storage & Encryption Policy (Effective: 2025-02-01)",
        [
            "Document Authority: INTERNAL_MEMO | Effective Date: 2025-02-01 | Policy ID: POL-STR-2025-01\n\n"
            "Section 1. Default Vector Storage Layer.\n"
            "All European sovereign cloud regions (eu-central-2 and eu-west-3) shall utilize LegacyVault-v2 as the default vector storage backend with p99 query latency targets of 45.0 milliseconds across 100 million embeddings.\n\n"
            "Section 2. Key Rotation Governance.\n"
            "Encryption key rotations for eu-central-2 clusters are authorized solely by the Frankfurt Ops Committee on a 90-day rotation cycle. External HSM quorums are optional for non-clinical workloads.",
            "Section 3. Compute Reservation SLA.\n"
            "Standard enterprise compute reservations for AI training clusters carry a 5% SLA credit cap for hardware delivery delays exceeding 8 weeks, applicable only to reservations above 25,000 GPU units."
        ],
    ),
    (
        "02_NovaVector_Acquisition_Addendum_2025_Q3.pdf",
        "Apex Cloud Infrastructure — NovaVector Acquisition & HelixDB Migration Directive (Effective: 2025-09-15)",
        [
            "Document Authority: EXECUTED_CONTRACT | Effective Date: 2025-09-15 | Contract ID: MA-NV-2025-09\n\n"
            "Section 1. Acquisition & Storage Layer Supersession.\n"
            "Following the $1.4 billion acquisition of NovaVector Systems, HelixDB formally replaces LegacyVault-v2 as the mandatory default vector storage engine across eu-central-2 and eu-west-3. Former NovaVector CTO Dr. Elena Rostova is appointed VP of Storage Architecture.\n\n"
            "Section 2. Excluded Patent Spin-Off.\n"
            "The acquisition explicitly excludes NovaVector's legacy consumer photo-indexing patents, which were spun off into Luminos Labs prior to closing. Apex Cloud Infrastructure retains zero legal liability or ownership stake in Luminos Labs.",
            "Section 3. HelixDB Latency Benchmark.\n"
            "Production DiskANN HNSW hybrid indexing on HelixDB achieves p99 query latency of 14.2 milliseconds across 500 million 1536-dimensional float16 vectors."
        ],
    ),
    (
        "03_Zurich_HSM_Board_Resolution_2026_Q1.pdf",
        "Apex Cloud Infrastructure — Board Resolution on Zurich HSM Quorum (Effective: 2026-01-20)",
        [
            "Document Authority: BOARD_RESOLUTION | Effective Date: 2026-01-20 | Resolution ID: BR-SEC-2026-04\n\n"
            "Section 1. Mandatory HSM Quorum Authorization.\n"
            "Superseding the 2025 Frankfurt Ops Committee memo, all encryption key rotations for HelixDB clusters in eu-central-2 and eu-west-3 must be authorized by the Zurich HSM Quorum on a 30-day cycle via 3-of-5 Shamir attestation.\n\n"
            "Section 2. Quorum Leadership & Governance.\n"
            "Dr. Elena Rostova is appointed Chair of the Zurich HSM Quorum. Per Swiss Commercial Registry filings, Dr. Rostova holds an independent non-executive seat at Alpine Photonics AG (Zurich optical interconnects) and holds zero board seats in any non-EU semiconductor firm."
        ],
    ),
    (
        "04_EMA_DORA_Clinical_Telemetry_Ruling_2026_Q1.pdf",
        "European Medicines Agency (EMA) — DORA Cross-Border Clinical Telemetry Ruling (Effective: 2026-02-10)",
        [
            "Document Authority: REGULATORY_BINDING | Effective Date: 2026-02-10 | Ruling ID: EMA-DORA-2026-11\n\n"
            "Article 4. Mandatory Audit Indexing Latency.\n"
            "Any clinical trial or genomic rare-disease platform storing EU patient biometric telemetry in sovereign cloud regions must maintain sub-20.0ms p99 audit log indexing. Any system exceeding 20.0ms faces an immediate 30-day clinical data freeze.\n\n"
            "Article 7. HSM Quorum Independence.\n"
            "Chairs of sovereign HSM encryption quorums must hold zero board seats in any non-EU semiconductor firm. Independent EU optical interconnect directorships remain permitted."
        ],
    ),
    (
        "05_Meridian_AI_SLA_Amendment_2026_Q1.pdf",
        "Meridian AI Labs & Apex Cloud — $680M Compute Reservation & TPF Delay Settlement (Effective: 2026-02-22)",
        [
            "Document Authority: EXECUTED_CONTRACT | Effective Date: 2026-02-22 | Amendment ID: AMD-MER-2026-02\n\n"
            "Section 1. Amended SLA Penalty Threshold.\n"
            "Amending the 2025 standard threshold of 25,000 units, compute reservations exceeding 10,000 Vertex V-900 units trigger mandatory SLA penalty credits if wafer packaging delays exceed 4 weeks.\n\n"
            "Section 2. Force Majeure Settlement.\n"
            "Due to Taiwan Precision Foundry (TPF) Hsinchu Fab 4 CoWoS-L packaging delays deferring Meridian AI Labs' 16,000 Vertex V-900 accelerators, Apex Cloud Infrastructure issued $19.5 million in SLA penalty credits."
        ],
    ),
    (
        "06_BioNova_VALOR2_Clinical_Telemetry_2026_Q1.pdf",
        "BioNova Therapeutics — Phase III VALOR-2 Trial Sovereign Data Architecture (Effective: 2026-01-12)",
        [
            "Document Authority: EXECUTED_CONTRACT | Effective Date: 2026-01-12 | Protocol ID: BN-402-VAL2\n\n"
            "Section 1. Clinical Data Lake Hosting.\n"
            "BioNova Therapeutics runs the Phase III VALOR-2 trial for oral GLP-1/GIP agonist BN-402 across 42 EU sites. Patient cohort telemetry is ingested into Apex Cloud Infrastructure eu-central-2 using HelixDB at 11.4ms p99 latency."
        ],
    ),
    (
        "07_DISTRACTOR_Apex_Sports_Apparel_Earnings.pdf",
        "Apex Sports Apparel — Q4 Retail Financial Release (Effective: 2026-01-18)",
        [
            "Document Authority: INTERNAL_MEMO | Effective Date: 2026-01-18 | Release ID: ASA-2025-Q4\n\n"
            "Apex Sports Apparel (unrelated to Apex Cloud Infrastructure) reported Q4 retail revenue of $320 million driven by winter running jackets across North America. CFO Sarah Jenkins confirmed zero exposure to European cloud software or vector databases."
        ],
    ),
    (
        "08_DISTRACTOR_Meridian_Maritime_Fleet_Update.pdf",
        "Meridian Maritime Group — LNG Container Vessel Christening (Effective: 2026-01-25)",
        [
            "Document Authority: INTERNAL_MEMO | Effective Date: 2026-01-25 | Release ID: MMG-2026-01\n\n"
            "Meridian Maritime Group (unrelated to Meridian AI Labs) christened two LNG-powered container vessels in Rotterdam for the Asia-Europe shipping corridor. The fleet has zero involvement with AI accelerators or semiconductor manufacturing."
        ],
    ),
]


def build_pdfs():
    for filename, title, pages in PDF_DOCS:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        for idx, body in enumerate(pages, 1):
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 13)
            hdr = (title if idx == 1 else f"{title} (Page {idx})").replace("\u2014", "-")
            pdf.multi_cell(0, 8, hdr)
            pdf.ln(3)
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, body.replace("\u2014", "-"))
        out_path = os.path.join(DOCS_DIR, filename)
        pdf.output(out_path)
    print(f"Generated {len(PDF_DOCS)} multi-page PDF documents in {DOCS_DIR}")


if __name__ == "__main__":
    build_pdfs()

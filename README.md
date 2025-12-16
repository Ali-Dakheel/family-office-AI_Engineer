# ADOR - Augmented Document Reader

A Proof of Concept (PoC) for Named Entity Recognition (NER) in financial documents. This tool extracts predefined financial entities from various document formats using rule-based parsing, NER models, and LLM-based approaches.

## Project Structure

```
ador_project/
├── src/                          # Source code
│   ├── docx_entity_extractor.py  # Rule-based parser for DOCX files
│   └── ner_entity_extractor.py   # Hybrid NER for chat/text files
├── data/                         # Sample input documents
│   ├── *.txt                     # Chat/text samples
│   ├── *.docx                    # Structured document samples
│   └── *.pdf                     # PDF samples (for LLM+RAG)
├── outputs/                      # Extraction results (JSON)
├── docs/                         # Documentation
│   ├── Architecture_GAD.md       # Global Architecture Document
│   ├── NER_Methodology_GMD.md    # NER Fine-tuning Methodology
│   └── LLM_RAG_Methodology_GMD.md # LLM+RAG Pipeline Methodology
└── requirements.txt              # Python dependencies
```

## Installation

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (optional, for NER)
python -m spacy download en_core_web_sm
```

## Quick Start

### DOCX Entity Extraction (Rule-Based Parser)

Extract entities from structured DOCX documents:

```bash
# Basic extraction
python src/docx_entity_extractor.py data/ZF4894_ALV_07Aug2026_physical.docx

# With verbose output
python src/docx_entity_extractor.py data/ZF4894_ALV_07Aug2026_physical.docx -v

# Custom output path
python src/docx_entity_extractor.py data/ZF4894_ALV_07Aug2026_physical.docx -o results.json
```

**Entities Extracted from DOCX:**
- Counterparty
- Initial Valuation Date
- Notional
- Valuation Date
- Maturity
- Underlying
- Coupon
- Barrier
- Calendar

### Chat/Text Entity Extraction (NER Model)

Extract entities from chat messages or text files:

```bash
# With spaCy NER (recommended)
python src/ner_entity_extractor.py data/FR001400QV82_AVMAFC_30Jun2028.txt -v

# Rule-based only (no spaCy required)
python src/ner_entity_extractor.py data/FR001400QV82_AVMAFC_30Jun2028.txt --no-spacy -v
```

**Entities Extracted from Chat:**
- Counterparty
- Notional
- ISIN
- Underlying
- Maturity
- Bid
- Offer
- Payment Frequency

## Output Format

Extraction results are saved as JSON with metadata:

```json
{
  "counterparty": "BANK ABC",
  "notional": "EUR 1 million",
  "maturity": "07 August 2026",
  "extraction_timestamp": "2025-12-16T11:42:10.424692",
  "source_document": "sample.docx"
}
```

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture_GAD.md](docs/Architecture_GAD.md) | System architecture, APIs, deployment |
| [NER_Methodology_GMD.md](docs/NER_Methodology_GMD.md) | NER model fine-tuning methodology |
| [LLM_RAG_Methodology_GMD.md](docs/LLM_RAG_Methodology_GMD.md) | LLM+RAG pipeline for PDFs |

## Extraction Approaches

| Document Type | Method | Tool |
|--------------|--------|------|
| DOCX (structured) | Rule-based parser | `docx_entity_extractor.py` |
| Chat/Text | Hybrid NER + regex | `ner_entity_extractor.py` |
| PDF (verbose) | LLM + RAG | See methodology doc |

## Dependencies

**Core:**
- `python-docx` - DOCX parsing
- `spacy` - NER model (optional)

**Full stack (for production):**
- FastAPI, Celery, Redis - API & async processing
- ChromaDB, sentence-transformers - Vector storage & embeddings
- anthropic, openai - LLM integration

## License

Internal use - CMI Architecture & Innovation Team

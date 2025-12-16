"""
ADOR - Augmented Document Reader
NER Entity Extractor Module

This module implements a Named Entity Recognition (NER) approach for extracting
financial entities from semi-structured chat/text documents. It combines
pre-trained NER models with custom rule-based patterns for financial entities.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ChatFinancialEntity:
    """Data class representing extracted financial entities from chat data."""
    counterparty: Optional[str] = None
    notional: Optional[str] = None
    isin: Optional[str] = None
    underlying: Optional[str] = None
    maturity: Optional[str] = None
    bid: Optional[str] = None
    offer: Optional[str] = None
    payment_frequency: Optional[str] = None
    
    # Metadata
    extraction_timestamp: str = None
    source_document: str = None
    extraction_method: str = "Hybrid NER + Rules"
    
    def __post_init__(self):
        if self.extraction_timestamp is None:
            self.extraction_timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class NEREntityExtractor:
    """
    Hybrid NER-based entity extractor for financial chat data.
    
    This extractor uses a two-stage approach:
    1. Pre-trained NER model for general entity recognition
    2. Financial domain-specific regex patterns for specialized entities
    
    The hybrid approach provides robustness against variations in chat format
    while maintaining high precision for well-defined financial entities.
    """
    
    # Financial entity patterns
    PATTERNS = {
        'isin': r'\b([A-Z]{2}[A-Z0-9]{9}[0-9])\b',  # ISIN code format
        'notional': r'\b(\d+(?:\.\d+)?\s*(?:mio|million|bn|billion))\b',
        'maturity': r'\b(\d+Y(?:\s*EVG)?)\b',  # e.g., 2Y EVG
        'bid': r'(?:bid|Bid)\s*[►:]\s*([^\s\n]+)',
        'offer': r'(?:offer)\s+(\d+Y\s+[^\n]+)',
        'payment_frequency': r'(?:Quarterly|Monthly|Annual|Semi-annual)',
        'underlying': r'([A-Z]+\s+FLOAT\s+\d{2}/\d{2}/\d{2})',
    }
    
    # Known financial institution patterns
    COUNTERPARTY_PATTERNS = [
        r'\b([A-Z]+\s+ABC)\b',  # BANK ABC format
        r'\b(BANK\s+[A-Z]+)\b',
        r'\b([A-Z]+\s+BANK)\b',
    ]
    
    def __init__(self, file_path: str, use_spacy: bool = True):
        """
        Initialize the NER extractor.
        
        Args:
            file_path: Path to the chat/text file
            use_spacy: Whether to use spaCy NER (requires installation)
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.use_spacy = use_spacy
        self.text = self.file_path.read_text(encoding='utf-8')
        self.entity = ChatFinancialEntity(source_document=self.file_path.name)
        
        # Load spaCy model if available
        self.nlp = None
        if use_spacy:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
            except (ImportError, OSError):
                print("Warning: spaCy not available, using rule-based extraction only")
                self.use_spacy = False
    
    def extract(self) -> ChatFinancialEntity:
        """
        Main extraction method.
        
        Returns:
            ChatFinancialEntity object with extracted data
        """
        # Stage 1: Rule-based extraction (always executed)
        self._extract_with_patterns()
        
        # Stage 2: NER-based extraction (if spaCy available)
        if self.use_spacy and self.nlp:
            self._extract_with_ner()
        
        # Stage 3: Post-processing
        self._post_process()
        
        return self.entity
    
    def _extract_with_patterns(self) -> None:
        """Extract entities using regex patterns."""
        # Extract ISIN
        isin_match = re.search(self.PATTERNS['isin'], self.text)
        if isin_match:
            self.entity.isin = isin_match.group(1)
        
        # Extract Notional
        notional_match = re.search(self.PATTERNS['notional'], self.text, re.IGNORECASE)
        if notional_match:
            self.entity.notional = notional_match.group(1)
        
        # Extract Maturity
        maturity_match = re.search(self.PATTERNS['maturity'], self.text)
        if maturity_match:
            self.entity.maturity = maturity_match.group(1)
        
        # Extract Bid
        bid_match = re.search(self.PATTERNS['bid'], self.text, re.IGNORECASE)
        if bid_match:
            self.entity.bid = bid_match.group(1).strip()
        
        # Extract Offer
        offer_match = re.search(self.PATTERNS['offer'], self.text, re.IGNORECASE)
        if offer_match:
            self.entity.offer = offer_match.group(1).strip()
        
        # Extract Payment Frequency
        freq_match = re.search(self.PATTERNS['payment_frequency'], self.text, re.IGNORECASE)
        if freq_match:
            self.entity.payment_frequency = freq_match.group(0)
        
        # Extract Underlying
        underlying_match = re.search(self.PATTERNS['underlying'], self.text)
        if underlying_match:
            self.entity.underlying = underlying_match.group(1)
        
        # Extract Counterparty
        for pattern in self.COUNTERPARTY_PATTERNS:
            counterparty_match = re.search(pattern, self.text)
            if counterparty_match:
                self.entity.counterparty = counterparty_match.group(1)
                break
    
    def _extract_with_ner(self) -> None:
        """
        Extract entities using spaCy NER.
        
        This method demonstrates how a pre-trained NER model can complement
        rule-based extraction by identifying organizations, dates, etc.
        """
        doc = self.nlp(self.text)
        
        # Extract organizations (potential counterparties)
        if not self.entity.counterparty:
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    # Validate it looks like a bank/financial institution
                    if any(keyword in ent.text.upper() for keyword in ['BANK', 'ABC']):
                        self.entity.counterparty = ent.text
                        break
        
        # Extract monetary values (can supplement notional detection)
        for ent in doc.ents:
            if ent.label_ == "MONEY" and not self.entity.notional:
                self.entity.notional = ent.text
    
    def _post_process(self) -> None:
        """Clean and normalize extracted entities."""
        # Normalize notional format
        if self.entity.notional:
            self.entity.notional = re.sub(r'\s+', ' ', self.entity.notional).strip()
        
        # Normalize offer (remove extra whitespace/newlines)
        if self.entity.offer:
            self.entity.offer = ' '.join(self.entity.offer.split())
        
        # Uppercase ISIN
        if self.entity.isin:
            self.entity.isin = self.entity.isin.upper()
    
    def export_json(self, output_path: Optional[str] = None) -> str:
        """Export extracted entities to JSON."""
        if output_path is None:
            output_dir = self.file_path.parent.parent / 'outputs'
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{self.file_path.stem}_entities.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.entity.to_dict(), f, indent=2, ensure_ascii=False)
        
        return str(output_path)
    
    def export_summary(self) -> str:
        """Generate a human-readable summary."""
        lines = ["=" * 60, "EXTRACTED FINANCIAL ENTITIES (NER)", "=" * 60, ""]
        
        for field, value in self.entity.to_dict().items():
            if field not in ['extraction_timestamp', 'source_document', 'extraction_method']:
                label = field.replace('_', ' ').title()
                lines.append(f"{label:.<30} {value}")
        
        lines.extend(["", "=" * 60,
                     f"Method: {self.entity.extraction_method}",
                     f"Source: {self.entity.source_document}",
                     f"Extracted: {self.entity.extraction_timestamp}",
                     "=" * 60])
        
        return "\n".join(lines)


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract financial entities from chat/text files using NER'
    )
    parser.add_argument('input_file', help='Path to input text file')
    parser.add_argument('-o', '--output', help='Output JSON file path', default=None)
    parser.add_argument('-v', '--verbose', action='store_true', help='Print summary')
    parser.add_argument('--no-spacy', action='store_true', help='Disable spaCy NER')
    
    args = parser.parse_args()
    
    try:
        extractor = NEREntityExtractor(args.input_file, use_spacy=not args.no_spacy)
        entity = extractor.extract()
        
        output_path = extractor.export_json(args.output)
        
        if args.verbose:
            print(extractor.export_summary())
            print(f"\nResults saved to: {output_path}")
        else:
            print(f"Extraction complete: {output_path}")
        
        return 0
    
    except Exception as e:
        import sys
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
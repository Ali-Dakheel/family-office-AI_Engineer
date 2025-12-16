"""
ADOR - Unit Tests for Entity Extractors

Tests for DOCX and NER entity extraction functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ner_entity_extractor import NEREntityExtractor, ChatFinancialEntity


class TestNEREntityExtractor:
    """Tests for NER-based entity extraction."""

    @pytest.fixture
    def sample_chat_file(self, tmp_path):
        """Create a sample chat file for testing."""
        content = """11:49:05 I'll revert regarding BANK ABC to try to do another 200 mio at 2Y
FR001400QV82	AVMAFC FLOAT	06/30/28
Bid ► estr+45bps
Offer 2Y EVG estr+50bps
estr average Estr average / Quarterly interest payment
"""
        file_path = tmp_path / "test_chat.txt"
        file_path.write_text(content, encoding='utf-8')
        return str(file_path)

    def test_extract_isin(self, sample_chat_file):
        """Test ISIN extraction."""
        extractor = NEREntityExtractor(sample_chat_file, use_spacy=False)
        entity = extractor.extract()
        assert entity.isin == "FR001400QV82"

    def test_extract_counterparty(self, sample_chat_file):
        """Test counterparty extraction."""
        extractor = NEREntityExtractor(sample_chat_file, use_spacy=False)
        entity = extractor.extract()
        assert entity.counterparty == "BANK ABC"

    def test_extract_notional(self, sample_chat_file):
        """Test notional extraction."""
        extractor = NEREntityExtractor(sample_chat_file, use_spacy=False)
        entity = extractor.extract()
        assert entity.notional == "200 mio"

    def test_extract_maturity(self, sample_chat_file):
        """Test maturity extraction."""
        extractor = NEREntityExtractor(sample_chat_file, use_spacy=False)
        entity = extractor.extract()
        assert entity.maturity == "2Y"

    def test_extract_bid(self, sample_chat_file):
        """Test bid extraction."""
        extractor = NEREntityExtractor(sample_chat_file, use_spacy=False)
        entity = extractor.extract()
        assert entity.bid == "estr+45bps"

    def test_extract_offer(self, sample_chat_file):
        """Test offer extraction."""
        extractor = NEREntityExtractor(sample_chat_file, use_spacy=False)
        entity = extractor.extract()
        assert entity.offer == "2Y EVG estr+50bps"

    def test_extract_payment_frequency(self, sample_chat_file):
        """Test payment frequency extraction."""
        extractor = NEREntityExtractor(sample_chat_file, use_spacy=False)
        entity = extractor.extract()
        assert entity.payment_frequency == "Quarterly"

    def test_extract_underlying(self, sample_chat_file):
        """Test underlying extraction."""
        extractor = NEREntityExtractor(sample_chat_file, use_spacy=False)
        entity = extractor.extract()
        assert "AVMAFC FLOAT" in entity.underlying

    def test_file_not_found(self):
        """Test error handling for non-existent files."""
        with pytest.raises(FileNotFoundError):
            NEREntityExtractor("non_existent_file.txt", use_spacy=False)

    def test_export_json(self, sample_chat_file, tmp_path):
        """Test JSON export functionality."""
        extractor = NEREntityExtractor(sample_chat_file, use_spacy=False)
        entity = extractor.extract()

        output_path = tmp_path / "output.json"
        extractor.export_json(str(output_path))

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert data["isin"] == "FR001400QV82"
        assert data["counterparty"] == "BANK ABC"
        assert "extraction_timestamp" in data

    def test_all_eight_entities_extracted(self, sample_chat_file):
        """Test that all 8 required entities are extracted."""
        extractor = NEREntityExtractor(sample_chat_file, use_spacy=False)
        entity = extractor.extract()

        # All 8 required entities
        assert entity.counterparty is not None, "Counterparty missing"
        assert entity.notional is not None, "Notional missing"
        assert entity.isin is not None, "ISIN missing"
        assert entity.underlying is not None, "Underlying missing"
        assert entity.maturity is not None, "Maturity missing"
        assert entity.bid is not None, "Bid missing"
        assert entity.offer is not None, "Offer missing"
        assert entity.payment_frequency is not None, "Payment frequency missing"


class TestISINValidation:
    """Tests for ISIN format validation."""

    @pytest.fixture
    def extractor_class(self):
        return NEREntityExtractor

    def test_valid_isin_format(self, tmp_path):
        """Test that valid ISIN codes are extracted."""
        # Valid ISIN: 2 letters + 9 alphanumeric + 1 digit
        content = "Trade ref: FR001400QV82 confirmed"
        file_path = tmp_path / "test.txt"
        file_path.write_text(content)

        extractor = NEREntityExtractor(str(file_path), use_spacy=False)
        entity = extractor.extract()

        assert entity.isin == "FR001400QV82"
        assert len(entity.isin) == 12

    def test_invalid_isin_not_extracted(self, tmp_path):
        """Test that invalid ISIN-like strings are not extracted."""
        content = "Reference: ABC123 not an ISIN"
        file_path = tmp_path / "test.txt"
        file_path.write_text(content)

        extractor = NEREntityExtractor(str(file_path), use_spacy=False)
        entity = extractor.extract()

        assert entity.isin is None


class TestChatFinancialEntity:
    """Tests for ChatFinancialEntity dataclass."""

    def test_entity_creation(self):
        """Test entity creation with default values."""
        entity = ChatFinancialEntity()
        assert entity.counterparty is None
        assert entity.extraction_method == "Hybrid NER + Rules"

    def test_entity_to_dict(self):
        """Test conversion to dictionary excludes None values."""
        entity = ChatFinancialEntity(counterparty="TEST BANK", isin="US0378331005")
        data = entity.to_dict()

        assert "counterparty" in data
        assert "isin" in data
        assert "notional" not in data  # None values excluded

    def test_timestamp_auto_generated(self):
        """Test that timestamp is auto-generated."""
        entity = ChatFinancialEntity()
        assert entity.extraction_timestamp is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

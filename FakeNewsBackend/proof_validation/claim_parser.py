import re
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio

# NLP libraries
try:
    import spacy
    from spacy import displacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. Using basic text processing.")

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. Using basic text processing.")

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents a named entity found in text."""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class Claim:
    """Represents a factual claim extracted from text."""
    text: str
    confidence: float
    entities: List[Entity]
    claim_type: str
    keywords: List[str]
    source_text: str
    position: Tuple[int, int]  # Start and end positions in source text
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert claim to dictionary representation."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "entities": [
                {
                    "text": entity.text,
                    "label": entity.label,
                    "start": entity.start,
                    "end": entity.end,
                    "confidence": entity.confidence
                }
                for entity in self.entities
            ],
            "claim_type": self.claim_type,
            "keywords": self.keywords,
            "source_text": self.source_text,
            "position": self.position
        }


class ClaimParsingError(Exception):
    """Custom exception for claim parsing errors."""
    pass


class ClaimParser:
    """Advanced claim parser with NLP capabilities."""
    
    def __init__(self, language: str = "en"):
        """
        Initialize claim parser.
        
        Args:
            language: Language code for NLP processing
        """
        self.language = language
        self.nlp = None
        self.stopwords = set()
        
        # Initialize NLP components
        self._initialize_nlp()
        
        # Claim pattern definitions
        self.claim_patterns = self._load_claim_patterns()
        
        # Entity types of interest for fact-checking
        self.important_entities = {
            "PERSON", "ORG", "GPE", "DATE", "TIME", "MONEY", 
            "PERCENT", "CARDINAL", "ORDINAL", "EVENT"
        }
        
        logger.info(f"ClaimParser initialized for language: {language}")
    
    def _initialize_nlp(self) -> None:
        """Initialize NLP libraries and models."""
        if SPACY_AVAILABLE:
            try:
                # Try to load spaCy model
                model_name = f"{self.language}_core_web_sm"
                self.nlp = spacy.load(model_name)
                logger.info(f"Loaded spaCy model: {model_name}")
            except OSError:
                logger.warning(f"spaCy model {model_name} not found. Using basic processing.")
                self.nlp = None
        
        if NLTK_AVAILABLE:
            try:
                # Download required NLTK data
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                nltk.download('maxent_ne_chunker', quiet=True)
                nltk.download('words', quiet=True)
                
                self.stopwords = set(stopwords.words('english'))
                logger.info("NLTK components initialized")
            except Exception as e:
                logger.warning(f"NLTK initialization failed: {e}")
    
    def _load_claim_patterns(self) -> List[Dict[str, Any]]:
        """Load regex patterns for identifying different types of claims."""
        return [
            {
                "name": "statistical_claim",
                "pattern": r'\b\d+(?:\.\d+)?%?\s+(?:of|percent|percentage)\b|\b\d+(?:,\d{3})*(?:\.\d+)?\s+(?:people|users|cases|deaths|infections)\b',
                "confidence": 0.8,
                "type": "statistical"
            },
            {
                "name": "temporal_claim", 
                "pattern": r'\b(?:in|on|during|since|until|by)\s+(?:\d{4}|\w+\s+\d{1,2},?\s+\d{4}|\w+\s+\d{4})\b',
                "confidence": 0.7,
                "type": "temporal"
            },
            {
                "name": "causal_claim",
                "pattern": r'\b(?:causes?|leads?\s+to|results?\s+in|due\s+to|because\s+of)\b',
                "confidence": 0.6,
                "type": "causal"
            },
            {
                "name": "comparative_claim",
                "pattern": r'\b(?:more|less|higher|lower|better|worse|faster|slower)\s+than\b|\b(?:most|least|highest|lowest|best|worst)\b',
                "confidence": 0.7,
                "type": "comparative"
            },
            {
                "name": "factual_claim",
                "pattern": r'\b(?:is|are|was|were|has|have|had)\s+(?:the|a|an)?\s*(?:first|last|only|main|primary|leading)\b',
                "confidence": 0.6,
                "type": "factual"
            },
            {
                "name": "location_claim",
                "pattern": r'\bin\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\bat\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
                "confidence": 0.5,
                "type": "location"
            }
        ]
    
    async def parse_claims(self, text: str, min_confidence: float = 0.5) -> List[Claim]:
        """
        Parse and extract claims from text.
        
        Args:
            text: Input text to parse
            min_confidence: Minimum confidence threshold for claims
            
        Returns:
            List of extracted claims
        """
        if not text or not text.strip():
            return []
        
        try:
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(text)
            
            # Split into sentences
            sentences = self._split_sentences(cleaned_text)
            
            # Extract claims from each sentence
            all_claims = []
            for i, sentence in enumerate(sentences):
                sentence_claims = await self._extract_claims_from_sentence(
                    sentence, text, min_confidence
                )
                all_claims.extend(sentence_claims)
            
            # Post-process and filter claims
            filtered_claims = self._filter_and_rank_claims(all_claims, min_confidence)
            
            logger.info(f"Extracted {len(filtered_claims)} claims from text")
            return filtered_claims
            
        except Exception as e:
            logger.error(f"Error parsing claims: {e}")
            raise ClaimParsingError(f"Failed to parse claims: {e}")
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for claim extraction."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Normalize quotes
        text = re.sub(r'[""''`]', '"', text)
        
        return text
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using available NLP tools."""
        if NLTK_AVAILABLE:
            try:
                return sent_tokenize(text)
            except Exception as e:
                logger.warning(f"NLTK sentence tokenization failed: {e}")
        
        # Fallback to simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    async def _extract_claims_from_sentence(
        self, 
        sentence: str, 
        original_text: str, 
        min_confidence: float
    ) -> List[Claim]:
        """Extract claims from a single sentence."""
        claims = []
        
        # Skip very short sentences
        if len(sentence.split()) < 4:
            return claims
        
        # Extract entities
        entities = self._extract_entities(sentence)
        
        # Check against claim patterns
        for pattern_info in self.claim_patterns:
            matches = re.finditer(pattern_info["pattern"], sentence, re.IGNORECASE)
            
            for match in matches:
                if pattern_info["confidence"] >= min_confidence:
                    # Find position in original text
                    start_pos = original_text.find(sentence)
                    if start_pos == -1:
                        start_pos = 0
                    
                    # Extract keywords
                    keywords = self._extract_keywords(sentence)
                    
                    claim = Claim(
                        text=sentence.strip(),
                        confidence=pattern_info["confidence"],
                        entities=entities,
                        claim_type=pattern_info["type"],
                        keywords=keywords,
                        source_text=original_text,
                        position=(start_pos, start_pos + len(sentence))
                    )
                    
                    claims.append(claim)
                    break  # Only one pattern per sentence
        
        # If no patterns matched but sentence has important entities, consider it a potential claim
        if not claims and entities:
            important_entity_count = sum(
                1 for entity in entities 
                if entity.label in self.important_entities
            )
            
            if important_entity_count >= 2:  # At least 2 important entities
                keywords = self._extract_keywords(sentence)
                start_pos = original_text.find(sentence)
                if start_pos == -1:
                    start_pos = 0
                
                claim = Claim(
                    text=sentence.strip(),
                    confidence=0.4,  # Lower confidence for entity-based claims
                    entities=entities,
                    claim_type="entity_based",
                    keywords=keywords,
                    source_text=original_text,
                    position=(start_pos, start_pos + len(sentence))
                )
                
                if claim.confidence >= min_confidence:
                    claims.append(claim)
        
        return claims
    
    def _extract_entities(self, text: str) -> List[Entity]:
        """Extract named entities from text."""
        entities = []
        
        if self.nlp and SPACY_AVAILABLE:
            # Use spaCy for entity extraction
            doc = self.nlp(text)
            for ent in doc.ents:
                entities.append(Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=1.0  # spaCy doesn't provide confidence scores
                ))
        
        elif NLTK_AVAILABLE:
            # Use NLTK for entity extraction
            try:
                tokens = word_tokenize(text)
                pos_tags = pos_tag(tokens)
                chunks = ne_chunk(pos_tags)
                
                current_pos = 0
                for chunk in chunks:
                    if hasattr(chunk, 'label'):
                        # This is a named entity
                        entity_text = ' '.join([token for token, pos in chunk])
                        start_pos = text.find(entity_text, current_pos)
                        if start_pos != -1:
                            entities.append(Entity(
                                text=entity_text,
                                label=chunk.label(),
                                start=start_pos,
                                end=start_pos + len(entity_text),
                                confidence=0.8
                            ))
                            current_pos = start_pos + len(entity_text)
            except Exception as e:
                logger.warning(f"NLTK entity extraction failed: {e}")
        
        return entities
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove stopwords if available
        if self.stopwords:
            words = [word for word in words if word not in self.stopwords]
        
        # Remove common words manually if NLTK not available
        common_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these',
            'those', 'his', 'her', 'its', 'their', 'our', 'your', 'was', 'were',
            'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should'
        }
        
        words = [word for word in words if word not in common_words]
        
        # Return unique keywords, limited to top 10
        return list(dict.fromkeys(words))[:10]
    
    def _filter_and_rank_claims(self, claims: List[Claim], min_confidence: float) -> List[Claim]:
        """Filter and rank claims by confidence and relevance."""
        # Filter by minimum confidence
        filtered_claims = [claim for claim in claims if claim.confidence >= min_confidence]
        
        # Remove duplicate claims (same text)
        seen_texts = set()
        unique_claims = []
        for claim in filtered_claims:
            if claim.text not in seen_texts:
                unique_claims.append(claim)
                seen_texts.add(claim.text)
        
        # Sort by confidence (highest first)
        unique_claims.sort(key=lambda x: x.confidence, reverse=True)
        
        return unique_claims
    
    async def extract_key_facts(self, text: str) -> Dict[str, Any]:
        """
        Extract key factual information from text for fact-checking.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing key facts and metadata
        """
        claims = await self.parse_claims(text)
        
        # Aggregate entities across all claims
        all_entities = []
        for claim in claims:
            all_entities.extend(claim.entities)
        
        # Group entities by type
        entities_by_type = {}
        for entity in all_entities:
            if entity.label not in entities_by_type:
                entities_by_type[entity.label] = []
            entities_by_type[entity.label].append(entity.text)
        
        # Remove duplicates
        for label in entities_by_type:
            entities_by_type[label] = list(set(entities_by_type[label]))
        
        # Extract all keywords
        all_keywords = []
        for claim in claims:
            all_keywords.extend(claim.keywords)
        
        unique_keywords = list(dict.fromkeys(all_keywords))
        
        return {
            "claims": [claim.to_dict() for claim in claims],
            "entities_by_type": entities_by_type,
            "keywords": unique_keywords,
            "claim_count": len(claims),
            "high_confidence_claims": len([c for c in claims if c.confidence > 0.7]),
            "claim_types": list(set(claim.claim_type for claim in claims)),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def get_claim_summary(self, claims: List[Claim]) -> Dict[str, Any]:
        """Generate a summary of extracted claims."""
        if not claims:
            return {
                "total_claims": 0,
                "claim_types": {},
                "confidence_distribution": {},
                "top_entities": [],
                "top_keywords": []
            }
        
        # Count claim types
        claim_types = {}
        for claim in claims:
            claim_types[claim.claim_type] = claim_types.get(claim.claim_type, 0) + 1
        
        # Confidence distribution
        confidence_ranges = {"high": 0, "medium": 0, "low": 0}
        for claim in claims:
            if claim.confidence >= 0.7:
                confidence_ranges["high"] += 1
            elif claim.confidence >= 0.5:
                confidence_ranges["medium"] += 1
            else:
                confidence_ranges["low"] += 1
        
        # Top entities
        entity_counts = {}
        for claim in claims:
            for entity in claim.entities:
                entity_counts[entity.text] = entity_counts.get(entity.text, 0) + 1
        
        top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Top keywords
        keyword_counts = {}
        for claim in claims:
            for keyword in claim.keywords:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_claims": len(claims),
            "claim_types": claim_types,
            "confidence_distribution": confidence_ranges,
            "top_entities": top_entities,
            "top_keywords": top_keywords
        }


## Suggestions for Upgrade:
# 1. Integrate advanced transformer models (BERT, RoBERTa) for better claim detection and classification
# 2. Add support for multilingual claim parsing with automatic language detection
# 3. Implement claim relationship analysis to identify contradictory or supporting claims
# 4. Add machine learning-based claim importance scoring using trained models on fact-checking datasets
#!/usr/bin/env python3
"""
Cross-Checking Logic for Claim Verification
Implements sophisticated fact-checking algorithms and entity matching

Author: AI Assistant
Date: January 2025
Version: 1.0.0
"""

import os
import logging
import re
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from collections import Counter
import difflib

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available for advanced NLP")

try:
    from fuzzywuzzy import fuzz, process
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False
    logging.warning("fuzzywuzzy not available for fuzzy matching")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available")

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Extracted entity with metadata"""
    text: str
    entity_type: str  # PERSON, ORG, GPE, DATE, MONEY, etc.
    confidence: float
    start_pos: int
    end_pos: int
    normalized_form: str = ''

@dataclass
class ClaimComponent:
    """Individual component of a claim"""
    text: str
    component_type: str  # 'entity', 'number', 'date', 'statement'
    importance: float  # 0-1 scale
    entities: List[Entity]
    keywords: List[str]

@dataclass
class VerificationMatch:
    """Match between claim and source"""
    source_url: str
    source_title: str
    source_snippet: str
    match_score: float
    match_type: str  # 'exact', 'partial', 'semantic', 'contradictory'
    matched_entities: List[str]
    matched_keywords: List[str]
    credibility_score: float
    evidence_strength: str  # 'strong', 'moderate', 'weak'

@dataclass
class CrossCheckResult:
    """Result of cross-checking analysis"""
    verdict: str  # 'Real', 'Fake', 'Unverified', 'Misleading'
    confidence: float
    evidence_score: float
    supporting_matches: List[VerificationMatch]
    contradicting_matches: List[VerificationMatch]
    neutral_matches: List[VerificationMatch]
    key_findings: List[str]
    reasoning: str
    processing_time: float
    claim_components: List[ClaimComponent]
    entity_verification: Dict[str, Any]

class CrossChecker:
    """Advanced cross-checking system for claim verification"""
    
    def __init__(self):
        # Initialize NLP components
        self.nlp = None
        self.spacy_available = SPACY_AVAILABLE
        if self.spacy_available:
            try:
                # Try to load English model
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy English model loaded successfully")
            except OSError:
                try:
                    # Fallback to basic model
                    self.nlp = spacy.load("en_core_web_md")
                    logger.info("spaCy medium model loaded")
                except OSError:
                    logger.warning("No spaCy English model found")
                    self.spacy_available = False
        
        # Initialize NLTK components
        self.lemmatizer = None
        self.stop_words = set()
        if NLTK_AVAILABLE:
            try:
                # Download required NLTK data
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
                logger.info("NLTK components initialized")
            except Exception as e:
                logger.warning(f"NLTK initialization failed: {e}")
        
        # Credible source domains with reliability scores
        self.credible_sources = {
            'reuters.com': 0.95,
            'apnews.com': 0.95,
            'bbc.com': 0.90,
            'cnn.com': 0.85,
            'npr.org': 0.90,
            'theguardian.com': 0.85,
            'nytimes.com': 0.85,
            'washingtonpost.com': 0.85,
            'factcheck.org': 0.95,
            'snopes.com': 0.90,
            'politifact.com': 0.90,
            'abc.net.au': 0.85,
            'cbsnews.com': 0.80,
            'nbcnews.com': 0.80,
            'usatoday.com': 0.75,
            'wsj.com': 0.85,
            'bloomberg.com': 0.80,
            'economist.com': 0.85
        }
        
        # Fact-checking keywords
        self.verification_keywords = {
            'positive': ['true', 'confirmed', 'verified', 'accurate', 'correct', 'factual', 'authentic', 'legitimate'],
            'negative': ['false', 'fake', 'debunked', 'misleading', 'incorrect', 'hoax', 'myth', 'fabricated', 'untrue'],
            'uncertain': ['unverified', 'unclear', 'disputed', 'controversial', 'alleged', 'claimed', 'reportedly']
        }
        
        # Entity patterns for extraction
        self.entity_patterns = {
            'money': r'\$[\d,]+(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD|cents?)',
            'percentage': r'\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?\s*percent',
            'date': r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            'number': r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        }
        
        logger.info("CrossChecker initialized")
    
    def _extract_entities_spacy(self, text: str) -> List[Entity]:
        """Extract entities using spaCy"""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                entity = Entity(
                    text=ent.text,
                    entity_type=ent.label_,
                    confidence=0.8,  # spaCy doesn't provide confidence scores
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    normalized_form=ent.text.lower().strip()
                )
                entities.append(entity)
            
            return entities
        except Exception as e:
            logger.warning(f"spaCy entity extraction failed: {e}")
            return []
    
    def _extract_entities_regex(self, text: str) -> List[Entity]:
        """Extract entities using regex patterns"""
        entities = []
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity = Entity(
                    text=match.group(),
                    entity_type=entity_type.upper(),
                    confidence=0.7,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    normalized_form=match.group().lower().strip()
                )
                entities.append(entity)
        
        return entities
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        keywords = []
        
        if NLTK_AVAILABLE and self.lemmatizer:
            try:
                # Tokenize and process with NLTK
                tokens = word_tokenize(text.lower())
                
                # Remove stopwords and short words
                filtered_tokens = [
                    self.lemmatizer.lemmatize(token)
                    for token in tokens
                    if token.isalpha() and len(token) > 2 and token not in self.stop_words
                ]
                
                # Get most common keywords
                word_freq = Counter(filtered_tokens)
                keywords = [word for word, freq in word_freq.most_common(20)]
            
            except Exception as e:
                logger.warning(f"NLTK keyword extraction failed: {e}")
        
        # Fallback: simple keyword extraction
        if not keywords:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            word_freq = Counter(words)
            keywords = [word for word, freq in word_freq.most_common(20)]
        
        return keywords
    
    def _analyze_claim_components(self, claim: str) -> List[ClaimComponent]:
        """Break down claim into analyzable components"""
        components = []
        
        # Extract entities
        spacy_entities = self._extract_entities_spacy(claim)
        regex_entities = self._extract_entities_regex(claim)
        all_entities = spacy_entities + regex_entities
        
        # Extract keywords
        keywords = self._extract_keywords(claim)
        
        # Split claim into sentences
        if NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(claim)
            except:
                sentences = claim.split('. ')
        else:
            sentences = claim.split('. ')
        
        # Create components for each sentence
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 10:
                continue
            
            # Find entities in this sentence
            sentence_entities = [
                entity for entity in all_entities
                if entity.start_pos >= sum(len(s) + 2 for s in sentences[:i])
                and entity.end_pos <= sum(len(s) + 2 for s in sentences[:i+1])
            ]
            
            # Calculate importance based on entities and keywords
            importance = 0.5  # Base importance
            if sentence_entities:
                importance += 0.3
            if any(keyword in sentence.lower() for keyword in keywords[:5]):
                importance += 0.2
            
            component = ClaimComponent(
                text=sentence.strip(),
                component_type='statement',
                importance=min(importance, 1.0),
                entities=sentence_entities,
                keywords=[kw for kw in keywords if kw in sentence.lower()]
            )
            components.append(component)
        
        return components
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        if FUZZYWUZZY_AVAILABLE:
            try:
                # Use fuzzy string matching
                ratio = fuzz.ratio(text1.lower(), text2.lower()) / 100.0
                token_sort_ratio = fuzz.token_sort_ratio(text1.lower(), text2.lower()) / 100.0
                token_set_ratio = fuzz.token_set_ratio(text1.lower(), text2.lower()) / 100.0
                
                # Return weighted average
                return (ratio * 0.3 + token_sort_ratio * 0.4 + token_set_ratio * 0.3)
            except Exception as e:
                logger.warning(f"Fuzzy matching failed: {e}")
        
        # Fallback: simple similarity using difflib
        return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def _find_entity_matches(self, claim_entities: List[Entity], source_text: str) -> List[str]:
        """Find entity matches between claim and source"""
        matches = []
        source_lower = source_text.lower()
        
        for entity in claim_entities:
            entity_text = entity.normalized_form
            
            # Exact match
            if entity_text in source_lower:
                matches.append(entity.text)
                continue
            
            # Fuzzy match for names and organizations
            if entity.entity_type in ['PERSON', 'ORG', 'GPE'] and FUZZYWUZZY_AVAILABLE:
                try:
                    # Extract potential matches from source
                    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', source_text)
                    best_match = process.extractOne(entity.text, words)
                    
                    if best_match and best_match[1] > 80:  # 80% similarity threshold
                        matches.append(entity.text)
                except Exception:
                    pass
        
        return matches
    
    def _analyze_source_credibility(self, source_url: str, source_title: str, source_snippet: str) -> Tuple[float, str]:
        """Analyze credibility of a source"""
        from urllib.parse import urlparse
        
        domain = urlparse(source_url).netloc.lower()
        
        # Base credibility from domain
        base_credibility = self.credible_sources.get(domain, 0.5)
        
        # Adjust based on content indicators
        content_text = (source_title + ' ' + source_snippet).lower()
        
        # Check for fact-checking indicators
        verification_score = 0
        for category, keywords in self.verification_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in content_text)
            if category == 'positive':
                verification_score += matches * 0.1
            elif category == 'negative':
                verification_score -= matches * 0.1
            elif category == 'uncertain':
                verification_score -= matches * 0.05
        
        # Final credibility score
        final_credibility = max(0.0, min(1.0, base_credibility + verification_score))
        
        # Determine evidence strength
        if final_credibility > 0.8:
            strength = 'strong'
        elif final_credibility > 0.6:
            strength = 'moderate'
        else:
            strength = 'weak'
        
        return final_credibility, strength
    
    def _classify_match_type(self, claim_text: str, source_text: str, similarity_score: float) -> str:
        """Classify the type of match between claim and source"""
        claim_lower = claim_text.lower()
        source_lower = source_text.lower()
        
        # Check for contradictory indicators
        contradictory_patterns = [
            (r'\bnot\s+true\b', r'\btrue\b'),
            (r'\bfalse\b', r'\btrue\b'),
            (r'\bdebunked\b', r'\bconfirmed\b'),
            (r'\bhoax\b', r'\breal\b')
        ]
        
        for neg_pattern, pos_pattern in contradictory_patterns:
            if re.search(neg_pattern, source_lower) and re.search(pos_pattern, claim_lower):
                return 'contradictory'
            if re.search(pos_pattern, source_lower) and re.search(neg_pattern, claim_lower):
                return 'contradictory'
        
        # Classify based on similarity
        if similarity_score > 0.8:
            return 'exact'
        elif similarity_score > 0.6:
            return 'partial'
        elif similarity_score > 0.4:
            return 'semantic'
        else:
            return 'weak'
    
    def cross_check_claim(self, claim: str, search_results: List[Dict[str, Any]]) -> CrossCheckResult:
        """Perform comprehensive cross-checking of claim against search results"""
        start_time = time.time()
        
        try:
            # Analyze claim components
            claim_components = self._analyze_claim_components(claim)
            
            # Extract all entities from claim
            all_entities = []
            for component in claim_components:
                all_entities.extend(component.entities)
            
            # Process search results
            supporting_matches = []
            contradicting_matches = []
            neutral_matches = []
            
            for result in search_results:
                source_url = result.get('url', '')
                source_title = result.get('title', '')
                source_snippet = result.get('snippet', '')
                
                if not source_title and not source_snippet:
                    continue
                
                # Calculate similarity
                title_similarity = self._calculate_text_similarity(claim, source_title)
                snippet_similarity = self._calculate_text_similarity(claim, source_snippet)
                overall_similarity = max(title_similarity, snippet_similarity)
                
                # Find entity matches
                title_entity_matches = self._find_entity_matches(all_entities, source_title)
                snippet_entity_matches = self._find_entity_matches(all_entities, source_snippet)
                all_entity_matches = list(set(title_entity_matches + snippet_entity_matches))
                
                # Find keyword matches
                claim_keywords = self._extract_keywords(claim)
                source_text = (source_title + ' ' + source_snippet).lower()
                keyword_matches = [kw for kw in claim_keywords if kw in source_text]
                
                # Analyze source credibility
                credibility, evidence_strength = self._analyze_source_credibility(
                    source_url, source_title, source_snippet
                )
                
                # Classify match type
                match_type = self._classify_match_type(
                    claim, source_title + ' ' + source_snippet, overall_similarity
                )
                
                # Calculate match score
                match_score = (
                    overall_similarity * 0.4 +
                    (len(all_entity_matches) / max(len(all_entities), 1)) * 0.3 +
                    (len(keyword_matches) / max(len(claim_keywords), 1)) * 0.2 +
                    credibility * 0.1
                )
                
                # Create verification match
                verification_match = VerificationMatch(
                    source_url=source_url,
                    source_title=source_title,
                    source_snippet=source_snippet,
                    match_score=match_score,
                    match_type=match_type,
                    matched_entities=all_entity_matches,
                    matched_keywords=keyword_matches,
                    credibility_score=credibility,
                    evidence_strength=evidence_strength
                )
                
                # Categorize match
                if match_type == 'contradictory':
                    contradicting_matches.append(verification_match)
                elif match_score > 0.6 and credibility > 0.6:
                    supporting_matches.append(verification_match)
                else:
                    neutral_matches.append(verification_match)
            
            # Sort matches by score
            supporting_matches.sort(key=lambda x: x.match_score, reverse=True)
            contradicting_matches.sort(key=lambda x: x.match_score, reverse=True)
            neutral_matches.sort(key=lambda x: x.match_score, reverse=True)
            
            # Calculate evidence scores
            supporting_score = sum(
                match.match_score * match.credibility_score
                for match in supporting_matches
            ) / max(len(supporting_matches), 1)
            
            contradicting_score = sum(
                match.match_score * match.credibility_score
                for match in contradicting_matches
            ) / max(len(contradicting_matches), 1)
            
            # Determine verdict
            if supporting_score > contradicting_score * 1.5 and supporting_matches:
                verdict = 'Real'
                confidence = min(supporting_score, 0.95)
            elif contradicting_score > supporting_score * 1.5 and contradicting_matches:
                verdict = 'Fake'
                confidence = min(contradicting_score, 0.95)
            elif supporting_score > 0.4 and contradicting_score > 0.4:
                verdict = 'Misleading'
                confidence = 0.6
            else:
                verdict = 'Unverified'
                confidence = 0.3
            
            # Generate key findings
            key_findings = []
            if supporting_matches:
                key_findings.append(f"Found {len(supporting_matches)} supporting sources")
            if contradicting_matches:
                key_findings.append(f"Found {len(contradicting_matches)} contradicting sources")
            if all_entities:
                verified_entities = set()
                for match in supporting_matches + contradicting_matches:
                    verified_entities.update(match.matched_entities)
                key_findings.append(f"Verified {len(verified_entities)}/{len(all_entities)} key entities")
            
            # Generate reasoning
            reasoning_parts = []
            if verdict == 'Real':
                reasoning_parts.append(f"Multiple credible sources support the claim")
            elif verdict == 'Fake':
                reasoning_parts.append(f"Credible sources contradict the claim")
            elif verdict == 'Misleading':
                reasoning_parts.append(f"Mixed evidence suggests the claim may be partially true or misleading")
            else:
                reasoning_parts.append(f"Insufficient reliable evidence to verify the claim")
            
            reasoning = '. '.join(reasoning_parts)
            
            # Entity verification summary
            entity_verification = {
                'total_entities': len(all_entities),
                'verified_entities': len(set(
                    entity for match in supporting_matches + contradicting_matches
                    for entity in match.matched_entities
                )),
                'entity_types': list(set(entity.entity_type for entity in all_entities))
            }
            
            return CrossCheckResult(
                verdict=verdict,
                confidence=confidence,
                evidence_score=max(supporting_score, contradicting_score),
                supporting_matches=supporting_matches[:10],  # Top 10
                contradicting_matches=contradicting_matches[:10],  # Top 10
                neutral_matches=neutral_matches[:5],  # Top 5
                key_findings=key_findings,
                reasoning=reasoning,
                processing_time=time.time() - start_time,
                claim_components=claim_components,
                entity_verification=entity_verification
            )
        
        except Exception as e:
            logger.error(f"Cross-checking error: {e}")
            return CrossCheckResult(
                verdict='Unverified',
                confidence=0.0,
                evidence_score=0.0,
                supporting_matches=[],
                contradicting_matches=[],
                neutral_matches=[],
                key_findings=[f'Error during analysis: {str(e)}'],
                reasoning='Analysis failed due to technical error',
                processing_time=time.time() - start_time,
                claim_components=[],
                entity_verification={}
            )

# Global instance
cross_checker = CrossChecker()

# Convenience functions
def verify_claim_against_sources(claim: str, search_results: List[Dict[str, Any]]) -> CrossCheckResult:
    """Verify claim against search results"""
    return cross_checker.cross_check_claim(claim, search_results)

def extract_claim_entities(claim: str) -> List[Entity]:
    """Extract entities from claim text"""
    spacy_entities = cross_checker._extract_entities_spacy(claim)
    regex_entities = cross_checker._extract_entities_regex(claim)
    return spacy_entities + regex_entities

def get_nlp_status() -> Dict[str, bool]:
    """Get status of NLP components"""
    return {
        'spacy': cross_checker.spacy_available and cross_checker.nlp is not None,
        'nltk': NLTK_AVAILABLE and cross_checker.lemmatizer is not None,
        'fuzzywuzzy': FUZZYWUZZY_AVAILABLE
    }
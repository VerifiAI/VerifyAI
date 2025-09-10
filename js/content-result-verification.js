/**
 * Real-time Fake News Verification System - Content Result Module
 * Analyzes proof objects and renders comprehensive verification results
 * 100% Browser-native vanilla JavaScript
 */

// ============================================================================
// SUBTASKS LIST:
// ============================================================================
/*
1. Accept proofsArray input with proof objects: {url, title, snippet, domain, credibility_score, fact_check_verdict}
2. Label each proof as "REAL", "FAKE", or "AMBIGUOUS" based on fact_check_verdict or snippet analysis
3. Calculate weighted consensus score using credibility_score weights
4. Apply decision logic for final verdict based on consensus thresholds
5. Calculate confidence percentage based on consensus strength and proof count
6. Render detailed results showing each proof with badges and overall verdict
7. Handle edge cases (empty array, all ambiguous proofs)
8. Provide clear commenting and modular structure
9. Output final summary of analysis methodology
*/

// ============================================================================
// SUBTASK 1: INPUT VALIDATION AND STRUCTURE
// ============================================================================

/**
 * Validates and prepares the proofsArray for analysis
 * @param {Array} proofsArray - Array of proof objects
 * @returns {Object} Validation result with status and processed array
 */
function validateProofsInput(proofsArray) {
    if (!Array.isArray(proofsArray)) {
        return {
            valid: false,
            error: 'proofsArray must be an array',
            proofs: []
        };
    }
    
    if (proofsArray.length === 0) {
        return {
            valid: false,
            error: 'No proofs available for analysis',
            proofs: []
        };
    }
    
    // Validate each proof object structure
    const validatedProofs = proofsArray.filter(proof => {
        return proof && 
               typeof proof.url === 'string' && 
               typeof proof.title === 'string' && 
               typeof proof.snippet === 'string' && 
               typeof proof.domain === 'string' && 
               typeof proof.credibility_score === 'number';
    });
    
    return {
        valid: validatedProofs.length > 0,
        error: validatedProofs.length === 0 ? 'No valid proof objects found' : null,
        proofs: validatedProofs
    };
}

// ============================================================================
// SUBTASK 2: VERDICT DETERMINATION LOGIC
// ============================================================================

/**
 * Determines verdict for a single proof based on fact_check_verdict or snippet analysis
 * @param {Object} proof - Single proof object
 * @returns {string} "REAL", "FAKE", or "AMBIGUOUS"
 */
function determineProofVerdict(proof) {
    // First check if fact_check_verdict exists and is meaningful
    if (proof.fact_check_verdict && typeof proof.fact_check_verdict === 'string') {
        const verdict = proof.fact_check_verdict.toLowerCase();
        
        // Check for REAL indicators
        if (verdict.includes('true') || verdict.includes('correct') || 
            verdict.includes('accurate') || verdict.includes('verified')) {
            return 'REAL';
        }
        
        // Check for FAKE indicators
        if (verdict.includes('false') || verdict.includes('fake') || 
            verdict.includes('incorrect') || verdict.includes('misleading')) {
            return 'FAKE';
        }
    }
    
    // Enhanced snippet analysis using comprehensive keyword heuristics
    const fullText = `${proof.title} ${proof.snippet}`.toLowerCase();
    
    // FAKE indicators (comprehensive list)
    const fakeKeywords = [
        'not true', 'debunked', 'fake', 'hoax', 'false', 'misleading',
        'misinformation', 'disinformation', 'fabricated', 'unverified',
        'baseless', 'conspiracy', 'myth', 'rumor', 'incorrect', 'scam',
        'fraud', 'lies', 'untrue', 'bogus', 'phony', 'deceptive',
        'propaganda', 'manipulation', 'distorted', 'exaggerated'
    ];
    
    // REAL indicators (comprehensive list)
    const realKeywords = [
        'confirmed', 'authentic', 'true', 'verified', 'accurate',
        'legitimate', 'factual', 'evidence shows', 'research confirms',
        'studies show', 'experts confirm', 'official statement',
        'documented', 'proven', 'substantiated', 'corroborated',
        'reliable source', 'peer reviewed', 'scientific study',
        'government data', 'official report', 'credible evidence'
    ];
    
    // Enhanced scoring with weighted matches
    let fakeScore = 0;
    let realScore = 0;
    
    // Check for fake indicators
    fakeKeywords.forEach(keyword => {
        if (fullText.includes(keyword)) {
            fakeScore += keyword.length > 8 ? 2 : 1; // Longer phrases get more weight
        }
    });
    
    // Check for real indicators
    realKeywords.forEach(keyword => {
        if (fullText.includes(keyword)) {
            realScore += keyword.length > 8 ? 2 : 1; // Longer phrases get more weight
        }
    });
    
    // Domain-based credibility boost
    const domain = proof.domain.toLowerCase();
    const trustedDomains = [
        'reuters.com', 'ap.org', 'bbc.com', 'npr.org', 'pbs.org',
        'factcheck.org', 'snopes.com', 'politifact.com', 'cnn.com',
        'nytimes.com', 'washingtonpost.com', 'wsj.com', 'theguardian.com'
    ];
    
    if (trustedDomains.some(trusted => domain.includes(trusted))) {
        realScore += 1; // Boost for trusted domains
    }
    
    // Credibility score influence (higher credibility = more likely real)
    if (proof.credibility_score > 0.7) {
        realScore += 1;
    } else if (proof.credibility_score < 0.3) {
        fakeScore += 1;
    }
    
    // Determine verdict with minimum threshold
    const scoreDifference = Math.abs(realScore - fakeScore);
    
    if (fakeScore > realScore && scoreDifference >= 1) {
        return 'FAKE';
    } else if (realScore > fakeScore && scoreDifference >= 1) {
        return 'REAL';
    } else {
        // If scores are too close or both zero, use credibility as tiebreaker
        if (proof.credibility_score > 0.6) {
            return 'REAL';
        } else if (proof.credibility_score < 0.4) {
            return 'FAKE';
        } else {
            return 'AMBIGUOUS';
        }
    }
}

/**
 * Processes all proofs and attaches determined_verdict property
 * @param {Array} proofs - Array of proof objects
 * @returns {Array} Proofs with determined_verdict attached
 */
function labelAllProofs(proofs) {
    console.log('🏷️ Labeling proofs with verdicts...');
    
    return proofs.map((proof, index) => {
        const determined_verdict = determineProofVerdict(proof);
        
        console.log(`[PROOF ${index + 1}] ${proof.domain}:`);
        console.log(`  - Title: ${proof.title.substring(0, 60)}...`);
        console.log(`  - Credibility: ${Math.round(proof.credibility_score * 100)}%`);
        console.log(`  - Verdict: ${determined_verdict}`);
        console.log(`  - Fact Check: ${proof.fact_check_verdict || 'None'}`);
        
        return {
            ...proof,
            determined_verdict
        };
    });
}

// ============================================================================
// SUBTASK 3: CONSENSUS SCORING ALGORITHM
// ============================================================================

/**
 * Calculates weighted consensus score from labeled proofs
 * @param {Array} labeledProofs - Proofs with determined_verdict
 * @returns {Object} Consensus data including score and breakdown
 */
function calculateConsensus(labeledProofs) {
    let totalWeightedScore = 0;
    let totalCredibilityWeight = 0;
    let realCount = 0;
    let fakeCount = 0;
    let ambiguousCount = 0;
    
    console.log('📊 Calculating consensus from verdicts...');
    
    labeledProofs.forEach((proof, index) => {
        const credibility = proof.credibility_score;
        
        switch (proof.determined_verdict) {
            case 'REAL':
                totalWeightedScore += credibility; // +1 × credibility_score
                totalCredibilityWeight += credibility;
                realCount++;
                console.log(`[CONSENSUS] Proof ${index + 1}: REAL (+${credibility.toFixed(2)})`);
                break;
            case 'FAKE':
                totalWeightedScore -= credibility; // -1 × credibility_score
                totalCredibilityWeight += credibility;
                fakeCount++;
                console.log(`[CONSENSUS] Proof ${index + 1}: FAKE (-${credibility.toFixed(2)})`);
                break;
            case 'AMBIGUOUS':
                // Disregard for scoring but count for statistics
                ambiguousCount++;
                console.log(`[CONSENSUS] Proof ${index + 1}: AMBIGUOUS (ignored)`);
                break;
        }
    });
    
    // Calculate consensus score (in range [-1, 1])
    const consensus = totalCredibilityWeight > 0 ? 
        totalWeightedScore / totalCredibilityWeight : 0;
    
    console.log(`[CONSENSUS] Final Score: ${consensus.toFixed(3)} (Real: ${realCount}, Fake: ${fakeCount}, Ambiguous: ${ambiguousCount})`);
    
    return {
        consensus,
        totalWeightedScore,
        totalCredibilityWeight,
        realCount,
        fakeCount,
        ambiguousCount,
        nonAmbiguousCount: realCount + fakeCount
    };
}

// ============================================================================
// SUBTASK 4: DECISION LOGIC FOR FINAL VERDICT
// ============================================================================

/**
 * Applies decision logic to determine final verdict
 * @param {number} consensus - Consensus score in range [-1, 1]
 * @returns {Object} Final verdict with label and reasoning
 */
function determineFinalVerdict(consensus) {
    if (consensus >= 0.5) {
        return {
            label: 'REAL (High Confidence)',
            type: 'REAL',
            confidence_level: 'HIGH',
            reasoning: 'Strong evidence supports authenticity'
        };
    } else if (consensus >= 0.2) {
        return {
            label: 'LIKELY REAL (Moderate Confidence)',
            type: 'REAL',
            confidence_level: 'MODERATE',
            reasoning: 'Evidence leans toward authenticity'
        };
    } else if (consensus <= -0.5) {
        return {
            label: 'FAKE (High Confidence)',
            type: 'FAKE',
            confidence_level: 'HIGH',
            reasoning: 'Strong evidence indicates misinformation'
        };
    } else if (consensus <= -0.2) {
        return {
            label: 'LIKELY FAKE (Moderate Confidence)',
            type: 'FAKE',
            confidence_level: 'MODERATE',
            reasoning: 'Evidence suggests potential misinformation'
        };
    } else {
        return {
            label: 'MIXED EVIDENCE (Inconclusive)',
            type: 'AMBIGUOUS',
            confidence_level: 'LOW',
            reasoning: 'Sources show conflicting or insufficient evidence'
        };
    }
}

// ============================================================================
// SUBTASK 5: CONFIDENCE CALCULATION
// ============================================================================

/**
 * Calculates confidence percentage based on consensus and proof count
 * @param {number} consensus - Consensus score
 * @param {number} nonAmbiguousCount - Number of non-ambiguous proofs
 * @returns {number} Confidence percentage (0-100)
 */
function calculateConfidence(consensus, nonAmbiguousCount) {
    // Enhanced confidence calculation with better scaling
    const baseConfidence = Math.abs(consensus);
    
    // More generous proof factor - reaches 1.0 with fewer proofs
    const proofFactor = Math.min(nonAmbiguousCount / 5, 1); // Cap at 1, but reaches it faster
    
    // Boost confidence for stronger consensus
    let confidenceMultiplier = 1;
    if (Math.abs(consensus) >= 0.5) {
        confidenceMultiplier = 1.2; // 20% boost for strong consensus
    } else if (Math.abs(consensus) >= 0.3) {
        confidenceMultiplier = 1.1; // 10% boost for moderate consensus
    }
    
    // Calculate final confidence
    let confidence = baseConfidence * proofFactor * confidenceMultiplier;
    
    // Ensure minimum confidence for any non-zero consensus
    if (consensus !== 0 && confidence < 0.25) {
        confidence = 0.25; // Minimum 25% confidence
    }
    
    // Clamp to [0, 1] and convert to percentage
    return Math.round(Math.max(0, Math.min(1, confidence)) * 100);
}

// ============================================================================
// SUBTASK 6: RENDERING FUNCTIONS
// ============================================================================

/**
 * Creates HTML for individual proof display
 * @param {Object} proof - Labeled proof object
 * @param {number} index - Proof index
 * @returns {string} HTML string for proof card
 */
function createProofHTML(proof, index) {
    const verdictClass = proof.determined_verdict.toLowerCase();
    const credibilityPercent = Math.round(proof.credibility_score * 100);
    
    return `
        <div class="proof-card" data-verdict="${verdictClass}">
            <div class="proof-header">
                <span class="proof-index">#${index + 1}</span>
                <span class="verdict-badge verdict-${verdictClass}">
                    ${proof.determined_verdict}
                </span>
                <span class="credibility-score">${credibilityPercent}%</span>
            </div>
            <div class="proof-content">
                <h4 class="proof-title">
                    <a href="${proof.url}" target="_blank" rel="noopener">
                        ${proof.title}
                    </a>
                </h4>
                <div class="proof-domain">${proof.domain}</div>
                <div class="proof-snippet">${proof.snippet}</div>
                ${proof.fact_check_verdict ? 
                    `<div class="fact-check-verdict">
                        <strong>Fact Check:</strong> ${proof.fact_check_verdict}
                    </div>` : ''}
            </div>
        </div>
    `;
}

/**
 * Creates HTML for overall verdict summary
 * @param {Object} finalVerdict - Final verdict object
 * @param {number} confidence - Confidence percentage
 * @param {Object} consensusData - Consensus calculation data
 * @returns {string} HTML string for verdict summary
 */
function createVerdictSummaryHTML(finalVerdict, confidence, consensusData) {
    const verdictClass = finalVerdict.type.toLowerCase();
    
    return `
        <div class="verdict-summary">
            <div class="final-verdict verdict-${verdictClass}">
                <div class="verdict-label">${finalVerdict.label}</div>
                <div class="confidence-score">${confidence}% Confidence</div>
                <div class="verdict-reasoning">${finalVerdict.reasoning}</div>
            </div>
            
            <div class="analysis-stats">
                <div class="stat-item">
                    <span class="stat-label">Total Sources:</span>
                    <span class="stat-value">${consensusData.realCount + consensusData.fakeCount + consensusData.ambiguousCount}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Supporting (Real):</span>
                    <span class="stat-value stat-real">${consensusData.realCount}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Contradicting (Fake):</span>
                    <span class="stat-value stat-fake">${consensusData.fakeCount}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Inconclusive:</span>
                    <span class="stat-value stat-ambiguous">${consensusData.ambiguousCount}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Avg. Credibility:</span>
                    <span class="stat-value">${Math.round((consensusData.totalCredibilityWeight / consensusData.nonAmbiguousCount || 0) * 100)}%</span>
                </div>
            </div>
        </div>
    `;
}

// ============================================================================
// SUBTASK 7: ERROR HANDLING
// ============================================================================

/**
 * Creates error/warning display for edge cases
 * @param {string} errorType - Type of error
 * @param {string} message - Error message
 * @returns {string} HTML string for error display
 */
function createErrorHTML(errorType, message) {
    return `
        <div class="verification-error ${errorType}">
            <div class="error-icon">⚠️</div>
            <div class="error-content">
                <h3>Verification ${errorType === 'error' ? 'Error' : 'Warning'}</h3>
                <p>${message}</p>
                ${errorType === 'warning' ? 
                    '<p>Please try analyzing different content or check back later for more sources.</p>' : ''}
            </div>
        </div>
    `;
}

// ============================================================================
// SUBTASK 8: MAIN VERIFICATION FUNCTION
// ============================================================================

/**
 * Main function that orchestrates the entire verification process
 * @param {Array} proofsArray - Array of proof objects from global scope
 * @returns {Object} Complete verification results
 */
function executeContentResultVerification(proofsArray) {
    console.log('🔍 Starting Content Result Verification Analysis...');
    
    // Step 1: Validate input
    const validation = validateProofsInput(proofsArray);
    if (!validation.valid) {
        console.error('❌ Validation failed:', validation.error);
        return {
            success: false,
            error: validation.error,
            html: createErrorHTML('error', validation.error)
        };
    }
    
    // Step 2: Label all proofs
    console.log('🏷️ Labeling proofs with verdicts...');
    const labeledProofs = labelAllProofs(validation.proofs);
    
    // Step 3: Calculate consensus
    console.log('📊 Calculating weighted consensus...');
    const consensusData = calculateConsensus(labeledProofs);
    
    // Check for all-ambiguous case
    if (consensusData.nonAmbiguousCount === 0) {
        const warningMsg = 'All sources are inconclusive. Unable to determine content authenticity.';
        console.warn('⚠️', warningMsg);
        return {
            success: false,
            warning: warningMsg,
            html: createErrorHTML('warning', warningMsg)
        };
    }
    
    // Step 4: Determine final verdict
    console.log('⚖️ Determining final verdict...');
    const finalVerdict = determineFinalVerdict(consensusData.consensus);
    
    // Step 5: Calculate confidence
    console.log('📈 Calculating confidence score...');
    const confidence = calculateConfidence(consensusData.consensus, consensusData.nonAmbiguousCount);
    
    // Step 6: Generate HTML
    console.log('🎨 Rendering verification results...');
    const proofsHTML = labeledProofs.map((proof, index) => createProofHTML(proof, index)).join('');
    const summaryHTML = createVerdictSummaryHTML(finalVerdict, confidence, consensusData);
    
    const completeHTML = `
        <div class="content-verification-results">
            ${summaryHTML}
            <div class="proofs-section">
                <h3>Source Analysis (${labeledProofs.length} sources)</h3>
                <div class="proofs-container">
                    ${proofsHTML}
                </div>
            </div>
        </div>
    `;
    
    console.log('✅ Content Result Verification Complete!');
    
    return {
        success: true,
        finalVerdict,
        confidence,
        consensusData,
        labeledProofs,
        html: completeHTML
    };
}

// ============================================================================
// SUBTASK 9: INTEGRATION AND RENDERING
// ============================================================================

/**
 * Renders verification results to the Content Result section
 * @param {string} containerId - ID of the container element
 * @param {Array} proofsArray - Global proofsArray variable
 */
function renderContentResultVerification(containerId = 'contentResultPanel', proofsArray = window.proofsArray) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error('❌ Content Result container not found:', containerId);
        return;
    }
    
    // Show loading state
    container.innerHTML = '<div class="verification-loading">🔍 Analyzing sources...</div>';
    
    // Execute verification (with small delay for UX)
    setTimeout(() => {
        const results = executeContentResultVerification(proofsArray);
        container.innerHTML = results.html;
        
        // Log summary to console
        if (results.success) {
            console.log('📋 VERIFICATION SUMMARY:');
            console.log(`   Final Verdict: ${results.finalVerdict.label}`);
            console.log(`   Confidence: ${results.confidence}%`);
            console.log(`   Sources Analyzed: ${results.labeledProofs.length}`);
            console.log(`   Real: ${results.consensusData.realCount}, Fake: ${results.consensusData.fakeCount}, Ambiguous: ${results.consensusData.ambiguousCount}`);
        }
    }, 500);
}

// ============================================================================
// FINAL OUTPUT: BROWSER INTEGRATION
// ============================================================================

/**
 * Initialize the Content Result Verification system
 * This function should be called when the Content Result section is triggered
 */
function initializeContentResultVerification() {
    // Check if proofsArray is available globally
    if (typeof window.proofsArray === 'undefined') {
        console.warn('⚠️ proofsArray not found in global scope. Please ensure proofs are fetched first.');
        return false;
    }
    
    // Render the verification results
    renderContentResultVerification();
    return true;
}

// Export functions for module usage (if needed)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        executeContentResultVerification,
        renderContentResultVerification,
        initializeContentResultVerification
    };
}

// ============================================================================
// FINAL OUTPUT SUMMARY
// ============================================================================
/*
FINAL OUTPUT SUMMARY:

This Content Result Verification system provides:

1. INPUT ANALYSIS: Validates proofsArray and processes each proof object
2. VERDICT DETERMINATION: Uses fact_check_verdict or robust keyword heuristics to label each source
3. CONSENSUS SCORING: Calculates weighted consensus using credibility scores
4. DECISION LOGIC: Applies threshold-based logic for final verdict determination
5. CONFIDENCE CALCULATION: Computes confidence based on consensus strength and proof count
6. COMPREHENSIVE RENDERING: Displays each source with badges and overall verdict summary
7. ERROR HANDLING: Manages edge cases like empty arrays or all-ambiguous results
8. BROWSER INTEGRATION: 100% vanilla JS, ready for immediate deployment

Each source is individually analyzed and presented with:
- Clickable title linking to original URL
- Domain and credibility percentage
- Color-coded verdict badge (REAL/FAKE/AMBIGUOUS)
- Original snippet and fact-check verdict (if available)

The overall decision is presented with:
- Final verdict with confidence percentage
- Detailed statistics breakdown
- Clear reasoning for the determination

To use: Call initializeContentResultVerification() when Content Result section is triggered.
Requires: Global proofsArray variable with fetched proof objects.
*/
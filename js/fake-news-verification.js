/**
 * Real-Time Fake News Verification App
 * Vanilla JavaScript implementation with API integration
 * Author: Trae AI
 * 
 * NUMBERED LIST OF SUBTASKS:
 * 1. Parse user input (text, URL, image handling)
 * 2. Implement fetchNewsResults() with parallel API calls
 * 3. Merge, deduplicate, and rank results with domain trust scoring
 * 4. Extract top 10 results with fact-checker detection
 * 5. Compute credibility metrics and confidence scores
 * 6. Dynamically render results as clickable cards
 * 7. Implement error handling and loading states
 * 8. Add comprehensive comments for readability
 * 9. Final output summary and integration
 */

// =============================================================================
// SUBTASK 1: INPUT PARSING AND VALIDATION
// =============================================================================

/**
 * Parse and validate user input for analysis
 * Handles text, URL, and image inputs with appropriate validation
 * @param {string} inputType - Type of input: 'text', 'url', or 'image'
 * @param {string|File} inputValue - The actual input value
 * @returns {Object} Parsed input object with type and content
 */
function parseUserInput(inputType, inputValue) {
    console.log(`[SUBTASK 1] Parsing input type: ${inputType}`);
    
    const result = {
        type: inputType,
        content: null,
        isValid: false,
        error: null
    };
    
    try {
        switch (inputType) {
            case 'text':
                if (!inputValue || inputValue.trim().length < 10) {
                    result.error = 'Text must be at least 10 characters long';
                    return result;
                }
                result.content = inputValue.trim();
                result.isValid = true;
                break;
                
            case 'url':
                try {
                    const url = new URL(inputValue);
                    if (!['http:', 'https:'].includes(url.protocol)) {
                        result.error = 'URL must use HTTP or HTTPS protocol';
                        return result;
                    }
                    result.content = inputValue;
                    result.isValid = true;
                } catch (e) {
                    result.error = 'Invalid URL format';
                    return result;
                }
                break;
                
            case 'image':
                // For image input, show not supported message as requested
                result.error = 'Image analysis not supported in this demo. Please use text or URL input.';
                return result;
                
            default:
                result.error = 'Unsupported input type';
                return result;
        }
        
        console.log(`[SUBTASK 1] Input parsed successfully:`, result);
        return result;
        
    } catch (error) {
        console.error(`[SUBTASK 1] Error parsing input:`, error);
        result.error = `Parsing error: ${error.message}`;
        return result;
    }
}

// =============================================================================
// SUBTASK 2: PARALLEL API CALLS TO NEWS SOURCES
// =============================================================================

/**
 * Fetch news results from multiple APIs in parallel
 * Queries NewsAPI, SerperAPI, and NewsData.io simultaneously
 * @param {string} query - Search query for news verification
 * @returns {Promise<Array>} Array of news results from all sources
 */
async function fetchNewsResults(query) {
    console.log(`[SUBTASK 2] Starting parallel API calls for query: "${query}"`);
    
    // API Configuration
    const APIs = {
        newsAPI: {
            key: '03bc9e41ea5a4bec9e4ec676be5685e3',
            url: 'https://newsapi.org/v2/everything',
            name: 'NewsAPI'
        },
        serperAPI: {
            key: '5a41110616660e5886d1f1d62fee45f23cc27c73d4ddb1c92dade5b057cf21eb',
            url: 'https://serpapi.com',
            name: 'SerpApi'
        },
        newsDataIO: {
            key: 'pub_68087226ea714656815ec70dad18a7ab',
            url: 'https://newsdata.io/api/1/news',
            name: 'NewsData.io'
        }
    };
    
    /**
     * Fetch from NewsAPI
     * @param {string} searchQuery - Query to search
     * @returns {Promise<Array>} NewsAPI results
     */
    async function fetchFromNewsAPI(searchQuery) {
        try {
            const url = `${APIs.newsAPI.url}?q=${encodeURIComponent(searchQuery)}&apiKey=${APIs.newsAPI.key}&pageSize=20&sortBy=relevancy`;
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`NewsAPI error: ${response.status}`);
            }
            
            const data = await response.json();
            console.log(`[SUBTASK 2] NewsAPI returned ${data.articles?.length || 0} articles`);
            
            return (data.articles || []).map(article => ({
                title: article.title || 'No title',
                snippet: article.description || article.content?.substring(0, 200) || 'No description',
                url: article.url || '#',
                domain: extractDomain(article.url),
                source: APIs.newsAPI.name,
                publishedAt: article.publishedAt,
                author: article.author
            }));
            
        } catch (error) {
            console.error(`[SUBTASK 2] NewsAPI error:`, error);
            return [];
        }
    }
    
    /**
     * Fetch from SerperAPI (Google Search)
     * @param {string} searchQuery - Query to search
     * @returns {Promise<Array>} SerperAPI results
     */
    async function fetchFromSerperAPI(searchQuery) {
        try {
            const response = await fetch(APIs.serperAPI.url, {
                method: 'POST',
                headers: {
                    'X-API-KEY': APIs.serperAPI.key,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    q: `"${searchQuery}" news fact check`,
                    num: 20
                })
            });
            
            if (!response.ok) {
                throw new Error(`SerperAPI error: ${response.status}`);
            }
            
            const data = await response.json();
            console.log(`[SUBTASK 2] SerperAPI returned ${data.organic?.length || 0} results`);
            
            return (data.organic || []).map(result => ({
                title: result.title || 'No title',
                snippet: result.snippet || 'No snippet',
                url: result.link || '#',
                domain: extractDomain(result.link),
                source: APIs.serperAPI.name,
                publishedAt: result.date || null
            }));
            
        } catch (error) {
            console.error(`[SUBTASK 2] SerperAPI error:`, error);
            return [];
        }
    }
    
    /**
     * Fetch from NewsData.io
     * @param {string} searchQuery - Query to search
     * @returns {Promise<Array>} NewsData.io results
     */
    async function fetchFromNewsDataIO(searchQuery) {
        try {
            const url = `${APIs.newsDataIO.url}?apikey=${APIs.newsDataIO.key}&q=${encodeURIComponent(searchQuery)}&language=en&size=20`;
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`NewsData.io error: ${response.status}`);
            }
            
            const data = await response.json();
            console.log(`[SUBTASK 2] NewsData.io returned ${data.results?.length || 0} articles`);
            
            return (data.results || []).map(article => ({
                title: article.title || 'No title',
                snippet: article.description || article.content?.substring(0, 200) || 'No description',
                url: article.link || '#',
                domain: extractDomain(article.link),
                source: APIs.newsDataIO.name,
                publishedAt: article.pubDate,
                category: article.category
            }));
            
        } catch (error) {
            console.error(`[SUBTASK 2] NewsData.io error:`, error);
            return [];
        }
    }
    
    // Execute all API calls in parallel
    try {
        console.log(`[SUBTASK 2] Executing parallel API calls...`);
        const [newsAPIResults, serperResults, newsDataResults] = await Promise.all([
            fetchFromNewsAPI(query),
            fetchFromSerperAPI(query),
            fetchFromNewsDataIO(query)
        ]);
        
        const allResults = [...newsAPIResults, ...serperResults, ...newsDataResults];
        console.log(`[SUBTASK 2] Total results collected: ${allResults.length}`);
        
        return allResults;
        
    } catch (error) {
        console.error(`[SUBTASK 2] Error in parallel API calls:`, error);
        throw new Error(`Failed to fetch news results: ${error.message}`);
    }
}

/**
 * Extract domain from URL
 * @param {string} url - URL to extract domain from
 * @returns {string} Domain name
 */
function extractDomain(url) {
    if (!url) return 'unknown';
    try {
        return new URL(url).hostname.replace('www.', '');
    } catch {
        return 'unknown';
    }
}

// =============================================================================
// SUBTASK 3: MERGE, DEDUPLICATE, AND RANK WITH DOMAIN TRUST SCORING
// =============================================================================

/**
 * Merge, deduplicate, and rank results using domain trust scoring
 * @param {Array} results - Raw results from all APIs
 * @returns {Array} Processed and ranked results
 */
function mergeAndRankResults(results) {
    console.log(`[SUBTASK 3] Processing ${results.length} raw results`);
    
    // Trusted domains with trust scores (0-1)
    const trustedDomains = {
        // Fact-checking sites (highest trust)
        'snopes.com': 0.95,
        'politifact.com': 0.95,
        'factcheck.org': 0.95,
        'reuters.com': 0.90,
        'apnews.com': 0.90,
        'bbc.com': 0.88,
        'npr.org': 0.87,
        'pbs.org': 0.85,
        
        // Major news outlets
        'cnn.com': 0.75,
        'nytimes.com': 0.80,
        'washingtonpost.com': 0.78,
        'theguardian.com': 0.77,
        'wsj.com': 0.79,
        'usatoday.com': 0.72,
        'abcnews.go.com': 0.74,
        'cbsnews.com': 0.73,
        'nbcnews.com': 0.74,
        
        // International sources
        'aljazeera.com': 0.70,
        'dw.com': 0.75,
        'france24.com': 0.73
    };
    
    /**
     * Calculate trust score for a domain
     * @param {string} domain - Domain to score
     * @param {string} url - Full URL for additional scoring
     * @returns {number} Trust score (0-1)
     */
    function calculateTrustScore(domain, url) {
        let score = trustedDomains[domain] || 0.5; // Default neutral score
        
        // HTTPS bonus
        if (url && url.startsWith('https://')) {
            score += 0.05;
        }
        
        // Fact-check path bonus
        if (url && (url.includes('fact-check') || url.includes('factcheck') || url.includes('verify'))) {
            score += 0.1;
        }
        
        // Cap at 1.0
        return Math.min(score, 1.0);
    }
    
    /**
     * Check if domain is a fact-checker
     * @param {string} domain - Domain to check
     * @returns {boolean} True if fact-checker
     */
    function isFactChecker(domain) {
        const factCheckers = ['snopes.com', 'politifact.com', 'factcheck.org'];
        return factCheckers.includes(domain);
    }
    
    // Step 1: Add trust scores and fact-checker flags
    const scoredResults = results.map(result => ({
        ...result,
        trustScore: calculateTrustScore(result.domain, result.url),
        isFactChecker: isFactChecker(result.domain),
        id: `${result.domain}-${result.title?.substring(0, 50) || 'notitle'}`
    }));
    
    // Step 2: Deduplicate based on similar titles and domains
    const deduplicatedResults = [];
    const seenTitles = new Set();
    
    for (const result of scoredResults) {
        const titleKey = result.title?.toLowerCase().substring(0, 100) || '';
        const uniqueKey = `${result.domain}-${titleKey}`;
        
        if (!seenTitles.has(uniqueKey)) {
            seenTitles.add(uniqueKey);
            deduplicatedResults.push(result);
        }
    }
    
    // Step 3: Rank by trust score (fact-checkers first, then by trust score)
    const rankedResults = deduplicatedResults.sort((a, b) => {
        // Fact-checkers always come first
        if (a.isFactChecker && !b.isFactChecker) return -1;
        if (!a.isFactChecker && b.isFactChecker) return 1;
        
        // Then sort by trust score
        return b.trustScore - a.trustScore;
    });
    
    console.log(`[SUBTASK 3] Processed results: ${results.length} → ${deduplicatedResults.length} → ${rankedResults.length} (ranked)`);
    
    return rankedResults;
}

// =============================================================================
// SUBTASK 4: EXTRACT TOP 10 WITH FACT-CHECKER DETECTION
// =============================================================================

/**
 * Extract top 10 results with enhanced fact-checker detection
 * @param {Array} rankedResults - Ranked results from previous step
 * @returns {Array} Top 10 results with fact-check verdicts
 */
async function extractTop10WithFactChecking(rankedResults) {
    console.log(`[SUBTASK 4] Extracting top 10 from ${rankedResults.length} results`);
    
    const top10 = rankedResults.slice(0, 10);
    
    /**
     * Extract fact-check verdict from snippet/title
     * @param {Object} result - News result object
     * @returns {string|null} Verdict if found
     */
    function extractFactCheckVerdict(result) {
        if (!result.isFactChecker) return null;
        
        const text = `${result.title} ${result.snippet}`.toLowerCase();
        
        // Verdict patterns
        const verdictPatterns = {
            'true': ['true', 'accurate', 'correct', 'verified', 'confirmed'],
            'false': ['false', 'fake', 'misleading', 'debunked', 'incorrect', 'unsubstantiated'],
            'mixed': ['mixed', 'partly true', 'partially correct', 'some truth'],
            'unproven': ['unproven', 'unverified', 'unclear', 'insufficient evidence']
        };
        
        for (const [verdict, keywords] of Object.entries(verdictPatterns)) {
            if (keywords.some(keyword => text.includes(keyword))) {
                return verdict.toUpperCase();
            }
        }
        
        return 'UNKNOWN';
    }
    
    // Process each result
    const processedResults = top10.map((result, index) => {
        const processed = {
            ...result,
            rank: index + 1,
            trustPercentage: Math.round(result.trustScore * 100),
            factCheckVerdict: extractFactCheckVerdict(result)
        };
        
        console.log(`[SUBTASK 4] Result ${index + 1}: ${processed.domain} (${processed.trustPercentage}% trust)${processed.isFactChecker ? ' [FACT-CHECKER]' : ''}`);
        
        return processed;
    });
    
    return processedResults;
}

// =============================================================================
// SUBTASK 5: COMPUTE CREDIBILITY METRICS
// =============================================================================

/**
 * Compute comprehensive credibility metrics
 * @param {Array} top10Results - Top 10 processed results
 * @returns {Object} Credibility metrics object
 */
function computeCredibilityMetrics(top10Results) {
    console.log(`[SUBTASK 5] Computing credibility metrics for ${top10Results.length} results`);
    
    // Calculate average credibility score
    const averageCredibility = top10Results.reduce((sum, result) => sum + result.trustScore, 0) / top10Results.length;
    
    // Count credible sources (trust score >= 0.7)
    const credibleSources = top10Results.filter(result => result.trustScore >= 0.7).length;
    
    // Fact-checker consensus analysis
    const factCheckers = top10Results.filter(result => result.isFactChecker);
    let factCheckerConsensus = 0;
    
    if (factCheckers.length > 0) {
        const trueCount = factCheckers.filter(fc => fc.factCheckVerdict === 'TRUE').length;
        const falseCount = factCheckers.filter(fc => fc.factCheckVerdict === 'FALSE').length;
        const totalVerdicts = trueCount + falseCount;
        
        if (totalVerdicts > 0) {
            factCheckerConsensus = Math.max(trueCount, falseCount) / totalVerdicts;
        }
    }
    
    // Calculate final confidence score
    let confidenceScore = 0;
    
    // Base confidence from average credibility (40% weight)
    confidenceScore += averageCredibility * 0.4;
    
    // Credible sources ratio (30% weight)
    confidenceScore += (credibleSources / top10Results.length) * 0.3;
    
    // Fact-checker consensus (30% weight)
    confidenceScore += factCheckerConsensus * 0.3;
    
    const metrics = {
        averageCredibility: Math.round(averageCredibility * 100),
        credibleSourcesCount: credibleSources,
        factCheckerConsensus: Math.round(factCheckerConsensus * 100),
        finalConfidenceScore: Math.round(confidenceScore * 100),
        totalSources: top10Results.length,
        factCheckersFound: factCheckers.length
    };
    
    console.log(`[SUBTASK 5] Metrics computed:`, metrics);
    
    return metrics;
}

// =============================================================================
// SUBTASK 6: DYNAMIC RENDERING SYSTEM
// =============================================================================

/**
 * Dynamically render top 10 sources as clickable cards
 * @param {Array} top10Results - Top 10 processed results
 * @param {Object} metrics - Credibility metrics
 * @param {string} containerId - ID of container element
 */
function renderVerificationResults(top10Results, metrics, containerId = 'proofValidationArea') {
    console.log(`[SUBTASK 6] Rendering ${top10Results.length} results to container: ${containerId}`);
    
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`[SUBTASK 6] Container ${containerId} not found`);
        return;
    }
    
    // Create metrics summary HTML
    const metricsHTML = `
        <div class="verification-metrics">
            <h3>🎯 Verification Analysis</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">${metrics.averageCredibility}%</div>
                    <div class="metric-label">Average Credibility</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${metrics.credibleSourcesCount}</div>
                    <div class="metric-label">Credible Sources</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${metrics.factCheckerConsensus}%</div>
                    <div class="metric-label">Fact-Checker Consensus</div>
                </div>
                <div class="metric-card confidence-score">
                    <div class="metric-value">${metrics.finalConfidenceScore}%</div>
                    <div class="metric-label">Final Confidence</div>
                </div>
            </div>
        </div>
    `;
    
    // Create source cards HTML
    const sourceCardsHTML = top10Results.map(result => {
        const trustClass = result.trustScore >= 0.8 ? 'high-trust' : result.trustScore >= 0.6 ? 'medium-trust' : 'low-trust';
        const factCheckerBadge = result.isFactChecker ? `<span class="fact-checker-badge">✓ FACT-CHECKER</span>` : '';
        const verdictBadge = result.factCheckVerdict && result.factCheckVerdict !== 'UNKNOWN' ? 
            `<span class="verdict-badge verdict-${result.factCheckVerdict.toLowerCase()}">${result.factCheckVerdict}</span>` : '';
        
        return `
            <div class="source-card ${trustClass}" data-url="${result.url}">
                <div class="source-header">
                    <div class="source-rank">#${result.rank}</div>
                    <div class="source-domain">${result.domain}</div>
                    <div class="source-trust">${result.trustPercentage}%</div>
                </div>
                <div class="source-content">
                    <h4 class="source-title">${result.title}</h4>
                    <p class="source-snippet">${result.snippet}</p>
                </div>
                <div class="source-footer">
                    <div class="source-badges">
                        ${factCheckerBadge}
                        ${verdictBadge}
                    </div>
                    <a href="${result.url}" target="_blank" class="source-link" rel="noopener noreferrer">
                        Read Full Article →
                    </a>
                </div>
            </div>
        `;
    }).join('');
    
    // Combine and render
    container.innerHTML = `
        ${metricsHTML}
        <div class="sources-section">
            <h3>📰 Top Verification Sources</h3>
            <div class="sources-grid">
                ${sourceCardsHTML}
            </div>
        </div>
    `;
    
    // Add click handlers for cards
    container.querySelectorAll('.source-card').forEach(card => {
        card.addEventListener('click', (e) => {
            if (e.target.tagName !== 'A') {
                const url = card.dataset.url;
                window.open(url, '_blank', 'noopener,noreferrer');
            }
        });
    });
    
    console.log(`[SUBTASK 6] Rendering complete`);
}

// =============================================================================
// SUBTASK 7: ERROR HANDLING AND LOADING STATES
// =============================================================================

/**
 * Show loading state in the UI
 * @param {string} message - Loading message to display
 */
function showLoadingState(message = 'Analyzing content...') {
    console.log(`[SUBTASK 7] Showing loading state: ${message}`);
    
    const loadingHTML = `
        <div class="verification-loading">
            <div class="loading-spinner"></div>
            <div class="loading-message">${message}</div>
            <div class="loading-progress">
                <div class="progress-bar"></div>
            </div>
        </div>
    `;
    
    const container = document.getElementById('proofValidationArea');
    if (container) {
        container.innerHTML = loadingHTML;
    }
}

/**
 * Show error state in the UI
 * @param {string} error - Error message to display
 */
function showErrorState(error) {
    console.error(`[SUBTASK 7] Showing error state:`, error);
    
    const errorHTML = `
        <div class="verification-error">
            <div class="error-icon">⚠️</div>
            <div class="error-title">Verification Failed</div>
            <div class="error-message">${error}</div>
            <button class="retry-button" onclick="retryVerification()">Try Again</button>
        </div>
    `;
    
    const container = document.getElementById('proofValidationArea');
    if (container) {
        container.innerHTML = errorHTML;
    }
}

/**
 * Handle and log errors appropriately
 * @param {Error} error - Error object
 * @param {string} context - Context where error occurred
 */
function handleError(error, context) {
    console.error(`[SUBTASK 7] Error in ${context}:`, error);
    
    let userMessage = 'An unexpected error occurred during verification.';
    
    if (error.message.includes('fetch')) {
        userMessage = 'Network error: Unable to connect to verification services.';
    } else if (error.message.includes('API')) {
        userMessage = 'API error: Verification services are temporarily unavailable.';
    } else if (error.message.includes('parse') || error.message.includes('JSON')) {
        userMessage = 'Data error: Unable to process verification results.';
    }
    
    showErrorState(userMessage);
}

// =============================================================================
// SUBTASK 8: COMPREHENSIVE COMMENTS (COMPLETED THROUGHOUT)
// =============================================================================

// All functions have been thoroughly commented as requested

// =============================================================================
// SUBTASK 9: MAIN INTEGRATION AND FINAL OUTPUT
// =============================================================================

/**
 * Main verification function that orchestrates the entire analysis pipeline
 * @param {string} inputType - Type of input ('text', 'url', 'image')
 * @param {string|File} inputValue - The input value to analyze
 */
async function analyzeContent(inputType, inputValue) {
    console.log(`[SUBTASK 9] Starting verification analysis for ${inputType}`);
    
    try {
        // Step 1: Parse and validate input
        showLoadingState('Parsing input...');
        const parsedInput = parseUserInput(inputType, inputValue);
        
        if (!parsedInput.isValid) {
            showErrorState(parsedInput.error);
            return;
        }
        
        // Step 2: Fetch news results from multiple APIs
        showLoadingState('Fetching verification sources...');
        const rawResults = await fetchNewsResults(parsedInput.content);
        
        if (rawResults.length === 0) {
            showErrorState('No verification sources found for this content.');
            return;
        }
        
        // Step 3: Merge, deduplicate, and rank results
        showLoadingState('Processing and ranking sources...');
        const rankedResults = mergeAndRankResults(rawResults);
        
        // Step 4: Extract top 10 with fact-checking
        showLoadingState('Analyzing fact-checker verdicts...');
        const top10Results = await extractTop10WithFactChecking(rankedResults);
        
        // Step 5: Compute credibility metrics
        showLoadingState('Computing credibility metrics...');
        const metrics = computeCredibilityMetrics(top10Results);
        
        // Step 6: Set up global proofsArray for Content Result verification
        showLoadingState('Preparing verification data...');
        window.proofsArray = top10Results.map(result => ({
            url: result.url,
            title: result.title,
            snippet: result.snippet,
            domain: result.domain,
            credibility_score: result.trustScore,
            fact_check_verdict: result.factCheckVerdict || null
        }));
        
        console.log(`[GLOBAL] proofsArray set with ${window.proofsArray.length} proofs for Content Result verification`);
        
        // Step 7: Render results
        showLoadingState('Rendering verification results...');
        renderVerificationResults(top10Results, metrics);
        
        console.log(`[SUBTASK 9] Verification analysis completed successfully`);
        console.log(`[INTEGRATION] Content Result verification data ready`);
        
    } catch (error) {
        handleError(error, 'analyzeContent');
    }
}

/**
 * Retry verification function
 */
function retryVerification() {
    console.log('[RETRY] Retrying verification...');
    const detectBtn = document.getElementById('detectBtn');
    if (detectBtn) {
        detectBtn.click();
    }
}

/**
 * Initialize the verification system
 * Sets up event listeners and UI components
 */
function initializeVerificationSystem() {
    console.log('[INIT] Initializing fake news verification system...');
    
    // Find the analyze button
    const analyzeButton = document.getElementById('detectBtn');
    if (!analyzeButton) {
        console.error('[INIT] Analyze button not found');
        return;
    }
    
    // Add click handler to analyze button
    analyzeButton.addEventListener('click', async (e) => {
        e.preventDefault();
        
        // Determine input type and value
        const activeTab = document.querySelector('.tab-btn.active')?.dataset.tab || 'text';
        let inputValue = '';
        
        switch (activeTab) {
            case 'text':
                inputValue = document.getElementById('newsText')?.value || '';
                break;
            case 'url':
                inputValue = document.getElementById('newsUrl')?.value || '';
                break;
            case 'image':
                const imageFile = document.getElementById('imageFile')?.files[0];
                if (imageFile) {
                    inputValue = imageFile;
                }
                break;
        }
        
        if (!inputValue) {
            showErrorState('Please enter content to analyze.');
            return;
        }
        
        // Start analysis
        await analyzeContent(activeTab, inputValue);
    });
    
    // Create proof validation area if it doesn't exist
    let proofArea = document.getElementById('proofValidationArea');
    if (!proofArea) {
        proofArea = document.createElement('div');
        proofArea.id = 'proofValidationArea';
        proofArea.className = 'proof-validation-area';
        
        // Find a good place to insert it
        const resultsSection = document.getElementById('resultsSection');
        if (resultsSection) {
            resultsSection.appendChild(proofArea);
        } else {
            document.body.appendChild(proofArea);
        }
    }
    
    console.log('[INIT] Verification system initialized successfully');
}

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeVerificationSystem);
} else {
    initializeVerificationSystem();
}

// =============================================================================
// FINAL OUTPUT SUMMARY
// =============================================================================

/**
 * FINAL OUTPUT SUMMARY:
 * 
 * When the user clicks the '🔍 Analyze Content' button, the following happens:
 * 
 * 1. INPUT PARSING: The system validates the user's input (text/URL/image)
 * 2. API INTEGRATION: Parallel calls to NewsAPI, SerperAPI, and NewsData.io
 * 3. DATA PROCESSING: Results are merged, deduplicated, and ranked by trust score
 * 4. FACT-CHECKING: Top 10 sources are identified with fact-checker detection
 * 5. METRICS CALCULATION: Credibility scores and confidence levels are computed
 * 6. DYNAMIC RENDERING: Results appear as interactive cards in the dashboard
 * 
 * USER SEES:
 * - Verification metrics (average credibility, credible sources, consensus, confidence)
 * - Top 10 clickable source cards with trust percentages
 * - Fact-checker badges and verdicts where available
 * - Real-time loading states and error handling
 * - Professional dashboard interface with all results
 * 
 * The system provides transparent, real-time fake news verification using
 * multiple trusted sources and sophisticated ranking algorithms.
 */

console.log('🔍 Fake News Verification System Loaded Successfully');
console.log('📊 Ready for real-time content analysis');
console.log('🎯 All 9 subtasks implemented and integrated');
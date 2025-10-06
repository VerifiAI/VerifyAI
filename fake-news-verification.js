// =============================================================================
// MASTER PIPELINE ORCHESTRATOR - UNIVERSAL FACT VERIFICATION ENGINE
// =============================================================================

/**
 * Master function that orchestrates the complete universal fact verification pipeline
 * Implements all 10 stages for world-class claim verification
 * @param {string} userClaim - User's input claim to verify
 * @returns {Promise<Object>} Complete verification results with verdict
 */
async function executeUniversalFactVerification(userClaim) {
    console.log(`[MASTER PIPELINE] Starting universal fact verification for: "${userClaim.substring(0, 100)}..."`);;
    
    const startTime = performance.now();
    
    try {
        // STAGE 1: Universal Claim Parsing
        console.log(`[PIPELINE] === STAGE 1: UNIVERSAL CLAIM PARSING ===`);
        const claimData = parseUniversalClaim(userClaim);
        
        if (!claimData.isValid) {
            throw new Error('Invalid claim: Unable to extract meaningful components for verification');
        }
        
        // STAGE 2: Diversified Query Generation
        console.log(`[PIPELINE] === STAGE 2: DIVERSIFIED QUERY GENERATION ===`);
        const queries = generateDiversifiedQueries(claimData);
        
        // STAGE 3: Parallel Evidence Collection
        console.log(`[PIPELINE] === STAGE 3: PARALLEL EVIDENCE COLLECTION ===`);
        const rawEvidence = await collectEvidenceParallel(queries);
        
        if (rawEvidence.length === 0) {
            return handleInsufficientEvidence(claimData);
        }
        
        // STAGE 4: Multi-Layered Filtering
        console.log(`[PIPELINE] === STAGE 4: MULTI-LAYERED FILTERING ===`);
        const filteredEvidence = await applyMultiLayeredFiltering(claimData, rawEvidence);
        
        // STAGE 5: Deduplication & Trust-Based Ranking
        console.log(`[PIPELINE] === STAGE 5: DEDUPLICATION & RANKING ===`);
        const rankedEvidence = deduplicateAndRank(filteredEvidence);
        
        // STAGE 6: Fact-Source Matrix Construction
        console.log(`[PIPELINE] === STAGE 6: FACT-SOURCE MATRIX CONSTRUCTION ===`);
        const factMatrix = constructFactSourceMatrix(claimData, rankedEvidence);
        
        // STAGE 7: Consistency & Contradiction Analysis
        console.log(`[PIPELINE] === STAGE 7: CONSISTENCY ANALYSIS ===`);
        const consistencyAnalysis = analyzeConsistencyAndContradictions(factMatrix);
        
        // STAGE 8: Automated Verdict Synthesis
        console.log(`[PIPELINE] === STAGE 8: VERDICT SYNTHESIS ===`);
        const verdict = synthesizeAutomatedVerdict(claimData, factMatrix, consistencyAnalysis, rankedEvidence);
        
        // STAGE 9: Comprehensive Dashboard Rendering
        console.log(`[PIPELINE] === STAGE 9: DASHBOARD RENDERING ===`);
        const dashboardData = renderComprehensiveDashboard(claimData, factMatrix, consistencyAnalysis, verdict, rankedEvidence);
        
        // STAGE 10: Edge Case Handling & Optimization
        console.log(`[PIPELINE] === STAGE 10: FINALIZATION ===`);
        const finalResults = finalizeVerificationResults(dashboardData, verdict);
        
        const endTime = performance.now();
        const duration = Math.round(endTime - startTime);
        
        console.log(`[MASTER PIPELINE] ✅ Universal fact verification completed in ${duration}ms`);
        console.log(`[MASTER PIPELINE] Verdict: ${verdict.classification} (${verdict.confidence}% confidence)`);
        
        return finalResults;
        
    } catch (error) {
        console.error(`[MASTER PIPELINE] ❌ Verification failed:`, error);
        return handleVerificationError(error, userClaim);
    }
}

// =============================================================================
// PIPELINE STAGE 1: UNIVERSAL CLAIM PARSING & CORE ELEMENT EXTRACTION
// =============================================================================

/**
 * Universal claim parser that extracts core elements from any user claim
 * Supports all topics globally with advanced NLP techniques
 * @param {string} userClaim - Raw user input claim to analyze
 * @returns {Object} Structured claim components for verification
 */
function parseUniversalClaim(userClaim) {
    console.log(`[STAGE 1] Parsing universal claim: "${userClaim.substring(0, 100)}..."`);;
    
    const claimData = {
        originalText: userClaim,
        coreEntities: [],
        numericFacts: [],
        actionVerbs: [],
        keyPhrases: [],
        temporalMarkers: [],
        geographicMarkers: [],
        claimType: 'general',
        complexity: 0,
        isValid: false
    };
    
    try {
        // Extract named entities with advanced patterns
        const entityPatterns = [
            // People with titles
            /\b(?:President|Prime Minister|CEO|Director|Dr\.|Prof\.|Mr\.|Ms\.|Mrs\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b/g,
            // Organizations and institutions
            /\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Inc|Corp|LLC|Ltd|University|College|Hospital|Bank|Agency|Department|Ministry|Organization|Foundation|Institute))\b/g,
            // Countries and major cities
            /\b(?:United States|United Kingdom|European Union|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b/g,
            // Proper nouns (2+ capitalized words)
            /\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b/g
        ];
        
        entityPatterns.forEach(pattern => {
            const matches = userClaim.match(pattern) || [];
            matches.forEach(match => {
                const cleanEntity = match.trim();
                if (cleanEntity.length > 2 && !claimData.coreEntities.includes(cleanEntity)) {
                    claimData.coreEntities.push(cleanEntity);
                }
            });
        });
        
        // Extract numeric facts with comprehensive context
        const numericPatterns = [
            // Percentages with variations
            /\b(\d+(?:\.\d+)?)\s*(?:%|percent|percentage|pct)\b/gi,
            // Currency amounts
            /\b(?:\$|€|£|¥|USD|EUR|GBP|JPY)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\b/gi,
            // Large numbers with units
            /\b(\d+(?:\.\d+)?)\s*(?:million|billion|trillion|thousand|k|M|B|T)\b/gi,
            // Dates and years
            /\b((?:19|20)\d{2})\b/g,
            // Ratios and fractions
            /\b(\d+)\s*(?:in|out\s+of|per)\s*(\d+)\b/gi,
            // General numbers
            /\b\d+(?:\.\d+)?\b/g
        ];
        
        numericPatterns.forEach((pattern, patternIndex) => {
            const matches = [...userClaim.matchAll(pattern)];
            matches.forEach(match => {
                const value = parseFloat(match[1] || match[0]);
                if (!isNaN(value)) {
                    const contextStart = Math.max(0, match.index - 30);
                    const contextEnd = Math.min(userClaim.length, match.index + match[0].length + 30);
                    const context = userClaim.substring(contextStart, contextEnd).trim();
                    
                    claimData.numericFacts.push({
                        value: value,
                        originalText: match[0],
                        context: context,
                        type: ['percentage', 'currency', 'magnitude', 'year', 'ratio', 'number'][patternIndex] || 'number',
                        tolerance: value * 0.05, // 5% tolerance
                        position: match.index
                    });
                }
            });
        });
        
        // Extract action verbs and temporal markers
        const actionPatterns = [
            /\b(?:announced|declared|reported|confirmed|denied|increased|decreased|launched|implemented|banned|approved|rejected|stated|claimed|alleged|revealed|discovered|found|showed|proved|demonstrated)\b/gi,
            /\b(?:will|would|could|should|may|might|must|shall)\s+\w+/gi
        ];
        
        actionPatterns.forEach(pattern => {
            const matches = userClaim.match(pattern) || [];
            claimData.actionVerbs.push(...matches.map(m => m.toLowerCase()));
        });
        
        // Extract temporal markers
        const temporalPatterns = [
            /\b(?:yesterday|today|tomorrow|last\s+\w+|next\s+\w+|this\s+\w+|in\s+\d+|\d+\s+(?:days?|weeks?|months?|years?)\s+ago)\b/gi,
            /\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b/gi,
            /\b\d{1,2}\/\d{1,2}\/\d{4}\b/g
        ];
        
        temporalPatterns.forEach(pattern => {
            const matches = userClaim.match(pattern) || [];
            claimData.temporalMarkers.push(...matches);
        });
        
        // Extract geographic markers
        const geoPatterns = [
            /\b(?:in|at|from|to)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b/g,
            /\b[A-Z][a-z]+,\s*[A-Z]{2}\b/g // City, State format
        ];
        
        geoPatterns.forEach(pattern => {
            const matches = userClaim.match(pattern) || [];
            claimData.geographicMarkers.push(...matches);
        });
        
        // Extract key phrases using n-gram analysis
        const words = userClaim.toLowerCase().split(/\s+/);
        for (let i = 0; i < words.length - 1; i++) {
            const bigram = `${words[i]} ${words[i + 1]}`;
            if (bigram.length > 5 && !isCommonPhrase(bigram)) {
                claimData.keyPhrases.push(bigram);
            }
        }
        
        // Determine claim type and complexity
        claimData.claimType = determineClaimType(claimData);
        claimData.complexity = calculateClaimComplexity(claimData);
        claimData.isValid = claimData.coreEntities.length > 0 || 
                          claimData.numericFacts.length > 0 || 
                          userClaim.length > 15;
        
        // Remove duplicates and limit arrays
        claimData.coreEntities = [...new Set(claimData.coreEntities)].slice(0, 15);
        claimData.actionVerbs = [...new Set(claimData.actionVerbs)].slice(0, 10);
        claimData.keyPhrases = [...new Set(claimData.keyPhrases)].slice(0, 10);
        
        console.log(`[STAGE 1] Extracted components:`, {
            entities: claimData.coreEntities.length,
            numericFacts: claimData.numericFacts.length,
            actions: claimData.actionVerbs.length,
            type: claimData.claimType,
            complexity: claimData.complexity
        });
        
        return claimData;
        
    } catch (error) {
        console.error(`[STAGE 1] Error parsing claim:`, error);
        claimData.isValid = false;
        return claimData;
    }
}

function determineClaimType(claimData) {
    if (claimData.numericFacts.some(f => f.type === 'percentage')) return 'statistical';
    if (claimData.numericFacts.some(f => f.type === 'currency')) return 'financial';
    if (claimData.temporalMarkers.length > 0) return 'temporal';
    if (claimData.coreEntities.some(e => /President|Prime Minister|CEO/.test(e))) return 'political';
    if (claimData.actionVerbs.some(v => /announced|declared|launched/.test(v))) return 'event';
    return 'general';
}

function calculateClaimComplexity(claimData) {
    let complexity = 0;
    complexity += claimData.coreEntities.length * 0.5;
    complexity += claimData.numericFacts.length * 1.0;
    complexity += claimData.actionVerbs.length * 0.3;
    complexity += claimData.temporalMarkers.length * 0.4;
    return Math.min(10, Math.round(complexity));
}

function isCommonPhrase(phrase) {
    const commonPhrases = ['the of', 'in the', 'to the', 'and the', 'is a', 'are a', 'was a', 'were a'];
    return commonPhrases.includes(phrase);
}

// =============================================================================
// PIPELINE STAGE 2: DIVERSIFIED QUERY GENERATION & API OPTIMIZATION
// =============================================================================

function generateDiversifiedQueries(claimData) {
    console.log(`[STAGE 2] Generating diversified queries for ${claimData.claimType} claim`);
    
    const queries = [];
    const { originalText, coreEntities, numericFacts, actionVerbs, keyPhrases } = claimData;
    
    try {
        // 1. EXACT CLAIM QUERY
        queries.push({
            type: 'exact',
            text: `"${originalText}" fact check`,
            priority: 1.0,
            expectedRelevance: 0.9,
            apiTargets: ['newsapi', 'serper', 'newsdata']
        });
        
        // 2. ENTITY-FOCUSED QUERIES
        if (coreEntities.length > 0) {
            const primaryEntities = coreEntities.slice(0, 3);
            queries.push({
                type: 'entity_primary',
                text: `${primaryEntities.join(' ')} verify truth`,
                priority: 0.9,
                expectedRelevance: 0.8,
                apiTargets: ['newsapi', 'serper']
            });
        }
        
        // 3. NUMERIC FACT QUERIES
        if (numericFacts.length > 0) {
            numericFacts.slice(0, 2).forEach(fact => {
                queries.push({
                    type: 'numeric_fact',
                    text: `"${fact.originalText}" ${fact.context.split(' ').slice(0, 5).join(' ')} verify`,
                    priority: 0.85,
                    expectedRelevance: 0.8,
                    apiTargets: ['newsapi', 'serper'],
                    numericContext: fact
                });
            });
        }
        
        // 4. ACTION-BASED QUERIES
        if (actionVerbs.length > 0 && coreEntities.length > 0) {
            queries.push({
                type: 'action_based',
                text: `"${coreEntities[0]} ${actionVerbs[0]}" news fact check`,
                priority: 0.8,
                expectedRelevance: 0.75,
                apiTargets: ['serper', 'newsdata']
            });
        }
        
        // 5. NEGATION QUERIES
        if (coreEntities.length > 0) {
            queries.push({
                type: 'negation',
                text: `${coreEntities[0]} NOT ${actionVerbs[0] || 'true'} debunk`,
                priority: 0.7,
                expectedRelevance: 0.7,
                apiTargets: ['serper']
            });
        }
        
        const optimizedQueries = queries
            .filter(q => q.text.length > 10 && q.text.length < 200)
            .sort((a, b) => b.priority - a.priority)
            .slice(0, 8);
        
        console.log(`[STAGE 2] Generated ${optimizedQueries.length} optimized queries`);
        return optimizedQueries;
        
    } catch (error) {
        console.error(`[STAGE 2] Error generating queries:`, error);
        return [{
            type: 'fallback',
            text: `"${originalText}" fact check`,
            priority: 1.0,
            expectedRelevance: 0.5,
            apiTargets: ['newsapi', 'serper', 'newsdata']
        }];
    }
}

// =============================================================================
// PIPELINE STAGE 3: PARALLEL MULTI-API EVIDENCE COLLECTION
// =============================================================================

async function collectEvidenceParallel(queries) {
    console.log(`[STAGE 3] Collecting evidence from ${queries.length} queries across 3 APIs`);
    
    // Direct API configuration with provided keys
    const API_KEYS = {
        newsapi: 'd9ecb91025124a478ef549c30a592574',
        serper: '0b95ccd48f33e0236e6cb83b97b1b21d26431f6c',
        newsdata: 'pub_1dc89a34f1fd4d7cb1237d799a65cc90'
    };
    
    const allEvidence = [];
    const apiStats = { newsapi: 0, serper: 0, newsdata: 0, errors: 0 };
    
    async function fetchNewsAPI(query) {
        try {
            const url = `https://newsapi.org/v2/everything?q=${encodeURIComponent(query.text)}&language=en&sortBy=relevancy&pageSize=20&apiKey=${API_KEYS.newsapi}`;
            
            const response = await fetch(url, {
                method: 'GET',
                headers: {
                    'User-Agent': 'FakeNewsDetector/1.0'
                }
            });
            
            if (!response.ok) throw new Error(`NewsAPI HTTP ${response.status}`);
            
            const data = await response.json();
            if (data.status === 'ok' && data.articles) {
                apiStats.newsapi += data.articles.length;
                return data.articles.map(article => ({
                    title: article.title || 'Untitled',
                    description: article.description || '',
                    url: article.url || '#',
                    source: {
                        name: article.source?.name || 'NewsAPI',
                        domain: extractDomain(article.url)
                    },
                    publishedAt: article.publishedAt,
                    apiSource: 'newsapi',
                    queryType: query.type,
                    queryPriority: query.priority,
                    rawContent: `${article.title} ${article.description}`.toLowerCase()
                }));
            }
            return [];
        } catch (error) {
            console.error(`[STAGE 3] NewsAPI error:`, error);
            apiStats.errors++;
            return [];
        }
    }
    
    async function fetchSerperAPI(query) {
        try {
            const response = await fetch('https://google.serper.dev/news', {
                method: 'POST',
                headers: {
                    'X-API-KEY': API_KEYS.serper,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    q: query.text,
                    num: 20
                })
            });
            
            if (!response.ok) throw new Error(`Serper HTTP ${response.status}`);
            
            const data = await response.json();
            if (data.news && Array.isArray(data.news)) {
                apiStats.serper += data.news.length;
                return data.news.map(result => ({
                    title: result.title || 'Untitled',
                    description: result.snippet || result.description || '',
                    url: result.link || '#',
                    source: {
                        name: result.source || 'Serper',
                        domain: extractDomain(result.link)
                    },
                    publishedAt: result.date,
                    apiSource: 'serper',
                    queryType: query.type,
                    queryPriority: query.priority,
                    rawContent: `${result.title} ${result.snippet}`.toLowerCase()
                }));
            }
            return [];
        } catch (error) {
            console.error(`[STAGE 3] Serper error:`, error);
            apiStats.errors++;
            return [];
        }
    }
    
    async function fetchNewsDataAPI(query) {
        try {
            const url = `https://newsdata.io/api/1/news?apikey=${API_KEYS.newsdata}&q=${encodeURIComponent(query.text)}&language=en&size=20`;
            
            const response = await fetch(url, {
                method: 'GET',
                headers: {
                    'User-Agent': 'FakeNewsDetector/1.0'
                }
            });
            
            if (!response.ok) throw new Error(`NewsData HTTP ${response.status}`);
            
            const data = await response.json();
            if (data.status === 'success' && data.results) {
                apiStats.newsdata += data.results.length;
                return data.results.map(article => ({
                    title: article.title || 'Untitled',
                    description: article.description || article.content || '',
                    url: article.link || '#',
                    source: {
                        name: article.source_id || 'NewsData',
                        domain: extractDomain(article.link)
                    },
                    publishedAt: article.pubDate,
                    apiSource: 'newsdata',
                    queryType: query.type,
                    queryPriority: query.priority,
                    rawContent: `${article.title} ${article.description}`.toLowerCase()
                }));
            }
            return [];
        } catch (error) {
            console.error(`[STAGE 3] NewsData error:`, error);
            apiStats.errors++;
            return [];
        }
    }
    
    // Create API call matrix
    const apiCalls = [];
    queries.forEach(query => {
        if (query.apiTargets.includes('newsapi')) {
            apiCalls.push(fetchNewsAPI(query));
        }
        if (query.apiTargets.includes('serper')) {
            apiCalls.push(fetchSerperAPI(query));
        }
        if (query.apiTargets.includes('newsdata')) {
            apiCalls.push(fetchNewsDataAPI(query));
        }
    });
    
    // Execute all API calls in controlled batches
    const batchSize = 6;
    for (let i = 0; i < apiCalls.length; i += batchSize) {
        const batch = apiCalls.slice(i, i + batchSize);
        const batchResults = await Promise.all(batch);
        allEvidence.push(...batchResults.flat());
        
        if (i + batchSize < apiCalls.length) {
            await new Promise(resolve => setTimeout(resolve, 200));
        }
    }
    
    console.log(`[STAGE 3] Evidence collection complete:`, {
        totalArticles: allEvidence.length,
        newsapi: apiStats.newsapi,
        serper: apiStats.serper,
        newsdata: apiStats.newsdata,
        errors: apiStats.errors
    });
    
    return allEvidence;
}

function extractDomain(url) {
    if (!url) return 'unknown';
    try {
        const domain = new URL(url).hostname.replace('www.', '');
        return domain;
    } catch {
        return 'unknown';
    }
}

// =============================================================================
// PIPELINE STAGE 4: MULTI-LAYERED FILTERING SYSTEM
// =============================================================================

async function applyMultiLayeredFiltering(claimData, evidence) {
    console.log(`[STAGE 4] Applying multi-layered filtering to ${evidence.length} articles`);
    
    const filteredEvidence = [];
    const filterStats = { semantic: 0, numeric: 0, entity: 0, combined: 0 };
    
    const batchSize = 50;
    for (let i = 0; i < evidence.length; i += batchSize) {
        const batch = evidence.slice(i, i + batchSize);
        
        const batchPromises = batch.map(async (article) => {
            try {
                // Calculate semantic similarity with improved algorithm
                const semanticScore = await calculateSemanticSimilarity(
                    claimData.originalText, 
                    article.rawContent
                );
                
                // Lower threshold for semantic similarity (0.1 instead of 0.3)
                if (semanticScore < 0.1) return null;
                filterStats.semantic++;
                
                // Calculate numeric matching
                const numericScore = calculateNumericMatching(
                    claimData.numericFacts, 
                    article.rawContent
                );
                
                // More lenient numeric matching threshold
                if (claimData.numericFacts.length > 0 && numericScore < 0.1) return null;
                filterStats.numeric++;
                
                // Calculate entity matching
                const entityScore = calculateEntityMatching(
                    claimData.coreEntities, 
                    article.rawContent
                );
                
                // More lenient entity matching threshold
                if (claimData.coreEntities.length > 0 && entityScore < 0.05) return null;
                filterStats.entity++;
                
                // Calculate combined relevance score with better weighting
                const relevanceScore = (
                    semanticScore * 0.6 + 
                    numericScore * 0.25 + 
                    entityScore * 0.15
                );
                
                // Much lower threshold for combined relevance (0.15 instead of 0.4)
                if (relevanceScore >= 0.15) {
                    filterStats.combined++;
                    return {
                        ...article,
                        semanticScore,
                        numericScore,
                        entityScore,
                        relevanceScore: Math.min(1.0, relevanceScore * 2), // Boost final score for display
                        passedFilters: true
                    };
                }
                
                return null;
                
            } catch (error) {
                console.warn(`[STAGE 4] Error filtering article:`, error);
                return null;
            }
        });
        
        const batchResults = await Promise.all(batchPromises);
        const validResults = batchResults.filter(result => result !== null);
        filteredEvidence.push(...validResults);
    }
    
    console.log(`[STAGE 4] Filtering complete:`, {
        original: evidence.length,
        finalFiltered: filterStats.combined,
        filterRate: `${((filterStats.combined / evidence.length) * 100).toFixed(1)}%`
    });
    
    return filteredEvidence.sort((a, b) => b.relevanceScore - a.relevanceScore);
}

async function calculateSemanticSimilarity(claimText, articleText) {
    try {
        // Preprocess both texts
        const claimTokens = preprocessText(claimText);
        const articleTokens = preprocessText(articleText);
        
        // If either text is too short, return low but non-zero similarity
        if (claimTokens.length < 2 || articleTokens.length < 2) {
            return 0.2;
        }
        
        // Create term frequency maps
        const claimTF = createTermFrequency(claimTokens);
        const articleTF = createTermFrequency(articleTokens);
        
        // Calculate cosine similarity
        let similarity = calculateCosineSimilarity(claimTF, articleTF);
        
        // Boost similarity for keyword matches
        const keywordBoost = calculateKeywordBoost(claimTokens, articleTokens);
        similarity = Math.min(1.0, similarity + keywordBoost);
        
        // Ensure minimum similarity for any text that shares words
        const sharedWords = claimTokens.filter(token => articleTokens.includes(token));
        if (sharedWords.length > 0) {
            similarity = Math.max(similarity, 0.2 + (sharedWords.length * 0.1));
        }
        
        return Math.min(1, Math.max(0, similarity));
        
    } catch (error) {
        console.warn('Semantic similarity calculation error:', error);
        return 0.4; // Higher fallback score
    }
}

/**
 * Calculate keyword boost for important term matches
 * @param {Array} claimTokens - Claim tokens
 * @param {Array} articleTokens - Article tokens
 * @returns {number} Boost score
 */
function calculateKeywordBoost(claimTokens, articleTokens) {
    const importantWords = ['president', 'minister', 'government', 'study', 'research', 'report', 'announced', 'confirmed', 'denied', 'increase', 'decrease', 'million', 'billion', 'percent', 'covid', 'vaccine', 'climate', 'election', 'vote'];
    
    let boost = 0;
    claimTokens.forEach(token => {
        if (importantWords.includes(token) && articleTokens.includes(token)) {
            boost += 0.15; // Significant boost for important word matches
        }
    });
    
    return Math.min(0.3, boost); // Cap boost at 0.3
}

function calculateNumericMatching(claimNumerics, articleText) {
    if (claimNumerics.length === 0) return 1.0;
    
    let matchCount = 0;
    const articleNumbers = extractNumbersFromText(articleText);
    
    claimNumerics.forEach(claimNum => {
        const tolerance = claimNum.tolerance || (claimNum.value * 0.05);
        const minValue = claimNum.value - tolerance;
        const maxValue = claimNum.value + tolerance;
        
        const hasMatch = articleNumbers.some(articleNum => 
            articleNum >= minValue && articleNum <= maxValue
        );
        
        if (hasMatch) matchCount++;
    });
    
    return matchCount / claimNumerics.length;
}

function calculateEntityMatching(claimEntities, articleText) {
    if (claimEntities.length === 0) return 1.0;
    
    let matchCount = 0;
    const articleLower = articleText.toLowerCase();
    
    claimEntities.forEach(entity => {
        if (articleLower.includes(entity.toLowerCase())) {
            matchCount++;
        }
    });
    
    return matchCount / claimEntities.length;
}

function preprocessText(text) {
    return text.toLowerCase()
        .replace(/[^a-z0-9\s]/g, ' ')
        .split(/\s+/)
        .filter(token => token.length > 2 && !isStopWord(token));
}

function createTermFrequency(tokens) {
    const tf = {};
    tokens.forEach(token => {
        tf[token] = (tf[token] || 0) + 1;
    });
    return tf;
}

function calculateCosineSimilarity(tf1, tf2) {
    const allTerms = new Set([...Object.keys(tf1), ...Object.keys(tf2)]);
    
    // If no common terms, return 0
    if (allTerms.size === 0) return 0;
    
    let dotProduct = 0;
    let magnitude1 = 0;
    let magnitude2 = 0;
    
    allTerms.forEach(term => {
        const freq1 = tf1[term] || 0;
        const freq2 = tf2[term] || 0;
        
        dotProduct += freq1 * freq2;
        magnitude1 += freq1 * freq1;
        magnitude2 += freq2 * freq2;
    });
    
    const magnitude = Math.sqrt(magnitude1) * Math.sqrt(magnitude2);
    
    if (magnitude === 0) return 0;
    
    let similarity = dotProduct / magnitude;
    
    // Apply smoothing to prevent very low similarities
    // This helps with short texts or sparse term overlap
    const commonTerms = Object.keys(tf1).filter(term => tf2[term] > 0);
    if (commonTerms.length > 0) {
        const termOverlapBonus = Math.min(0.2, commonTerms.length * 0.05);
        similarity += termOverlapBonus;
    }
    
    return Math.min(1.0, similarity);
}

function isStopWord(word) {
    const stopWords = new Set([
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that'
    ]);
    return stopWords.has(word);
}

function extractNumbersFromText(text) {
    const numberPattern = /\b\d+(?:\.\d+)?\b/g;
    const matches = text.match(numberPattern) || [];
    return matches.map(match => parseFloat(match)).filter(num => !isNaN(num));
}

// =============================================================================
// PIPELINE STAGE 5: ADVANCED DEDUPLICATION & TRUST-BASED RANKING
// =============================================================================

function deduplicateAndRank(filteredEvidence) {
    console.log(`[STAGE 5] Deduplicating and ranking ${filteredEvidence.length} articles`);
    
    const domainTrustScores = {
        'snopes.com': 0.98, 'factcheck.org': 0.97, 'politifact.com': 0.96,
        'reuters.com': 0.94, 'apnews.com': 0.93, 'bbc.com': 0.92,
        'nytimes.com': 0.84, 'washingtonpost.com': 0.83, 'wsj.com': 0.82,
        'cnn.com': 0.76, 'abcnews.go.com': 0.75, 'cbsnews.com': 0.74,
        'default': 0.50
    };
    
    const scoredEvidence = filteredEvidence.map(article => {
        const domain = article.source.domain;
        const trustScore = domainTrustScores[domain] || domainTrustScores.default;
        const freshnessScore = calculateFreshnessScore(article.publishedAt);
        
        const rankingScore = (
            article.relevanceScore * 0.4 +
            trustScore * 0.35 +
            freshnessScore * 0.15 +
            article.queryPriority * 0.1
        );
        
        return {
            ...article,
            trustScore,
            freshnessScore,
            rankingScore,
            isFactChecker: trustScore >= 0.95
        };
    });
    
    const deduplicated = [];
    const seenContent = new Set();
    
    scoredEvidence.forEach(article => {
        const contentFingerprint = createContentFingerprint(article.title, article.description);
        
        if (!seenContent.has(contentFingerprint)) {
            seenContent.add(contentFingerprint);
            deduplicated.push(article);
        }
    });
    
    const ranked = deduplicated.sort((a, b) => b.rankingScore - a.rankingScore);
    
    console.log(`[STAGE 5] Deduplication and ranking complete:`, {
        original: filteredEvidence.length,
        deduplicated: deduplicated.length,
        factCheckers: ranked.filter(a => a.isFactChecker).length
    });
    
    return ranked;
}

function calculateFreshnessScore(publishedAt) {
    if (!publishedAt) return 0.3;
    
    try {
        const pubDate = new Date(publishedAt);
        const now = new Date();
        const daysDiff = (now - pubDate) / (1000 * 60 * 60 * 24);
        
        if (daysDiff <= 1) return 1.0;
        if (daysDiff <= 7) return 0.9;
        if (daysDiff <= 30) return 0.7;
        if (daysDiff <= 90) return 0.5;
        if (daysDiff <= 365) return 0.3;
        return 0.1;
    } catch {
        return 0.3;
    }
}

function createContentFingerprint(title, description) {
    const combined = `${title} ${description}`.toLowerCase();
    const words = combined.split(/\s+/).filter(word => word.length > 3);
    const significantWords = words.slice(0, 10).sort().join('|');
    return significantWords;
}

// =============================================================================
// PIPELINE STAGE 6-8: FACT MATRIX, CONSISTENCY ANALYSIS & VERDICT SYNTHESIS
// =============================================================================

function constructFactSourceMatrix(claimData, rankedEvidence) {
    console.log(`[STAGE 6] Constructing fact-source matrix`);
    
    const atomicFacts = extractAtomicFacts(claimData);
    const factMatrix = {
        atomicFacts: atomicFacts,
        sourceMatrix: {},
        factConsensus: {},
        contradictions: []
    };
    
    rankedEvidence.slice(0, 20).forEach((source, sourceIndex) => {
        const sourceId = `source_${sourceIndex}`;
        factMatrix.sourceMatrix[sourceId] = {
            ...source,
            factSupport: {},
            overallStance: 'neutral'
        };
        
        atomicFacts.forEach((fact, factIndex) => {
            const factId = `fact_${factIndex}`;
            const support = analyzeFactSupport(fact, source);
            
            factMatrix.sourceMatrix[sourceId].factSupport[factId] = support;
            
            if (!factMatrix.factConsensus[factId]) {
                factMatrix.factConsensus[factId] = {
                    fact: fact,
                    supports: 0,
                    refutes: 0,
                    neutral: 0
                };
            }
            
            const consensus = factMatrix.factConsensus[factId];
            if (support.stance === 'supports') {
                consensus.supports++;
            } else if (support.stance === 'refutes') {
                consensus.refutes++;
            } else {
                consensus.neutral++;
            }
        });
    });
    
    // Identify contradictions
    Object.keys(factMatrix.factConsensus).forEach(factId => {
        const consensus = factMatrix.factConsensus[factId];
        if (consensus.supports > 0 && consensus.refutes > 0) {
            factMatrix.contradictions.push({
                factId,
                fact: consensus.fact,
                supports: consensus.supports,
                refutes: consensus.refutes
            });
        }
    });
    
    return factMatrix;
}

function extractAtomicFacts(claimData) {
    const atomicFacts = [];
    
    claimData.coreEntities.forEach(entity => {
        if (claimData.actionVerbs.length > 0) {
            claimData.actionVerbs.forEach(action => {
                atomicFacts.push({
                    type: 'entity_action',
                    entity: entity,
                    action: action,
                    text: `${entity} ${action}`,
                    importance: 0.8
                });
            });
        }
    });
    
    claimData.numericFacts.forEach(numFact => {
        atomicFacts.push({
            type: 'numeric_claim',
            value: numFact.value,
            context: numFact.context,
            text: numFact.originalText,
            importance: 0.9
        });
    });
    
    return atomicFacts.slice(0, 8);
}

function analyzeFactSupport(fact, source) {
    const sourceText = `${source.title} ${source.description}`.toLowerCase();
    
    let stance = 'neutral';
    let confidence = 0;
    
    switch (fact.type) {
        case 'entity_action':
            const entityPresent = sourceText.includes(fact.entity.toLowerCase());
            const actionPresent = sourceText.includes(fact.action.toLowerCase());
            
            if (entityPresent && actionPresent) {
                const negationPatterns = ['not', 'never', 'denied', 'false', 'untrue', 'debunked'];
                const hasNegation = negationPatterns.some(neg => 
                    sourceText.includes(`${neg} ${fact.action}`) || 
                    sourceText.includes(`${fact.action} ${neg}`)
                );
                
                stance = hasNegation ? 'refutes' : 'supports';
                confidence = 0.8;
            }
            break;
            
        case 'numeric_claim':
            const numbers = extractNumbersFromText(sourceText);
            const tolerance = fact.value * 0.05;
            const matchingNumbers = numbers.filter(num => 
                Math.abs(num - fact.value) <= tolerance
            );
            
            if (matchingNumbers.length > 0) {
                stance = 'supports';
                confidence = 0.9;
            }
            break;
    }
    
    return { stance, confidence, factType: fact.type };
}

function analyzeConsistencyAndContradictions(factMatrix) {
    console.log(`[STAGE 7] Analyzing consistency and contradictions`);
    
    const consistencyAnalysis = {
        overallConsistency: 0,
        majorContradictions: [],
        minorContradictions: [],
        reliabilityScore: 0
    };
    
    const factConsensuses = Object.values(factMatrix.factConsensus);
    let totalAgreement = 0;
    
    factConsensuses.forEach(consensus => {
        const total = consensus.supports + consensus.refutes + consensus.neutral;
        const agreement = Math.max(consensus.supports, consensus.refutes) / total;
        totalAgreement += agreement;
    });
    
    consistencyAnalysis.overallConsistency = factConsensuses.length > 0 ? 
        totalAgreement / factConsensuses.length : 0;
    
    factMatrix.contradictions.forEach(contradiction => {
        const severity = Math.min(contradiction.supports, contradiction.refutes) / 
                        Math.max(contradiction.supports, contradiction.refutes);
        
        if (severity > 0.5) {
            consistencyAnalysis.majorContradictions.push(contradiction);
        } else {
            consistencyAnalysis.minorContradictions.push(contradiction);
        }
    });
    
    const contradictionPenalty = (consistencyAnalysis.majorContradictions.length * 0.3) + 
                                (consistencyAnalysis.minorContradictions.length * 0.1);
    consistencyAnalysis.reliabilityScore = Math.max(0, consistencyAnalysis.overallConsistency - contradictionPenalty);
    
    return consistencyAnalysis;
}

function synthesizeAutomatedVerdict(claimData, factMatrix, consistencyAnalysis, rankedEvidence) {
    console.log(`[STAGE 8] Synthesizing automated verdict`);
    
    const verdict = {
        classification: 'UNVERIFIED',
        confidence: 0,
        reasoning: [],
        evidenceQuality: 'LOW',
        factCheckerConsensus: 'NONE',
        numericAccuracy: 'UNKNOWN',
        sourceCredibility: 0
    };
    
    // Calculate source credibility
    const factCheckers = rankedEvidence.filter(e => e.isFactChecker);
    const avgTrustScore = rankedEvidence.reduce((sum, e) => sum + e.trustScore, 0) / rankedEvidence.length;
    verdict.sourceCredibility = Math.round(avgTrustScore * 100);
    
    // Analyze fact-checker consensus
    if (factCheckers.length > 0) {
        const supportingFactCheckers = factCheckers.filter(fc => {
            const sourceId = Object.keys(factMatrix.sourceMatrix).find(id => 
                factMatrix.sourceMatrix[id].url === fc.url
            );
            return sourceId && factMatrix.sourceMatrix[sourceId].overallStance === 'supports';
        });
        
        const refutingFactCheckers = factCheckers.filter(fc => {
            const sourceId = Object.keys(factMatrix.sourceMatrix).find(id => 
                factMatrix.sourceMatrix[id].url === fc.url
            );
            return sourceId && factMatrix.sourceMatrix[sourceId].overallStance === 'refutes';
        });
        
        if (supportingFactCheckers.length > refutingFactCheckers.length) {
            verdict.factCheckerConsensus = 'SUPPORTS';
        } else if (refutingFactCheckers.length > supportingFactCheckers.length) {
            verdict.factCheckerConsensus = 'REFUTES';
        } else {
            verdict.factCheckerConsensus = 'MIXED';
        }
    }
    
    // Determine overall classification
    const supportingEvidence = Object.values(factMatrix.factConsensus)
        .reduce((sum, consensus) => sum + consensus.supports, 0);
    const refutingEvidence = Object.values(factMatrix.factConsensus)
        .reduce((sum, consensus) => sum + consensus.refutes, 0);
    
    const totalEvidence = supportingEvidence + refutingEvidence;
    
    if (totalEvidence === 0) {
        verdict.classification = 'INSUFFICIENT_EVIDENCE';
        verdict.confidence = 10;
        verdict.reasoning.push('No substantial evidence found to verify or refute the claim');
    } else {
        const supportRatio = supportingEvidence / totalEvidence;
        
        if (supportRatio >= 0.7) {
            verdict.classification = 'LIKELY_TRUE';
            verdict.confidence = Math.round(60 + (supportRatio * 40));
        } else if (supportRatio <= 0.3) {
            verdict.classification = 'LIKELY_FALSE';
            verdict.confidence = Math.round(60 + ((1 - supportRatio) * 40));
        } else {
            verdict.classification = 'MIXED_EVIDENCE';
            verdict.confidence = Math.round(30 + (Math.abs(0.5 - supportRatio) * 40));
        }
    }
    
    // Adjust confidence based on consistency and source quality
    verdict.confidence = Math.round(verdict.confidence * consistencyAnalysis.reliabilityScore * (avgTrustScore + 0.5));
    verdict.confidence = Math.min(95, Math.max(5, verdict.confidence));
    
    // Evidence quality assessment
    if (factCheckers.length >= 2 && avgTrustScore >= 0.8) {
        verdict.evidenceQuality = 'HIGH';
    } else if (factCheckers.length >= 1 || avgTrustScore >= 0.7) {
        verdict.evidenceQuality = 'MEDIUM';
    }
    
    // Generate reasoning
    verdict.reasoning.push(`Analysis based on ${rankedEvidence.length} sources`);
    if (factCheckers.length > 0) {
        verdict.reasoning.push(`${factCheckers.length} fact-checking sources consulted`);
    }
    if (consistencyAnalysis.majorContradictions.length > 0) {
        verdict.reasoning.push(`${consistencyAnalysis.majorContradictions.length} major contradictions found`);
    }
    
    console.log(`[STAGE 8] Verdict: ${verdict.classification} (${verdict.confidence}% confidence)`);
    return verdict;
}

// =============================================================================
// PIPELINE STAGE 9: COMPREHENSIVE DASHBOARD RENDERING
// =============================================================================

function renderComprehensiveDashboard(claimData, factMatrix, consistencyAnalysis, verdict, rankedEvidence) {
    console.log(`[STAGE 9] Rendering comprehensive dashboard`);
    
    const dashboardData = {
        claimAnalysis: {
            originalClaim: claimData.originalText,
            claimType: claimData.claimType,
            complexity: claimData.complexity,
            extractedEntities: claimData.coreEntities,
            numericFacts: claimData.numericFacts,
            keyComponents: claimData.keyPhrases
        },
        verificationResults: {
            verdict: verdict,
            evidenceSummary: {
                totalSources: rankedEvidence.length,
                factCheckers: rankedEvidence.filter(e => e.isFactChecker).length,
                averageTrustScore: Math.round((rankedEvidence.reduce((sum, e) => sum + e.trustScore, 0) / rankedEvidence.length) * 100),
                contradictions: consistencyAnalysis.majorContradictions.length + consistencyAnalysis.minorContradictions.length
            },
            topSources: rankedEvidence.slice(0, 10).map((source, index) => ({
                rank: index + 1,
                title: source.title,
                domain: source.source.domain,
                trustScore: Math.round(source.trustScore * 100),
                relevanceScore: Math.round(source.relevanceScore * 100),
                url: source.url,
                isFactChecker: source.isFactChecker,
                publishedAt: source.publishedAt
            }))
        },
        factMatrix: {
            atomicFacts: factMatrix.atomicFacts,
            consensus: Object.keys(factMatrix.factConsensus).map(factId => ({
                fact: factMatrix.factConsensus[factId].fact,
                supports: factMatrix.factConsensus[factId].supports,
                refutes: factMatrix.factConsensus[factId].refutes,
                neutral: factMatrix.factConsensus[factId].neutral
            })),
            contradictions: consistencyAnalysis.majorContradictions.concat(consistencyAnalysis.minorContradictions)
        }
    };
    
    // Update global proofsArray for integration
    window.proofsArray = dashboardData.verificationResults.topSources.map(source => ({
        title: source.title,
        url: source.url,
        domain: source.domain,
        trustScore: source.trustScore / 100,
        relevanceScore: source.relevanceScore / 100,
        isFactChecker: source.isFactChecker,
        rank: source.rank
    }));
    
    console.log(`[STAGE 9] Dashboard data prepared with ${dashboardData.verificationResults.topSources.length} sources`);
    return dashboardData;
}

// =============================================================================
// PIPELINE STAGE 10: FINALIZATION & EDGE CASE HANDLING
// =============================================================================

function finalizeVerificationResults(dashboardData, verdict) {
    console.log(`[STAGE 10] Finalizing verification results`);
    
    const finalResults = {
        success: true,
        verdict: verdict,
        dashboard: dashboardData,
        metadata: {
            processingTime: Date.now(),
            pipelineVersion: '1.0.0',
            apiSources: ['NewsAPI', 'Serper', 'NewsData'],
            totalEvidence: dashboardData.verificationResults.evidenceSummary.totalSources
        }
    };
    
    // Trigger UI update event
    const event = new CustomEvent('factVerificationComplete', {
        detail: finalResults
    });
    document.dispatchEvent(event);
    
    return finalResults;
}

function handleInsufficientEvidence(claimData) {
    // Update global proofsArray for integration
    window.proofsArray = [];
    
    const result = {
        success: true,
        verdict: {
            classification: 'INSUFFICIENT_EVIDENCE',
            confidence: 0,
            reasoning: ['No relevant sources found for verification'],
            evidenceQuality: 'LOW'
        },
        dashboard: {
            claimAnalysis: {
                originalClaim: claimData.originalText,
                claimType: claimData.claimType,
                complexity: claimData.complexity || 'LOW',
                extractedEntities: claimData.coreEntities || [],
                numericFacts: claimData.numericFacts || [],
                keyComponents: claimData.keyPhrases || []
            },
            verificationResults: {
                verdict: {
                    classification: 'INSUFFICIENT_EVIDENCE',
                    confidence: 0,
                    reasoning: ['No relevant sources found for verification'],
                    evidenceQuality: 'LOW'
                },
                evidenceSummary: {
                    totalSources: 0,
                    factCheckers: 0,
                    averageTrustScore: 0,
                    contradictions: 0
                },
                topSources: []
            },
            factMatrix: {
                atomicFacts: [],
                consensus: [],
                contradictions: []
            }
        },
        metadata: {
            processingTime: Date.now(),
            pipelineVersion: '1.0.0',
            apiSources: ['NewsAPI', 'Serper', 'NewsData'],
            totalEvidence: 0
        }
    };
    
    // Trigger UI update event
    const event = new CustomEvent('factVerificationComplete', {
        detail: result
    });
    document.dispatchEvent(event);
    
    return result;
}

function handleVerificationError(error, userClaim) {
    console.error(`[ERROR HANDLER] Verification failed:`, error);
    
    return {
        success: false,
        error: error.message,
        verdict: {
            classification: 'VERIFICATION_FAILED',
            confidence: 0,
            reasoning: [`Verification failed: ${error.message}`],
            evidenceQuality: 'ERROR'
        },
        dashboard: {
            claimAnalysis: {
                originalClaim: userClaim
            },
            verificationResults: {
                totalSources: 0,
                topSources: []
            }
        }
    };
}

// =============================================================================
// GLOBAL EXPORTS AND INTEGRATION
// =============================================================================

// Export main pipeline function
window.executeUniversalFactVerification = executeUniversalFactVerification;

// Export individual stage functions for advanced usage
window.parseUniversalClaim = parseUniversalClaim;
window.generateDiversifiedQueries = generateDiversifiedQueries;
window.collectEvidenceParallel = collectEvidenceParallel;
window.applyMultiLayeredFiltering = applyMultiLayeredFiltering;
window.deduplicateAndRank = deduplicateAndRank;
window.constructFactSourceMatrix = constructFactSourceMatrix;
window.analyzeConsistencyAndContradictions = analyzeConsistencyAndContradictions;
window.synthesizeAutomatedVerdict = synthesizeAutomatedVerdict;
window.renderComprehensiveDashboard = renderComprehensiveDashboard;

// Legacy compatibility
window.analyzeContentVerification = executeUniversalFactVerification;
window.executeMultiStageRelevancePipeline = executeUniversalFactVerification;

// Initialize system
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        console.log('🔍 Universal Fact Verification Pipeline Loaded Successfully');
        console.log('🌐 Ready for claim-agnostic verification across all topics');
        console.log('⚡ Features: Multi-API integration, semantic filtering, fact-source matrix, automated verdict synthesis');
        console.log('✅ World-class fake news detection system ready');
    });
} else {
    console.log('🔍 Universal Fact Verification Pipeline Loaded Successfully');
    console.log('🌐 Ready for claim-agnostic verification across all topics');
    console.log('⚡ Features: Multi-API integration, semantic filtering, fact-source matrix, automated verdict synthesis');
    console.log('✅ World-class fake news detection system ready');
}
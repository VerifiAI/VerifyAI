/**
 * Serper API Integration Module
 * Direct frontend integration for news verification and fact-checking
 */

class SerperAPI {
    constructor(apiKey) {
        this.apiKey = apiKey;
        this.baseUrl = 'https://google.serper.dev/search';
        this.newsUrl = 'https://google.serper.dev/news';
    }

    /**
     * Search for news articles related to a claim
     */
    async searchNews(query, options = {}) {
        const searchParams = {
            q: query,
            num: options.num || 10,
            gl: options.country || 'us',
            hl: options.language || 'en'
        };

        try {
            const response = await fetch(this.newsUrl, {
                method: 'POST',
                headers: {
                    'X-API-KEY': this.apiKey,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(searchParams)
            });

            if (!response.ok) {
                throw new Error(`Serper API error: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();
            return this.processNewsResults(data);
        } catch (error) {
            console.error('Serper News API Error:', error);
            throw error;
        }
    }

    /**
     * General web search for fact-checking
     */
    async searchWeb(query, options = {}) {
        const searchParams = {
            q: query,
            num: options.num || 10,
            gl: options.country || 'us',
            hl: options.language || 'en'
        };

        try {
            const response = await fetch(this.baseUrl, {
                method: 'POST',
                headers: {
                    'X-API-KEY': this.apiKey,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(searchParams)
            });

            if (!response.ok) {
                throw new Error(`Serper API error: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();
            return this.processWebResults(data);
        } catch (error) {
            console.error('Serper Web API Error:', error);
            throw error;
        }
    }

    /**
     * Verify news claim by cross-referencing multiple sources
     */
    async verifyNewsClaim(claim, options = {}) {
        try {
            // Extract key terms from the claim
            const keywords = this.extractKeywords(claim);
            
            // Search for related news articles
            const newsResults = await this.searchNews(keywords, { num: 15 });
            
            // Search for fact-checking sources
            const factCheckQuery = `"${claim}" fact check OR debunk OR verify`;
            const factCheckResults = await this.searchWeb(factCheckQuery, { num: 10 });
            
            // Analyze results and generate verification report
            return this.generateVerificationReport(claim, newsResults, factCheckResults);
        } catch (error) {
            console.error('News verification error:', error);
            throw error;
        }
    }

    /**
     * Extract keywords from news claim
     */
    extractKeywords(text) {
        // Remove common stop words and extract meaningful terms
        const stopWords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'];
        
        const words = text.toLowerCase()
            .replace(/[^\w\s]/g, ' ')
            .split(/\s+/)
            .filter(word => word.length > 2 && !stopWords.includes(word));
        
        // Return top keywords (limit to avoid too long queries)
        return words.slice(0, 8).join(' ');
    }

    /**
     * Process news search results
     */
    processNewsResults(data) {
        if (!data.news || !Array.isArray(data.news)) {
            return { articles: [], totalResults: 0 };
        }

        const articles = data.news.map(article => ({
            title: article.title || '',
            snippet: article.snippet || '',
            link: article.link || '',
            source: article.source || '',
            date: article.date || '',
            imageUrl: article.imageUrl || ''
        }));

        return {
            articles,
            totalResults: articles.length,
            searchTime: data.searchParameters?.searchTime || 0
        };
    }

    /**
     * Process web search results
     */
    processWebResults(data) {
        if (!data.organic || !Array.isArray(data.organic)) {
            return { results: [], totalResults: 0 };
        }

        const results = data.organic.map(result => ({
            title: result.title || '',
            snippet: result.snippet || '',
            link: result.link || '',
            displayLink: result.displayLink || ''
        }));

        return {
            results,
            totalResults: results.length,
            searchTime: data.searchParameters?.searchTime || 0
        };
    }

    /**
     * Generate comprehensive verification report
     */
    generateVerificationReport(claim, newsResults, factCheckResults) {
        const report = {
            claim,
            timestamp: new Date().toISOString(),
            verification: {
                status: 'unknown',
                confidence: 0,
                reasoning: []
            },
            sources: {
                news: newsResults.articles || [],
                factCheck: factCheckResults.results || []
            },
            analysis: {
                supportingEvidence: [],
                contradictingEvidence: [],
                neutralEvidence: []
            }
        };

        // Analyze fact-checking sources
        const factCheckAnalysis = this.analyzeFactCheckSources(factCheckResults.results || []);
        
        // Analyze news coverage
        const newsAnalysis = this.analyzeNewsCoverage(newsResults.articles || [], claim);
        
        // Determine verification status
        report.verification = this.determineVerificationStatus(factCheckAnalysis, newsAnalysis);
        
        // Categorize evidence
        report.analysis = this.categorizeEvidence(newsResults.articles || [], factCheckResults.results || [], claim);
        
        return report;
    }

    /**
     * Analyze fact-checking sources for verification indicators
     */
    analyzeFactCheckSources(factCheckResults) {
        const factCheckKeywords = {
            false: ['false', 'fake', 'debunked', 'misleading', 'incorrect', 'untrue', 'hoax'],
            true: ['true', 'accurate', 'confirmed', 'verified', 'correct'],
            mixed: ['partially', 'mixed', 'mostly', 'some truth']
        };

        let falseCount = 0;
        let trueCount = 0;
        let mixedCount = 0;

        factCheckResults.forEach(result => {
            const text = (result.title + ' ' + result.snippet).toLowerCase();
            
            factCheckKeywords.false.forEach(keyword => {
                if (text.includes(keyword)) falseCount++;
            });
            
            factCheckKeywords.true.forEach(keyword => {
                if (text.includes(keyword)) trueCount++;
            });
            
            factCheckKeywords.mixed.forEach(keyword => {
                if (text.includes(keyword)) mixedCount++;
            });
        });

        return { falseCount, trueCount, mixedCount, totalSources: factCheckResults.length };
    }

    /**
     * Analyze news coverage patterns
     */
    analyzeNewsCoverage(newsArticles, claim) {
        const reliableSources = ['reuters', 'ap', 'bbc', 'cnn', 'nytimes', 'washingtonpost', 'npr', 'pbs'];
        
        let reliableSourceCount = 0;
        let totalCoverage = newsArticles.length;
        
        newsArticles.forEach(article => {
            const source = article.source.toLowerCase();
            if (reliableSources.some(reliable => source.includes(reliable))) {
                reliableSourceCount++;
            }
        });

        return {
            totalCoverage,
            reliableSourceCount,
            coverageRatio: totalCoverage > 0 ? reliableSourceCount / totalCoverage : 0
        };
    }

    /**
     * Determine overall verification status
     */
    determineVerificationStatus(factCheckAnalysis, newsAnalysis) {
        let status = 'unknown';
        let confidence = 0;
        let reasoning = [];

        // Analyze fact-check results
        if (factCheckAnalysis.totalSources > 0) {
            if (factCheckAnalysis.falseCount > factCheckAnalysis.trueCount) {
                status = 'likely_false';
                confidence = Math.min(0.8, 0.4 + (factCheckAnalysis.falseCount / factCheckAnalysis.totalSources) * 0.4);
                reasoning.push(`${factCheckAnalysis.falseCount} fact-checking sources indicate this claim is false`);
            } else if (factCheckAnalysis.trueCount > factCheckAnalysis.falseCount) {
                status = 'likely_true';
                confidence = Math.min(0.8, 0.4 + (factCheckAnalysis.trueCount / factCheckAnalysis.totalSources) * 0.4);
                reasoning.push(`${factCheckAnalysis.trueCount} fact-checking sources support this claim`);
            } else if (factCheckAnalysis.mixedCount > 0) {
                status = 'mixed';
                confidence = 0.5;
                reasoning.push('Mixed evidence from fact-checking sources');
            }
        }

        // Factor in news coverage
        if (newsAnalysis.totalCoverage === 0) {
            reasoning.push('No recent news coverage found');
            confidence = Math.max(0, confidence - 0.2);
        } else if (newsAnalysis.reliableSourceCount > 0) {
            reasoning.push(`${newsAnalysis.reliableSourceCount} reliable news sources found`);
            confidence = Math.min(0.9, confidence + 0.1);
        }

        return { status, confidence, reasoning };
    }

    /**
     * Categorize evidence into supporting, contradicting, and neutral
     */
    categorizeEvidence(newsArticles, factCheckResults, claim) {
        const supportingEvidence = [];
        const contradictingEvidence = [];
        const neutralEvidence = [];

        // Categorize fact-check results
        factCheckResults.forEach(result => {
            const text = (result.title + ' ' + result.snippet).toLowerCase();
            
            if (text.includes('false') || text.includes('fake') || text.includes('debunked')) {
                contradictingEvidence.push({
                    type: 'fact_check',
                    title: result.title,
                    source: result.displayLink,
                    link: result.link,
                    snippet: result.snippet
                });
            } else if (text.includes('true') || text.includes('confirmed') || text.includes('verified')) {
                supportingEvidence.push({
                    type: 'fact_check',
                    title: result.title,
                    source: result.displayLink,
                    link: result.link,
                    snippet: result.snippet
                });
            } else {
                neutralEvidence.push({
                    type: 'fact_check',
                    title: result.title,
                    source: result.displayLink,
                    link: result.link,
                    snippet: result.snippet
                });
            }
        });

        // Categorize news articles (simplified - could be enhanced with NLP)
        newsArticles.forEach(article => {
            neutralEvidence.push({
                type: 'news',
                title: article.title,
                source: article.source,
                link: article.link,
                snippet: article.snippet,
                date: article.date
            });
        });

        return { supportingEvidence, contradictingEvidence, neutralEvidence };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SerperAPI;
} else {
    window.SerperAPI = SerperAPI;
}
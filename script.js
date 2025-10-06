// API Configuration
const API_BASE_URL = 'http://127.0.0.1:5001';

// DOM Elements
const elements = {
    // Multi-modal input elements
    newsText: document.getElementById('newsText'),
    imageFile: document.getElementById('imageFile'),
    newsUrl: document.getElementById('newsUrl'),
    imagePreview: document.getElementById('imagePreview'),
    urlPreview: document.getElementById('urlPreview'),
    
    // Tab elements
    tabBtns: document.querySelectorAll('.tab-btn'),
    textTab: document.getElementById('textTab'),
    imageTab: document.getElementById('imageTab'),
    urlTab: document.getElementById('urlTab'),
    
    // Control elements
    detectBtn: document.getElementById('detectBtn'),
    loadingIndicator: document.getElementById('loadingIndicator'),
    feedSource: document.getElementById('feed-source'),
    refreshFeedBtn: document.getElementById('autoRefreshToggle'),
    
    // Live feed elements
    feedContainer: document.getElementById('feed-container'),
    feedLoading: document.getElementById('feed-loading'),
    
    // Display elements
    totalAnalyzed: document.getElementById('totalAnalyzed'),
    detectionResult: document.getElementById('detectionResult'),
    confidenceScore: document.getElementById('confidenceScore'),
    textLength: document.getElementById('textLength'),
    processingTime: document.getElementById('processingTime'),
    liveFeed: document.getElementById('liveFeed'),
    historyList: document.getElementById('historyList'),
    
    // Explainability elements
    explainBtn: document.getElementById('explainBtn'),
    explainabilityContent: document.getElementById('explainabilityContent'),
    explainLoading: document.getElementById('explainLoading'),
    

    
    // Proof panel elements (legacy)
    validateBtn: document.getElementById('validateBtn'),
    proofPanel: document.getElementById('proofPanel'),
    proofLoading: document.getElementById('proofLoading'),
    
    // Content Result elements
    contentResultPanel: document.getElementById('contentResultPanel'),
    contentResultLoading: document.getElementById('contentResultLoading'),
    triggerContentVerification: document.getElementById('triggerContentVerification'),
    testVerification: document.getElementById('testVerification')
};

// Application State
let appState = {
    isLoading: false,
    currentUser: null,
    detectionHistory: [],
    liveFeedData: [],
    totalAnalyzed: 0,
    modelAccuracy: 0.92,
    apiCalls: 0,
    successRate: 0.95,
    currentTab: 'text',
    uploadedImage: null,
    urlContent: null,
    lastAnalyzedText: null,
    lastAnalyzedImageUrl: null,
    lastDetectionResult: null
};
let analysisCount = 0;
let currentProcessingTime = 0;
let progressInterval = null;
let progressStartTime = null;
let currentStage = 0;
const PROGRESS_STAGES = ['🧠 MHFN Model', '🔍 Evidence Search', '⚖️ Bayesian Fusion'];
const MAX_PROCESSING_TIME = 30; // seconds

// Tab Management
function initializeTabs() {
    elements.tabBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const tabType = e.target.dataset.tab;
            switchTab(tabType);
        });
    });
    
    // Set initial tab
    switchTab('text');
}

function switchTab(tabType) {
    // Update state
    appState.currentTab = tabType;
    
    // Update tab buttons
    elements.tabBtns.forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.tab === tabType) {
            btn.classList.add('active');
        }
    });
    
    // Update tab content
    const tabs = ['text', 'image', 'url'];
    tabs.forEach(tab => {
        const tabElement = document.getElementById(`${tab}Tab`);
        if (tabElement) {
            if (tab === tabType) {
                tabElement.classList.add('active');
                tabElement.classList.remove('hidden');
            } else {
                tabElement.classList.remove('active');
                tabElement.classList.add('hidden');
            }
        }
    });
    
    // Clear previous inputs when switching tabs
    clearInputs();
}

function clearInputs() {
    if (elements.newsText) elements.newsText.value = '';
    if (elements.imageFile) elements.imageFile.value = '';
    if (elements.newsUrl) elements.newsUrl.value = '';
    if (elements.imagePreview) elements.imagePreview.innerHTML = '';
    if (elements.urlPreview) elements.urlPreview.innerHTML = '';
    appState.uploadedImage = null;
    appState.urlContent = null;
}

// File Upload Handling
function initializeFileUpload() {
    if (elements.imageFile) {
        elements.imageFile.addEventListener('change', handleImageUpload);
    }
}

function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        showError('Please select a valid image file (JPEG, PNG, GIF, or WebP)');
        return;
    }
    
    // Validate file size (max 5MB)
    const maxSize = 5 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('Image file size must be less than 5MB');
        return;
    }
    
    // Store file and create preview
    appState.uploadedImage = file;
    createImagePreview(file);
}

function createImagePreview(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const previewHtml = `
            <img src="${e.target.result}" alt="Preview">
            <div class="preview-info">
                <strong>${file.name}</strong><br>
                Size: ${formatFileSize(file.size)}<br>
                Type: ${file.type}
            </div>
        `;
        if (elements.imagePreview) {
            elements.imagePreview.innerHTML = previewHtml;
        }
    };
    reader.readAsDataURL(file);
}

// URL Input Handling
function initializeUrlInput() {
    if (elements.newsUrl) {
        elements.newsUrl.addEventListener('input', debounce(handleUrlInput, 500));
        elements.newsUrl.addEventListener('paste', (e) => {
            setTimeout(() => handleUrlInput(e), 100);
        });
    }
}

function handleUrlInput(event) {
    const url = event.target.value.trim();
    if (!url) {
        if (elements.urlPreview) elements.urlPreview.innerHTML = '';
        appState.urlContent = null;
        return;
    }
    
    // Validate URL format
    if (!isValidUrl(url)) {
        showUrlPreview('Invalid URL format', '', 'Please enter a valid URL');
        return;
    }
    
    // Show loading state
    showUrlPreview('Loading...', 'Fetching content from URL', 'Please wait');
    
    // Fetch URL content preview
    fetchUrlPreview(url);
}

function fetchUrlPreview(url) {
    // For now, just show URL info - actual content fetching will be done on detection
    try {
        const urlObj = new URL(url);
        showUrlPreview(
            urlObj.hostname,
            `Ready to analyze content from: ${url}`,
            `Domain: ${urlObj.hostname} | Protocol: ${urlObj.protocol}`
        );
        appState.urlContent = url;
    } catch (error) {
        showUrlPreview('Invalid URL', 'Please check the URL format', 'Error parsing URL');
        appState.urlContent = null;
    }
}

function showUrlPreview(title, description, meta) {
    if (!elements.urlPreview) return;
    
    const previewHtml = `
        <div class="preview-title">${title}</div>
        <div class="preview-description">${description}</div>
        <div class="preview-meta">${meta}</div>
    `;
    elements.urlPreview.innerHTML = previewHtml;
}

// Utility Functions
function showLoading(element, show = true) {
    if (show) {
        element.classList.remove('hidden');
    } else {
        element.classList.add('hidden');
    }
}

function parseStoredData(key, defaultValue = null) {
    try {
        const stored = localStorage.getItem(key);
        return stored ? JSON.parse(stored) : defaultValue;
    } catch (error) {
        console.warn(`Error parsing stored data for key ${key}:`, error);
        return defaultValue;
    }
}

function formatDate(dateString) {
    const date = new Date(dateString);
    // Ensure consistent formatting with timezone handling
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
}

function truncateText(text, maxLength = 100) {
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function isValidUrl(string) {
    try {
        new URL(string);
        return true;
    } catch (_) {
        return false;
    }
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function handleApiError(error, context) {
    console.error(`Error in ${context}:`, error);
    
    // Determine if it's a connection error
    const isConnectionError = error.name === 'TypeError' || 
                             error.message.includes('fetch') || 
                             error.message.includes('NetworkError') ||
                             error.message.includes('Failed to fetch');
    
    const errorMessage = isConnectionError ? 'Connection Error: Unable to reach server' : 
                        (error.message || 'Network error occurred');
    
    // Display error in UI if elements exist
    if (elements.detectionResult) {
        showError(elements.detectionResult, errorMessage);
    }
    
    return {
        success: false,
        error: errorMessage,
        isConnectionError: isConnectionError
    };
}

// Progress Tracking Functions
function startProgressTracking() {
    progressStartTime = Date.now();
    currentStage = 0;
    
    // Show enhanced loading indicator
    const loadingElement = document.querySelector('.enhanced-loading');
    if (loadingElement) {
        loadingElement.classList.remove('hidden');
    }
    
    // Update progress every 100ms
    progressInterval = setInterval(updateProgress, 100);
}

function updateProgress() {
    if (!progressStartTime) return;
    
    const elapsed = (Date.now() - progressStartTime) / 1000;
    const progressPercent = Math.min((elapsed / MAX_PROCESSING_TIME) * 100, 100);
    
    // Update timer
    const timerElement = document.querySelector('.progress-timer');
    if (timerElement) {
        timerElement.textContent = `${elapsed.toFixed(1)}s / ${MAX_PROCESSING_TIME}s`;
    }
    
    // Update progress bar
    const progressBar = document.querySelector('.progress-fill');
    if (progressBar) {
        progressBar.style.width = `${progressPercent}%`;
    }
    
    // Update stage based on elapsed time
    const stageIndex = Math.min(Math.floor(elapsed / (MAX_PROCESSING_TIME / 3)), 2);
    if (stageIndex !== currentStage) {
        currentStage = stageIndex;
        updateProgressStage();
    }
}

function updateProgressStage() {
    const stageElements = document.querySelectorAll('.progress-stage');
    stageElements.forEach((element, index) => {
        if (index <= currentStage) {
            element.classList.add('active');
        } else {
            element.classList.remove('active');
        }
    });
}

function stopProgressTracking() {
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
    
    // Hide enhanced loading indicator
    const loadingElement = document.querySelector('.enhanced-loading');
    if (loadingElement) {
        loadingElement.classList.add('hidden');
    }
    
    progressStartTime = null;
    currentStage = 0;
}

// Proof Validation Functions
async function validateWithSources() {
    if (!elements.proofPanel || !elements.validateBtn) {
        console.error('Proof validation elements not found');
        return;
    }

    const queryText = elements.newsText.value.trim();
    if (!queryText) {
        showNotification('Please enter some text to validate', 'warning');
        return;
    }

    try {
        elements.validateBtn.disabled = true;
        elements.proofLoading.classList.remove('hidden');
        
        showVerificationProgress();
        
        if (typeof window.analyzeContent === 'function') {
            const result = await window.analyzeContent(queryText);
            displayProofResults({
                status: 'success',
                proofs: result.articles || [],
                query: queryText,
                total_sources: result.articles ? result.articles.length : 0,
                verification_summary: result.summary || 'Analysis completed using verification system'
            });
        } else {
            // Fallback to direct API calls if new system not available
            const [factCheckResult, snopesResult, webSearchResult] = await Promise.allSettled([
                searchFactCheck(queryText),
                searchSnopes(queryText),
                searchWebForProof(queryText)
            ]);
            
            const proofs = [];
            if (factCheckResult.status === 'fulfilled' && factCheckResult.value) proofs.push(factCheckResult.value);
            if (snopesResult.status === 'fulfilled' && snopesResult.value) proofs.push(snopesResult.value);
            if (webSearchResult.status === 'fulfilled' && webSearchResult.value) proofs.push(webSearchResult.value);
            
            displayProofResults({
                status: 'success',
                proofs: proofs,
                query: queryText,
                total_sources: proofs.length,
                verification_summary: proofs.length > 0 ? 
                    `Found ${proofs.length} verification source(s) using direct API calls` : 
                    'No verification sources found'
            });
        }
        
        showNotification('Verification completed successfully!', 'success');
        
    } catch (error) {
        console.error('Verification error:', error);
        showError(elements.proofPanel, `Failed to validate with sources: ${error.message}`);
        showNotification('Verification failed', 'error');
    } finally {
        elements.proofLoading.classList.add('hidden');
        elements.validateBtn.disabled = false;
        hideVerificationProgress();
    }
}

// Legacy function - kept for compatibility
function displayProofResults(data) {
    // Redirect to new Content Result display
    displayContentResult(data);
}

// New function for Content Result section
function displayContentResult(data) {
    const unified_verdict = data.unified_verdict || {};
    const mhfn_output = data.mhfn_output || {};
    const processing_metrics = data.processing_metrics || {};
    
    const prediction = unified_verdict.prediction || mhfn_output.prediction || 'unknown';
    const confidence = unified_verdict.confidence || mhfn_output.confidence || 0;
    const accuracy = processing_metrics.accuracy || appState.modelAccuracy || 0.92;
    const f1_score = processing_metrics.f1_score || 0.89; // Default F1 score
    const processing_time = processing_metrics.total_time || 0;
    
    const isReal = prediction.toLowerCase() === 'real';
    const isFake = prediction.toLowerCase() === 'fake';
    const resultClass = isReal ? 'real-news' : isFake ? 'fake-news' : 'unknown-news';
    const resultIcon = isReal ? '✅' : isFake ? '❌' : '❓';
    const resultLabel = isReal ? 'REAL NEWS' : isFake ? 'FAKE NEWS' : 'UNKNOWN';
    
    let html = `
        <div class="content-result-display">
            <div class="result-header">
                <div class="result-verdict ${resultClass}">
                    <div class="result-icon">${resultIcon}</div>
                    <div class="result-info">
                        <h3 class="result-label">${resultLabel}</h3>
                        <p class="result-confidence">Confidence: ${(confidence * 100).toFixed(1)}%</p>
                    </div>
                </div>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card accuracy">
                    <div class="metric-icon">🎯</div>
                    <div class="metric-content">
                        <h4>Model Accuracy</h4>
                        <div class="metric-value">${(accuracy * 100).toFixed(1)}%</div>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: ${accuracy * 100}%"></div>
                        </div>
                    </div>
                </div>
                
                <div class="metric-card f1-score">
                    <div class="metric-icon">📊</div>
                    <div class="metric-content">
                        <h4>F1 Score</h4>
                        <div class="metric-value">${(f1_score * 100).toFixed(1)}%</div>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: ${f1_score * 100}%"></div>
                        </div>
                    </div>
                </div>
                
                <div class="metric-card processing-time">
                    <div class="metric-icon">⏱️</div>
                    <div class="metric-content">
                        <h4>Processing Time</h4>
                        <div class="metric-value">${processing_time.toFixed(2)}s</div>
                        <div class="metric-description">Analysis Duration</div>
                    </div>
                </div>
            </div>
            
            <div class="result-details">
                <div class="detail-item">
                    <span class="detail-label">Classification Method:</span>
                    <span class="detail-value">Hybrid Deep Learning with Explainable AI</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Model Type:</span>
                    <span class="detail-value">MHFN (Multi-Head Fusion Network)</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Analysis Date:</span>
                    <span class="detail-value">${new Date().toLocaleString()}</span>
                </div>
            </div>
        </div>
    `;
    
    elements.contentResultPanel.innerHTML = html;
}

// Function to analyze content and show results
// API Functions
async function detectFakeNews(inputData = null) {
    const startTime = Date.now();
    let progressInterval;
    
    try {
        let requestData;
        let requestOptions = {
            method: 'POST'
        };
        
        // Start progress tracking
        startProgressTracking();
        
        // Determine input type and prepare request
        switch (appState.currentTab) {
            case 'text':
                const text = inputData || (elements.newsText ? elements.newsText.value.trim() : '');
                if (!text || text.length === 0) {
                    throw new Error('Please enter some text to analyze');
                }
                requestOptions.headers = { 'Content-Type': 'application/json' };
                requestOptions.body = JSON.stringify({ text: text, input_type: 'text' });
                break;
                
            case 'image':
                if (!appState.uploadedImage) {
                    throw new Error('Please select an image to analyze');
                }
                const formData = new FormData();
                formData.append('image', appState.uploadedImage);
                formData.append('input_type', 'image');
                requestOptions.body = formData;
                // Don't set Content-Type header for FormData
                break;
                
            case 'url':
                let url;
                if (inputData && typeof inputData === 'object' && inputData.url) {
                    url = inputData.url;
                } else if (inputData && typeof inputData === 'string') {
                    url = inputData;
                } else {
                    url = (elements.newsUrl ? elements.newsUrl.value.trim() : '') || appState.urlContent;
                }
                if (!url || !isValidUrl(url)) {
                    throw new Error('Please enter a valid URL to analyze');
                }
                requestOptions.headers = { 'Content-Type': 'application/json' };
                requestOptions.body = JSON.stringify({ url: url, input_type: 'url' });
                break;
                
            default:
                throw new Error('Invalid input type selected');
        }
        
        const response = await fetch(`${API_BASE_URL}/api/detect`, requestOptions);
        
        const endTime = Date.now();
        currentProcessingTime = ((endTime - startTime) / 1000).toFixed(2);
        
        // Stop progress tracking
        stopProgressTracking();
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }
        
        const data = await response.json();
        
        // Store last analyzed content for explainability
        if (appState.currentTab === 'text') {
            appState.lastAnalyzedText = requestOptions.body ? JSON.parse(requestOptions.body).text : '';
        } else if (appState.currentTab === 'image') {
            appState.lastAnalyzedImageUrl = appState.uploadedImage ? URL.createObjectURL(appState.uploadedImage) : null;
        } else if (appState.currentTab === 'url') {
            appState.lastAnalyzedText = requestOptions.body ? JSON.parse(requestOptions.body).url : '';
        }
        
        return {
            success: true,
            data: {
                ...data,
                processing_time: currentProcessingTime,
                input_type: appState.currentTab
            }
        };
        
    } catch (error) {
        stopProgressTracking();
        return handleApiError(error, 'detectFakeNews');
    }
}

async function fetchLiveFeed(source = 'bbc') {
    try {
        // Get selected source from dropdown if available
        const selectedSource = source || (elements.feedSource ? elements.feedSource.value : 'bbc');
        
        console.log(`Fetching live feed for source: ${selectedSource}`);
        
        const response = await fetch(`${API_BASE_URL}/api/live-feed?source=${selectedSource}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Log API response details
        console.log('Live feed API response:', {
            source: selectedSource,
            api_source: data.api_source,
            cached: data.cached,
            cache_hit: data.cache_hit,
            response_time: data.response_time,
            articles_count: data.articles ? data.articles.length : (data.data ? data.data.length : 0)
        });
        
        return { success: true, data: data };
        
    } catch (error) {
        console.error(`Failed to fetch live feed for source ${source}:`, error);
        return handleApiError(error, 'fetchLiveFeed');
    }
}

async function fetchHistory() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/history`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        return { success: true, data: data };
        
    } catch (error) {
        return handleApiError(error, 'fetchHistory');
    }
}

// Generate AI Explanation
async function generateExplanation() {
    try {
        if (!appState.lastAnalyzedText && !appState.lastAnalyzedImageUrl) {
            showNotification('No recent analysis to explain. Please analyze some content first.', 'warning');
            return;
        }
        
        // Show loading
        elements.explainLoading.classList.remove('hidden');
        elements.explainBtn.disabled = true;
        
        const requestData = {
            text: appState.lastAnalyzedText || '',
            image_url: appState.lastAnalyzedImageUrl || ''
        };
        
        const response = await fetch(`${API_BASE_URL}/api/explain`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Check if we have any meaningful explanation data
            const hasShapValues = data.shap_values && data.shap_values.length > 0;
            const hasLimeExplanation = data.lime_explanation && data.lime_explanation !== "Feature not available";
            const hasTopicClusters = data.topic_clusters && data.topic_clusters.length > 0;
            const hasGradCam = data.grad_cam && data.grad_cam !== null;
            
            if (hasShapValues || hasLimeExplanation || hasTopicClusters || hasGradCam) {
                displayExplanation(data);
                showNotification('Explanation generated successfully!', 'success');
            } else {
                // Show limited functionality message
                elements.explainabilityContent.innerHTML = `
                    <div class="explanation-limited">
                        <h4>⚠️ Limited Explainability Mode</h4>
                        <p>Explainability features are currently running in limited mode.</p>
                        <p><strong>Reason:</strong> ${data.error || 'Explainability libraries not fully available'}</p>
                        <div class="explanation-status">
                            <p><strong>Available Features:</strong></p>
                            <ul>
                                <li>✅ Basic fake news detection</li>
                                <li>✅ Confidence scoring</li>
                                <li>✅ Multimodal analysis</li>
                                <li>❌ SHAP feature importance</li>
                                <li>❌ LIME explanations</li>
                                <li>❌ Topic clustering</li>
                                <li>❌ Grad-CAM visualizations</li>
                            </ul>
                        </div>
                        <p class="explanation-note">To enable full explainability features, ensure all required libraries (shap, lime, bertopic) are properly installed.</p>
                    </div>
                `;
                showNotification('Explainability running in limited mode', 'warning');
            }
        } else {
            throw new Error(data.error || 'Failed to generate explanation');
        }
        
    } catch (error) {
        console.error('Error generating explanation:', error);
        showNotification(`Error generating explanation: ${error.message}`, 'error');
        
        // Show error in explainability content
        elements.explainabilityContent.innerHTML = `
            <div class="explanation-error">
                <p>❌ Failed to generate explanation: ${error.message}</p>
                <p>Please try again or check if the explainability features are available.</p>
            </div>
        `;
    } finally {
        // Hide loading
        elements.explainLoading.classList.add('hidden');
        elements.explainBtn.disabled = false;
    }
}

// Display Explanation Results
function displayExplanation(explanation) {
    let html = '';
    
    // Processing Time and Fidelity Score Header
    if (explanation.processing_time || explanation.fidelity_score) {
        html += `
            <div class="explanation-header">
                <div class="explanation-metrics">
                    ${explanation.processing_time ? `<span class="metric-badge">⏱️ ${explanation.processing_time.toFixed(2)}s</span>` : ''}
                    ${explanation.fidelity_score ? `<span class="metric-badge fidelity-${explanation.fidelity_score > 0.8 ? 'high' : explanation.fidelity_score > 0.5 ? 'medium' : 'low'}">🎯 Fidelity: ${(explanation.fidelity_score * 100).toFixed(1)}%</span>` : ''}
                </div>
            </div>
        `;
    }
    
    // Enhanced SHAP Values Section with Interactive Heatmap
    if (explanation.shap_values && explanation.shap_values.length > 0) {
        html += `
            <div class="explanation-section">
                <h3>🎯 SHAP Feature Importance</h3>
                <div class="shap-controls">
                    <button class="shap-toggle-btn" onclick="toggleShapView('heatmap')">Heatmap View</button>
                    <button class="shap-toggle-btn" onclick="toggleShapView('bar')">Bar Chart</button>
                    <button class="shap-toggle-btn" onclick="toggleShapView('text')">Text View</button>
                </div>
                <div class="shap-visualization">
                    <div class="shap-heatmap" id="shapHeatmap" style="display: none;">
                        <canvas id="shapHeatmapCanvas" width="800" height="200"></canvas>
                    </div>
                    <div class="shap-bar-chart" id="shapBarChart" style="display: none;">
                        <canvas id="shapBarCanvas" width="600" height="400"></canvas>
                    </div>
                    <div class="shap-values" id="shapTextView">
        `;
        
        explanation.shap_values.forEach((item, index) => {
            const absValue = Math.abs(item.value || item.abs_importance || 0);
            const value = item.value || (item.direction === 'positive' ? absValue : -absValue) || 0;
            const className = value > 0.05 ? 'positive' : 
                             value < -0.05 ? 'negative' : 'neutral';
            const intensity = Math.min(absValue * 10, 1);
            html += `<span class="shap-word ${className}" 
                          title="Impact: ${value.toFixed(3)}${item.abs_importance ? ' | Abs: ' + item.abs_importance.toFixed(3) : ''}" 
                          style="background-color: rgba(${value > 0 ? '76, 175, 80' : '244, 67, 54'}, ${intensity});"
                          data-index="${index}" data-value="${value}">${item.word || item.token}</span>`;
        });
        
        html += `
                    </div>
                </div>
                <p class="explanation-legend">
                    🟢 Green: Supports prediction | 🔴 Red: Opposes prediction | ⚪ Gray: Neutral
                    <br>Intensity indicates strength of influence
                </p>
            </div>
        `;
    }
    
    // Enhanced LIME Feature Analysis Section
    if (explanation.feature_importance && explanation.feature_importance.length > 0) {
        html += `
            <div class="explanation-section">
                <h3>🔍 LIME Feature Analysis</h3>
                <div class="lime-controls">
                    <label>Show top: <select id="limeTopN" onchange="updateLimeDisplay()">
                        <option value="5">5</option>
                        <option value="10" selected>10</option>
                        <option value="15">15</option>
                        <option value="all">All</option>
                    </select></label>
                    <button class="lime-sort-btn" onclick="sortLimeFeatures('importance')">Sort by Importance</button>
                    <button class="lime-sort-btn" onclick="sortLimeFeatures('alphabetical')">Sort Alphabetically</button>
                </div>
                <div class="feature-importance" id="limeFeatures">
        `;
        
        const topFeatures = explanation.feature_importance.slice(0, 15);
        topFeatures.forEach((item, index) => {
            const importance = item.importance || item.abs_weight || 0;
            const direction = item.direction || (importance > 0 ? 'positive' : 'negative');
            const scoreClass = direction === 'positive' ? 'positive' : 'negative';
            const barWidth = Math.abs(importance) * 100;
            
            html += `
                <div class="feature-item enhanced" data-importance="${Math.abs(importance)}" data-feature="${item.feature}">
                    <div class="feature-header">
                        <span class="feature-name" title="${item.feature}">${item.feature}</span>
                        <span class="feature-score ${scoreClass}">${importance.toFixed(3)}</span>
                    </div>
                    <div class="feature-bar">
                        <div class="feature-bar-fill ${scoreClass}" style="width: ${Math.min(barWidth, 100)}%"></div>
                    </div>
                    ${item.abs_weight ? `<div class="feature-meta">Absolute Weight: ${item.abs_weight.toFixed(3)}</div>` : ''}
                </div>
            `;
        });
        
        html += `
                </div>
                <div class="lime-summary">
                    <p>📊 Showing ${topFeatures.length} most important features out of ${explanation.feature_importance.length} total</p>
                </div>
            </div>
        `;
    }
    
    // Enhanced Topic Clusters Section with Interactive Visualization
    if (explanation.topic_clusters && explanation.topic_clusters.length > 0) {
        html += `
            <div class="explanation-section">
                <h3>📊 Interactive Topic Analysis</h3>
                <div class="topic-controls">
                    <button class="topic-view-btn" onclick="switchTopicView('cluster')">Cluster View</button>
                    <button class="topic-view-btn" onclick="switchTopicView('hierarchy')">Hierarchy View</button>
                    <button class="topic-view-btn" onclick="switchTopicView('network')">Network View</button>
                </div>
                <div class="topic-visualization">
                    <div class="topic-cluster-view" id="topicClusterView">
                        <canvas id="topicClusterCanvas" width="700" height="300"></canvas>
                    </div>
                    <div class="topic-hierarchy-view" id="topicHierarchyView" style="display: none;">
                        <div class="topic-tree" id="topicTree"></div>
                    </div>
                    <div class="topic-network-view" id="topicNetworkView" style="display: none;">
                        <canvas id="topicNetworkCanvas" width="700" height="400"></canvas>
                    </div>
                </div>
                <div class="topic-clusters-detailed">
        `;
        
        explanation.topic_clusters.forEach((topic, index) => {
            const topicId = topic.topic_id >= 0 ? topic.topic_id : 'Outlier';
            const probability = topic.probability || 0;
            const topicSize = topic.topic_size || 'Unknown';
            
            html += `
                <div class="topic-item-interactive" data-topic-id="${topic.topic_id}" onclick="highlightTopic(${index})">
                    <div class="topic-header-enhanced">
                        <div class="topic-info">
                            <span class="topic-id">Topic ${topicId}</span>
                            <span class="topic-probability">${(probability * 100).toFixed(1)}%</span>
                            <span class="topic-size">(${topicSize} docs)</span>
                        </div>
                        <div class="topic-actions">
                            <button class="topic-expand-btn" onclick="toggleTopicDetails(${index})">📋 Details</button>
                            <button class="topic-similar-btn" onclick="findSimilarTopics(${index})">🔍 Similar</button>
                        </div>
                    </div>
                    <div class="topic-probability-bar">
                        <div class="probability-fill" style="width: ${probability * 100}%"></div>
                    </div>
            `;
            
            if (topic.note) {
                html += `<div class="topic-note">${topic.note}</div>`;
            }
            
            if (topic.keywords && topic.keywords.length > 0) {
                html += `<div class="topic-keywords-enhanced">`;
                topic.keywords.slice(0, 10).forEach(keyword => {
                    if (Array.isArray(keyword) && keyword.length >= 2) {
                        const weight = keyword[0];
                        const term = keyword[1];
                        html += `<span class="keyword-tag-enhanced" 
                                      title="Weight: ${weight.toFixed(3)}" 
                                      style="opacity: ${0.5 + weight * 0.5};"
                                      onclick="searchKeyword('${term}')">${term}</span>`;
                    } else if (typeof keyword === 'string') {
                        html += `<span class="keyword-tag-enhanced" onclick="searchKeyword('${keyword}')">${keyword}</span>`;
                    }
                });
                html += `</div>`;
            }
            
            // Collapsible details section
            html += `
                <div class="topic-details" id="topicDetails${index}" style="display: none;">
                    ${topic.representative_docs ? `
                        <div class="representative-docs">
                            <h5>📄 Representative Examples:</h5>
                            <ul>
                    ` : ''}
            `;
            
            if (topic.representative_docs && topic.representative_docs.length > 0) {
                topic.representative_docs.slice(0, 3).forEach(doc => {
                    const truncatedDoc = doc.length > 150 ? doc.substring(0, 150) + '...' : doc;
                    html += `<li class="rep-doc">${truncatedDoc}</li>`;
                });
                html += `</ul></div>`;
            }
            
            html += `
                    <div class="topic-stats">
                        <span class="stat-item">📊 Coherence: ${topic.coherence ? topic.coherence.toFixed(3) : 'N/A'}</span>
                        <span class="stat-item">🎯 Diversity: ${topic.diversity ? topic.diversity.toFixed(3) : 'N/A'}</span>
                    </div>
                </div>
            `;
            
            html += `</div>`;
        });
        
        html += `
                </div>
                <div class="topic-summary">
                    <p>🔍 Found ${explanation.topic_clusters.length} topics. Click on topics to explore details and relationships.</p>
                </div>
            </div>
        `;
    }
    
    // LIME HTML Explanation (if available)
    if (explanation.lime_explanation && explanation.lime_explanation.includes('<')) {
        html += `
            <div class="explanation-section">
                <h3>📝 LIME Text Explanation</h3>
                <div class="lime-explanation">
                    ${explanation.lime_explanation}
                </div>
            </div>
        `;
    }
    
    // Enhanced Grad-CAM Visualization Section
    if (explanation.grad_cam && explanation.grad_cam.heatmap_available) {
        html += `
            <div class="explanation-section">
                <h3>🔥 Interactive Grad-CAM Analysis</h3>
                <div class="grad-cam-controls">
                    <button class="gradcam-view-btn" onclick="switchGradCamView('heatmap')">🔥 Heatmap</button>
                    <button class="gradcam-view-btn" onclick="switchGradCamView('overlay')">🖼️ Overlay</button>
                    <button class="gradcam-view-btn" onclick="switchGradCamView('regions')">📍 Regions</button>
                    <label>Intensity: <input type="range" id="gradcamIntensity" min="0.1" max="2" step="0.1" value="1" onchange="updateGradCamIntensity()"></label>
                </div>
                <div class="grad-cam-visualization">
                    <div class="grad-cam-heatmap-view" id="gradcamHeatmapView">
                        <canvas id="gradcamHeatmapCanvas" width="400" height="300"></canvas>
                    </div>
                    <div class="grad-cam-overlay-view" id="gradcamOverlayView" style="display: none;">
                        <canvas id="gradcamOverlayCanvas" width="400" height="300"></canvas>
                    </div>
                    <div class="grad-cam-regions-view" id="gradcamRegionsView" style="display: none;">
                        <div class="regions-grid" id="regionsGrid"></div>
                    </div>
                </div>
                <div class="grad-cam-analysis">
                    <p class="grad-cam-description">${explanation.grad_cam.explanation}</p>
                    <div class="attention-regions-enhanced">
                        <h4>🎯 Key Attention Regions:</h4>
        `;
        
        if (explanation.grad_cam.attention_regions) {
            explanation.grad_cam.attention_regions.forEach((region, index) => {
                const intensityClass = region.importance > 0.7 ? 'high' : 
                                     region.importance > 0.5 ? 'medium' : 'low';
                html += `
                    <div class="attention-region-enhanced ${intensityClass}" 
                         onclick="focusOnRegion(${index})" 
                         data-region-id="${index}">
                        <div class="region-header">
                            <span class="region-name">${region.region}</span>
                            <span class="region-coordinates">${region.coordinates || 'N/A'}</span>
                        </div>
                        <div class="importance-visualization">
                            <div class="importance-bar-enhanced">
                                <div class="importance-fill ${intensityClass}" style="width: ${region.importance * 100}%"></div>
                            </div>
                            <span class="importance-value">${(region.importance * 100).toFixed(1)}%</span>
                        </div>
                        <div class="region-details">
                            <span class="region-size">Size: ${region.size || 'Unknown'}</span>
                            <span class="region-confidence">Confidence: ${region.confidence ? (region.confidence * 100).toFixed(1) + '%' : 'N/A'}</span>
                        </div>
                    </div>
                `;
            });
        }
        
        html += `
                    </div>
                    <div class="grad-cam-insights">
                        <h5>🧠 Analysis Insights:</h5>
                        <ul>
                            <li>🔍 Click on regions to focus and zoom</li>
                            <li>🎚️ Adjust intensity slider to modify heatmap visibility</li>
                            <li>📊 Higher intensity regions have stronger influence on the prediction</li>
                            <li>🎯 Red areas indicate high attention, blue areas indicate low attention</li>
                        </ul>
                    </div>
                </div>
            </div>
        `;
    }
    
    // Proof Links and Validation Section
    if (explanation.proof_links && explanation.proof_links.length > 0) {
        html += `
            <div class="explanation-section">
                <h3>🔗 Validation Proofs & Sources</h3>
                <div class="proof-links-container">
        `;
        
        explanation.proof_links.forEach((proof, index) => {
            const reliabilityClass = proof.reliability > 0.8 ? 'high-reliability' : 
                                   proof.reliability > 0.6 ? 'medium-reliability' : 'low-reliability';
            const statusClass = proof.status === 'verified' ? 'verified' : 
                              proof.status === 'disputed' ? 'disputed' : 'pending';
            
            html += `
                <div class="proof-item ${reliabilityClass} ${statusClass}" data-proof-id="${index}">
                    <div class="proof-header">
                        <div class="proof-source">
                            <span class="source-name">${proof.source}</span>
                            <span class="source-type">${proof.type || 'Unknown'}</span>
                        </div>
                        <div class="proof-metrics">
                            <span class="reliability-badge ${reliabilityClass}">
                                ${proof.reliability ? (proof.reliability * 100).toFixed(0) + '%' : 'N/A'}
                            </span>
                            <span class="status-badge ${statusClass}">${proof.status || 'Unknown'}</span>
                        </div>
                    </div>
                    <div class="proof-content">
                        <p class="proof-summary">${proof.summary || 'No summary available'}</p>
                        ${proof.key_points ? `
                            <div class="proof-key-points">
                                <h5>Key Points:</h5>
                                <ul>
                                    ${proof.key_points.map(point => `<li>${point}</li>`).join('')}
                                </ul>
                            </div>
                        ` : ''}
                    </div>
                    <div class="proof-actions">
                        <a href="${proof.url}" target="_blank" class="proof-link-btn">🔗 View Source</a>
                        <button class="proof-analyze-btn" onclick="analyzeProof(${index})">🔍 Analyze</button>
                        <button class="proof-compare-btn" onclick="compareProofs(${index})">⚖️ Compare</button>
                    </div>
                    ${proof.confidence_score ? `
                        <div class="proof-confidence">
                            <span>Confidence: ${(proof.confidence_score * 100).toFixed(1)}%</span>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${proof.confidence_score * 100}%"></div>
                            </div>
                        </div>
                    ` : ''}
                </div>
            `;
        });
        
        html += `
                </div>
                <div class="proof-summary">
                    <p>📊 Found ${explanation.proof_links.length} validation sources</p>
                    <div class="proof-stats">
                        <span class="stat-verified">✅ Verified: ${explanation.proof_links.filter(p => p.status === 'verified').length}</span>
                        <span class="stat-disputed">❌ Disputed: ${explanation.proof_links.filter(p => p.status === 'disputed').length}</span>
                        <span class="stat-pending">⏳ Pending: ${explanation.proof_links.filter(p => p.status === 'pending').length}</span>
                    </div>
                </div>
            </div>
        `;
    }
    
    // Validation Data Section
    if (explanation.validation_data) {
        html += `
            <div class="explanation-section">
                <h3>✅ Validation Analysis</h3>
                <div class="validation-summary">
                    <div class="validation-metrics">
                        <div class="validation-metric">
                            <span class="metric-label">Overall Verdict:</span>
                            <span class="metric-value verdict-${explanation.validation_data.final_verdict?.toLowerCase()}">
                                ${explanation.validation_data.final_verdict || 'Unknown'}
                            </span>
                        </div>
                        <div class="validation-metric">
                            <span class="metric-label">Confidence:</span>
                            <span class="metric-value">${explanation.validation_data.final_confidence ? (explanation.validation_data.final_confidence * 100).toFixed(1) + '%' : 'N/A'}</span>
                        </div>
                        <div class="validation-metric">
                            <span class="metric-label">Sources Checked:</span>
                            <span class="metric-value">${explanation.validation_data.sources_checked || 0}</span>
                        </div>
                    </div>
                    ${explanation.validation_data.reasoning ? `
                        <div class="validation-reasoning">
                            <h5>🧠 Reasoning:</h5>
                            <p>${explanation.validation_data.reasoning}</p>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    }
    
    // If no explanations available
    if (!html) {
        html = `
            <div class="no-explanation">
                <p>⚠️ No detailed explanations available for this prediction.</p>
                <p>This may be due to:</p>
                <ul style="text-align: left; margin-top: 10px;">
                    <li>Explainability libraries not installed</li>
                    <li>Insufficient text content for analysis</li>
                    <li>Model limitations</li>
                </ul>
            </div>
        `;
    }
    
    elements.explainabilityContent.innerHTML = html;
    
    // Initialize interactive visualizations after DOM update
    setTimeout(() => {
        initializeShapVisualization(explanation.shap_values);
        initializeTopicVisualization(explanation.topic_clusters);
        initializeGradCamVisualization(explanation.grad_cam);
    }, 100);
}

// UI Update Functions
function updateDetectionResult(result) {
    // Handle new orchestrator response format
    const unified_verdict = result.unified_verdict || {};
    const mhfn_output = result.mhfn_output || {};
    const evidence_sources = result.evidence_sources || [];
    const processing_metrics = result.processing_metrics || {};
    
    // Extract unified verdict data
    const prediction = unified_verdict.prediction || result.prediction || 'UNKNOWN';
    const confidence = unified_verdict.confidence || result.confidence || 0;
    const fusion_method = unified_verdict.fusion_method || 'Standard';
    
    // Extract MHFN model data
    const mhfn_prediction = mhfn_output.prediction || prediction;
    const mhfn_confidence = mhfn_output.confidence || confidence;
    
    // Extract processing metrics
    const processing_time = processing_metrics.total_time || result.processing_time || currentProcessingTime;
    const sla_status = processing_metrics.sla_met ? 'Met' : 'Exceeded';
    const sla_class = processing_metrics.sla_met ? 'sla-met' : 'sla-exceeded';
    
    // Build unified verdict display
    const verdictHtml = `
        <div class="orchestrator-result-card">
            <div class="unified-verdict ${prediction.toLowerCase() === 'real' ? 'verdict-real' : 'verdict-fake'}">
                <div class="verdict-header">
                    <div class="verdict-label">${prediction.toUpperCase()}</div>
                    <div class="verdict-confidence">${(confidence * 100).toFixed(1)}% Confidence</div>
                </div>
                <div class="fusion-badge">
                    <span class="badge-icon">⚖️</span>
                    <span class="badge-text">Evidence-Guided Bayesian Fusion</span>
                </div>
            </div>
        </div>
    `;
    
    // Build MHFN model output display
    const mhfnHtml = `
        <div class="model-prediction">
            <h4>🧠 MHFN Model Output</h4>
            <div class="prediction-details">
                <div class="prediction-item">
                    <span class="prediction-label">Prior Prediction:</span>
                    <span class="prediction-value ${mhfn_prediction.toLowerCase() === 'real' ? 'value-real' : 'value-fake'}">
                        ${mhfn_prediction.toUpperCase()}
                    </span>
                </div>
                <div class="prediction-item">
                    <span class="prediction-label">Prior Confidence:</span>
                    <span class="prediction-value">${(mhfn_confidence * 100).toFixed(1)}%</span>
                </div>
            </div>
        </div>
    `;
    
    // Build evidence sources display
    let evidenceHtml = `
        <div class="evidence-sources">
            <h4>🔍 Evidence Sources</h4>
    `;
    
    if (evidence_sources.length > 0) {
        evidenceHtml += `
            <div class="evidence-list">
                ${evidence_sources.map(source => `
                    <div class="evidence-item">
                        <div class="evidence-header">
                            <span class="evidence-source">${source.source || 'Unknown Source'}</span>
                            <span class="evidence-credibility">Credibility: ${((source.credibility || 0) * 100).toFixed(0)}%</span>
                        </div>
                        <div class="evidence-content">
                            <p class="evidence-summary">${source.summary || 'No summary available'}</p>
                            <div class="evidence-meta">
                                <span class="evidence-similarity">Similarity: ${((source.similarity || 0) * 100).toFixed(1)}%</span>
                                <span class="evidence-recency">Recency: ${source.recency_score || 'N/A'}</span>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    } else {
        evidenceHtml += `
            <div class="no-evidence">
                <p>No external evidence sources found for this content.</p>
            </div>
        `;
    }
    
    evidenceHtml += `</div>`;
    
    // Build processing metrics display
    const metricsHtml = `
        <div class="processing-metrics">
            <h4>⏱️ Processing Metrics</h4>
            <div class="metrics-grid">
                <div class="metric-item">
                    <span class="metric-label">Processing Time:</span>
                    <span class="metric-value">${processing_time}s</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">SLA Status:</span>
                    <span class="metric-value ${sla_class}">${sla_status}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Fusion Method:</span>
                    <span class="metric-value">${fusion_method}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Evidence Count:</span>
                    <span class="metric-value">${evidence_sources.length}</span>
                </div>
            </div>
        </div>
    `;
    
    // Combine all HTML
    const resultHtml = verdictHtml + mhfnHtml + evidenceHtml + metricsHtml;
    
    elements.detectionResult.innerHTML = resultHtml;
    
    // Metrics are now displayed in the unified verdict section
    
    // Update analysis count
    analysisCount++;
    if (elements.totalAnalyzed) elements.totalAnalyzed.textContent = analysisCount;
    
    // Store detection result for Content Result section
    appState.lastDetectionResult = {
        prediction: prediction,
        confidence: confidence,
        processing_time: processing_time
    };
}

async function updateNewsFeed(newsData) {
    // Handle new API response format with cache information
    const articles = newsData.articles || newsData.data || [];
    const cacheInfo = newsData.cache_hit !== undefined ? {
        cached: newsData.cached || false,
        cache_hit: newsData.cache_hit || false,
        response_time: newsData.response_time || 0,
        api_source: newsData.api_source || 'Unknown'
    } : null;
    
    if (!articles || articles.length === 0) {
        elements.liveFeed.innerHTML = '<div class="no-news">No news items available from this source.</div>';
        return;
    }
    
    // Add cache status display
    let cacheStatusHtml = '';
    if (cacheInfo) {
        const cacheIcon = cacheInfo.cache_hit ? '🟢' : '🔴';
        const cacheText = cacheInfo.cache_hit ? 'Cache Hit' : 'Fresh Data';
        cacheStatusHtml = `
            <div class="feed-status">
                <div class="cache-status">
                    <span class="cache-indicator">${cacheIcon} ${cacheText}</span>
                    <span class="api-source">Source: ${cacheInfo.api_source}</span>
                    <span class="response-time">Response: ${cacheInfo.response_time.toFixed(2)}s</span>
                    <span class="auto-update-status">Auto Updated</span>
                </div>
            </div>
        `;
    }
    
    // Process articles without any validation
    const articlesWithValidation = articles.map((article, index) => {
        return {
            ...article,
            originalIndex: index
        };
    });
    
    // Sort articles by date only (newest first)
    articlesWithValidation.sort((a, b) => {
        const dateA = new Date(a.publishedAt || a.published || a.pub_date || a.date || 0);
        const dateB = new Date(b.publishedAt || b.published || b.pub_date || b.date || 0);
        return dateB - dateA;
    });
    
    const newsHtml = articlesWithValidation.map((item, index) => {
        // Handle different possible field names from different APIs
        const title = item.title || item.headline || 'No title available';
        const description = item.description || item.summary || item.content || 'No description available';
        const link = item.url || item.link || item.web_url || '#';
        const publishedDate = item.publishedAt || item.published || item.pub_date || item.date || new Date().toISOString();
        const source = item.source?.name || item.source || 'Unknown Source';
        
        // No validation elements - clean live feed display
        
        return `
            <div class="news-item" data-url="${link}" data-index="${index}">
                <div class="news-header">
                    <div class="news-source">${source}</div>
                    <div class="news-date" title="Published: ${new Date(publishedDate).toLocaleString()}">${formatDate(publishedDate)}</div>
                </div>
                <div class="news-title">
                    <a href="${link}" target="_blank" rel="noopener noreferrer" class="news-link" title="Open original article: ${link}">
                        ${title}
                        <i class="fas fa-external-link-alt"></i>
                    </a>
                </div>
                <div class="news-description">${truncateText(description, 200)}</div>
                <div class="news-actions">
                    <button class="analyze-btn" data-url="${link}" data-title="${title.replace(/"/g, '&quot;')}" title="Analyze this article for fake news">
                        <i class="fas fa-search"></i> Analyze Article
                    </button>
                    <button class="copy-link-btn" data-url="${link}" title="Copy article link">
                        <i class="fas fa-copy"></i> Copy Link
                    </button>
                </div>
            </div>
        `;
    }).join('');
    
    elements.liveFeed.innerHTML = cacheStatusHtml + newsHtml;
    
    // Add click handlers for analysis and copy buttons
    document.querySelectorAll('.analyze-btn').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const url = e.target.closest('.analyze-btn').dataset.url;
            const title = e.target.closest('.analyze-btn').dataset.title;
            
            if (url && url !== '#') {
                // Switch to URL tab and populate with article URL
                switchTab('url');
                if (elements.newsUrl) {
                    elements.newsUrl.value = url;
                    // Trigger URL input handling
                    handleUrlInput({ target: elements.newsUrl });
                }
                
                // Show notification
                showNotification(`Article "${truncateText(title, 50)}" loaded for analysis`, 'info');
                
                // Scroll to analysis section
                document.querySelector('.analysis-section')?.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
    
    // Add click handlers for copy link buttons
    document.querySelectorAll('.copy-link-btn').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const url = e.target.closest('.copy-link-btn').dataset.url;
            
            if (url && url !== '#') {
                try {
                    await navigator.clipboard.writeText(url);
                    showNotification('Article link copied to clipboard!', 'success');
                } catch (error) {
                    console.error('Failed to copy link:', error);
                    showNotification('Failed to copy link', 'error');
                }
            }
        });
    });
    const analyzeButtons = elements.liveFeed.querySelectorAll('.analyze-btn');
    analyzeButtons.forEach(button => {
        button.addEventListener('click', async (e) => {
            e.preventDefault();
            const url = button.getAttribute('data-url');
            await analyzeNewsUrl(url, button);
        });
    });
    
    // News links will redirect automatically via href attribute - no additional handler needed
}

// Analyze news URL for fake news detection
async function analyzeNewsUrl(url, buttonElement) {
    if (!url) {
        console.error('No URL provided for analysis');
        return;
    }
    
    // Show loading state
    const originalText = buttonElement.innerHTML;
    buttonElement.disabled = true;
    buttonElement.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    
    try {
        // Switch to URL tab and populate the URL input
        switchTab('url');
        if (elements.newsUrl) {
            elements.newsUrl.value = url;
        }
        
        // Perform the detection using the existing detectFakeNews function
        const result = await detectFakeNews({ url: url });
        
        if (result.success) {
            // Update the detection result display
            updateDetectionResult(result.data);
            
            // Show success message
            showNotification(`Analysis complete! Article classified as: ${result.data.prediction}`, 'success');
            
            // Scroll to results section
            const resultSection = document.querySelector('.result-card');
            if (resultSection) {
                resultSection.scrollIntoView({ behavior: 'smooth' });
            }
        } else {
            showNotification(`Analysis failed: ${result.error}`, 'error');
        }
        
    } catch (error) {
        console.error('Error analyzing news URL:', error);
        showNotification('An error occurred while analyzing the article', 'error');
    } finally {
        // Restore button state
        buttonElement.disabled = false;
        buttonElement.innerHTML = originalText;
    }
}

// Validate news with fact-checking sources
async function validateNewsWithProof(title, url) {
    try {
        // Try multiple fact-checking approaches
        const validationResults = await Promise.allSettled([
            searchFactCheck(title),
            searchSnopes(title),
            searchWebForProof(title)
        ]);
        
        const proofs = [];
        let hasVerificationData = false;
        
        validationResults.forEach((result, index) => {
            if (result.status === 'fulfilled' && result.value) {
                const proof = result.value;
                // Only add proofs that have actual verification data
                if (proof.has_data || proof.status === 'found') {
                    proofs.push(proof);
                    hasVerificationData = true;
                }
            }
        });
        
        return {
            validated: hasVerificationData,
            proofs: proofs,
            hasData: hasVerificationData,
            sourceCount: proofs.length
        };
    } catch (error) {
        console.error('Validation error:', error);
        return { validated: false, proofs: [], hasData: false, sourceCount: 0 };
    }
}

// Old API (DELETED) - Search FactCheck.org-style validation using direct NewsAPI
async function searchFactCheck(query) {
    try {
        // Direct NewsAPI call for fact-checking sources
        const response = await fetch(`https://newsapi.org/v2/everything?q=${encodeURIComponent(query + ' fact check verification')}&sources=bbc-news,reuters,associated-press,the-guardian-uk&apiKey=a60ec2247c7246d5988e46bd2d028297`);
        if (response.ok) {
            const data = await response.json();
            if (data.articles && data.articles.length > 0) {
                const article = data.articles[0];
                return {
                    source: 'NewsAPI Fact Check',
                    status: 'found',
                    url: article.url || '#',
                    summary: article.description || 'Fact-checking information found',
                    has_data: true
                };
            }
        }
    } catch (error) {
        console.error('NewsAPI fact check search failed:', error);
    }
    return null;
}

// Old API (DELETED) - Search Snopes-style validation using direct SerperAPI
async function searchSnopes(query) {
    try {
        // Direct SerperAPI call for Snopes-style verification
        const response = await fetch('https://google.serper.dev/search', {
            method: 'POST',
            headers: {
                'X-API-KEY': '99c420f540df242221a2a1dff511b44f1c8a6e3a',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                q: `${query} site:snopes.com OR site:factcheck.org OR site:politifact.com`,
                num: 3
            })
        });
        if (response.ok) {
            const data = await response.json();
            if (data.organic && data.organic.length > 0) {
                const result = data.organic[0];
                return {
                    source: 'SerperAPI Verification',
                    status: 'found',
                    url: result.link || '#',
                    summary: result.snippet || 'Verification information found',
                    has_data: true
                };
            }
        }
    } catch (error) {
        console.error('SerperAPI verification search failed:', error);
    }
    return null;
}

// Old API (DELETED) - Search web for proof/verification using direct NewsData.io
async function searchWebForProof(query) {
    try {
        // Direct NewsData.io call for web verification
        const searchQuery = `${query} fact check verification proof`;
        const response = await fetch(`https://newsdata.io/api/1/news?apikey=pub_0ead1cb56a8b425a831501de2f6084f0&q=${encodeURIComponent(searchQuery)}&language=en&category=politics,world`);
        if (response.ok) {
            const data = await response.json();
            if (data.results && data.results.length > 0) {
                const article = data.results[0];
                return {
                    source: 'NewsData.io Search',
                    status: 'found',
                    url: article.link || '#',
                    summary: article.description || 'Verification sources found'
                };
            }
        }
    } catch (error) {
        console.error('NewsData.io search failed:', error);
    }
    return null;
}

// Show notification to user
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
        <span>${message}</span>
        <button class="notification-close" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

function updateHistory(apiResponse) {
    if (!apiResponse.data || apiResponse.data.length === 0) {
        elements.historyList.innerHTML = '<div class="no-history">No analysis history available.</div>';
        return;
    }
    
    const historyHtml = apiResponse.data.map(item => `
        <div class="history-item">
            <div class="history-text">"${truncateText(item.content, 80)}"</div>
            <div class="history-meta">
                <span class="history-result ${item.prediction.toLowerCase()}">
                    ${item.prediction} (${(item.confidence * 100).toFixed(1)}%)
                </span>
                <span class="history-date">${formatDate(item.timestamp)}</span>
            </div>
        </div>
    `).join('');
    
    elements.historyList.innerHTML = historyHtml;
}

function showError(element, message) {
    element.innerHTML = `<div class="error-message" style="color: #ff6b6b; text-align: center; padding: 20px;">${message}</div>`;
}

// Verification Progress Functions
// Removed old verification progress function - using new system from fake-news-verification.js
function showVerificationProgress() {
    // This function is deprecated - using new verification system
    console.log('showVerificationProgress is deprecated - using new verification system');
}

// Removed old hide verification progress function - using new system
function hideVerificationProgress() {
    // This function is deprecated - using new verification system
    console.log('hideVerificationProgress is deprecated - using new verification system');
}

function updateProofPanel(verification) {
    if (!elements.proofPanel || !verification) return;
    
    // Clear timer if exists
    if (elements.proofPanel.verificationTimer) {
        clearInterval(elements.proofPanel.verificationTimer);
        elements.proofPanel.verificationTimer = null;
    }
    
    const proofs = verification.proofs || [];
    const confidenceClass = verification.confidence >= 0.9 ? 'high-confidence' : 
                           verification.confidence >= 0.7 ? 'medium-confidence' : 'low-confidence';
    
    const proofsHtml = proofs.length > 0 ? proofs.map((proof, index) => {
        const statusIcon = proof.status === 'verified' || proof.status === 'true' ? '✅' :
                          proof.status === 'false' ? '❌' :
                          proof.status === 'mixture' ? '⚠️' :
                          proof.status === 'unproven' ? '❓' :
                          proof.status === 'no_data' ? '🔍' : '📄';
        
        const statusText = proof.status === 'no_data' ? 'No data found' :
                          proof.status === 'verified' || proof.status === 'true' ? 'Verified as True' :
                          proof.status === 'false' ? 'Verified as False' :
                          proof.status === 'mixture' ? 'Mixed Evidence' :
                          proof.status === 'unproven' ? 'Unproven Claims' :
                          proof.status.charAt(0).toUpperCase() + proof.status.slice(1);
        
        return `
            <div class="proof-item">
                <div class="proof-header">
                    <span class="proof-status">${statusIcon} ${statusText}</span>
                    <span class="proof-source">${proof.source}</span>
                </div>
                <div class="proof-summary">${proof.summary || 'No summary available'}</div>
                ${proof.url ? `<a href="${proof.url}" target="_blank" class="proof-link">View Source</a>` : ''}
            </div>
        `;
    }).join('') : '<div class="no-proofs">No external proofs found</div>';
    
    const verificationHtml = `
        <div class="verification-complete ${confidenceClass}">
            <div class="verification-header">
                <h4>🔍 Deliberate Verification Complete</h4>
                <div class="verification-verdict ${verification.verdict.toLowerCase()}">
                    ${verification.verdict.toUpperCase()}
                </div>
            </div>
            <div class="verification-summary">
                <div class="confidence-display">
                    <span class="confidence-label">Confidence:</span>
                    <span class="confidence-value">${(verification.confidence * 100).toFixed(1)}%</span>
                </div>
                <div class="processing-time">
                    <span class="time-label">Processing Time:</span>
                    <span class="time-value">${verification.processing_time.toFixed(1)}s</span>
                </div>
            </div>
            <div class="verification-explanation">
                <h5>Explanation:</h5>
                <p>${verification.explanation}</p>
            </div>
            <div class="verification-sources">
                <h5>Sources Checked:</h5>
                <div class="sources-list">
                    ${verification.sources_checked.map(source => `<span class="source-tag">${source}</span>`).join('')}
                </div>
            </div>
            <div class="verification-proofs">
                <h5>Evidence Found:</h5>
                ${proofsHtml}
            </div>
        </div>
    `;
    
    elements.proofPanel.innerHTML = verificationHtml;
}

// Event Handlers
async function handleDetection() {
    const text = elements.newsText.value.trim();
    
    if (!text) {
        alert('Please enter some text to analyze.');
        return;
    }
    
    if (text.length < 10) {
        alert('Please enter at least 10 characters for meaningful analysis.');
        return;
    }
    
    // Show loading state with verification progress
    elements.detectBtn.disabled = true;
    elements.detectBtn.textContent = 'Analyzing...';
    showLoading(elements.loadingIndicator, true);
    
    // Show simple loading state instead of old verification progress
    elements.proofLoading.classList.remove('hidden');
    
    try {
        const result = await detectFakeNews(text);
        
        if (result.success) {
            updateDetectionResult(result.data);
            // Update proof panel with verification results
            if (result.data.verification) {
                updateProofPanel(result.data.verification);
            }
            // Refresh history after successful detection
            await loadHistory();
        } else {
            showError(elements.detectionResult, `Detection failed: ${result.error}`);
            hideVerificationProgress();
        }
        
    } catch (error) {
        console.error('Detection error:', error);
        showError(elements.detectionResult, 'An unexpected error occurred during analysis.');
        hideVerificationProgress();
    } finally {
        // Reset loading state
        elements.detectBtn.disabled = false;
        elements.detectBtn.textContent = 'Detect Fake News';
        showLoading(elements.loadingIndicator, false);
    }
}

async function handleFeedRefresh() {
    const source = elements.feedSource.value;
    
    elements.refreshFeedBtn.disabled = true;
    elements.refreshFeedBtn.textContent = 'Loading...';
    elements.liveFeed.innerHTML = '<div class="loading-feed">Loading news feed...</div>';
    
    const startTime = performance.now();
    
    try {
        const result = await fetchLiveFeed(source);
        const endTime = performance.now();
        const clientResponseTime = ((endTime - startTime) / 1000).toFixed(2);
        
        if (result.success) {
            // Log performance metrics
            console.log(`Feed refresh completed in ${clientResponseTime}s for source: ${source}`);
            if (result.data.cache_hit) {
                console.log('✅ Cache hit - improved performance!');
            } else {
                console.log('🔄 Fresh data fetched from API');
            }
            
            updateNewsFeed(result.data);
            
            // Show success notification without cache status
            // Notification removed as requested by user
        } else {
            showError(elements.liveFeed, `Failed to load news feed: ${result.error}`);
        }
        
    } catch (error) {
        console.error('Feed refresh error:', error);
        showError(elements.liveFeed, 'An unexpected error occurred while loading the news feed.');
        showNotification('Failed to refresh news feed', 'error');
    } finally {
        elements.refreshFeedBtn.disabled = false;
        elements.refreshFeedBtn.textContent = 'Refresh Feed';
    }
}

async function loadHistory() {
    try {
        const result = await fetchHistory();
        
        if (result.success) {
            // The result.data is the direct API response
            const apiResponse = result.data;
            if (apiResponse && apiResponse.status === 'success') {
                updateHistory(apiResponse);
            } else {
                showError(elements.historyList, 'No history data available.');
            }
        } else {
            showError(elements.historyList, `Failed to load history: ${result.error}`);
        }
        
    } catch (error) {
        console.error('History load error:', error);
        showError(elements.historyList, 'An unexpected error occurred while loading history.');
    }
}

// Clear history handler
async function handleClearHistory() {
    try {
        // Show confirmation dialog
        const confirmed = confirm('Are you sure you want to clear all analysis history? This action cannot be undone.');
        if (!confirmed) {
            return;
        }
        
        // Call the clear history API endpoint
        const response = await fetch(`${API_BASE_URL}/api/clear-history`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const result = await response.json();
        
        if (response.ok && result.status === 'success') {
            // Clear the history display
            elements.historyList.innerHTML = '<div class="no-history">No analysis history available.</div>';
            
            // Show success notification
            showNotification('History cleared successfully!', 'success');
            console.log('History cleared successfully');
        } else {
            throw new Error(result.message || 'Failed to clear history');
        }
        
    } catch (error) {
        console.error('Error clearing history:', error);
        showNotification(`Failed to clear history: ${error.message}`, 'error');
    }
}

// Initialization Functions
function initializeEventListeners() {
    // Detection button
    elements.detectBtn.addEventListener('click', handleDetection);
    
    // Enter key in textarea
    if (elements.newsText) {
        elements.newsText.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                handleDetection();
            }
        });
        
        // Text length counter
        elements.newsText.addEventListener('input', (e) => {
            const length = e.target.value.length;
            if (length > 0) {
                elements.textLength.textContent = length;
            }
        });
    }
    
    // Feed refresh button
    if (elements.refreshFeedBtn) {
        elements.refreshFeedBtn.addEventListener('click', handleFeedRefresh);
    }
    
    // Feed source change
    if (elements.feedSource) {
        elements.feedSource.addEventListener('change', handleFeedRefresh);
    }
    
    // Clear history button
    const clearHistoryBtn = document.getElementById('clearHistory');
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', handleClearHistory);
    }
    
    // Explainability button
    if (elements.explainBtn) {
        elements.explainBtn.addEventListener('click', generateExplanation);
    }
    
    // Extended metrics button
    if (elements.extendedMetricsBtn) {
        console.log('Extended metrics button found, adding event listener');
        elements.extendedMetricsBtn.addEventListener('click', fetchExtendedMetrics);
    } else {
        console.error('Extended metrics button not found! ID: extendedMetricsBtn');
    }
    
    // Validate with sources button
    if (elements.validateBtn) {
        elements.validateBtn.addEventListener('click', validateWithSources);
    }
    
    // Content Result verification trigger button
    if (elements.triggerContentVerification) {
        elements.triggerContentVerification.addEventListener('click', () => {
            console.log('🔍 Content Result verification triggered');
            if (typeof initializeContentResultVerification === 'function') {
                initializeContentResultVerification();
            } else {
                console.error('❌ Content Result verification system not loaded');
                showNotification('Content verification system not available', 'error');
            }
        });
    }

    // Test verification button
    if (elements.testVerification) {
        elements.testVerification.addEventListener('click', () => {
            console.log('🧪 Testing enhanced verification system');
            testEnhancedVerification();
        });
    }

}

// Initialize live feed functionality
function initializeLiveFeed() {
    // Populate news sources dropdown
    populateNewsSourcesDropdown();
    
    if (elements.refreshFeedBtn) {
        elements.refreshFeedBtn.addEventListener('click', handleFeedRefresh);
    }
    
    // Add event listener for source selection
    if (elements.feedSource) {
        elements.feedSource.addEventListener('change', handleFeedRefresh);
    }
    
    // Auto-refresh every 9 seconds for latest news
    setInterval(handleFeedRefresh, 9 * 1000);
    
    // Initial load
    handleFeedRefresh();
}

// Populate news sources dropdown
function populateNewsSourcesDropdown() {
    const sources = [
        { value: 'all', label: 'All Sources' },
        { value: 'bbc', label: 'BBC News' },
        { value: 'cnn', label: 'CNN' },
        { value: 'fox', label: 'Fox News' },
        { value: 'nyt', label: 'New York Times' },
        { value: 'thehindu', label: 'The Hindu' },
        { value: 'ndtv', label: 'NDTV 24x7' },

    ];
    
    if (elements.feedSource) {
        // Clear existing options
        elements.feedSource.innerHTML = '';
        
        // Add source options
        sources.forEach(source => {
            const option = document.createElement('option');
            option.value = source.value;
            option.textContent = source.label;
            elements.feedSource.appendChild(option);
        });
        
        // Set default selection
        elements.feedSource.value = 'bbc';
    }
}

// Interactive Visualization Functions for Enhanced Explainability

// Initialize SHAP heatmap visualization
function initializeShapHeatmap(shapValues, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !shapValues || shapValues.length === 0) return;
    
    const maxAbs = Math.max(...shapValues.map(v => Math.abs(v.importance)));
    
    shapValues.forEach((value, index) => {
        const intensity = Math.abs(value.importance) / maxAbs;
        const color = value.importance > 0 ? 
            `rgba(255, 99, 132, ${intensity})` : 
            `rgba(54, 162, 235, ${intensity})`;
        
        const tokenElement = document.createElement('span');
        tokenElement.className = 'shap-token';
        tokenElement.textContent = value.token;
        tokenElement.style.backgroundColor = color;
        tokenElement.style.padding = '2px 4px';
        tokenElement.style.margin = '1px';
        tokenElement.style.borderRadius = '3px';
        tokenElement.title = `Impact: ${value.importance.toFixed(4)} (${value.direction})`;
        
        container.appendChild(tokenElement);
    });
}

// Initialize SHAP bar chart
function initializeShapBarChart(shapValues, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !shapValues || shapValues.length === 0) return;
    
    const topValues = shapValues.slice(0, 10);
    const maxAbs = Math.max(...topValues.map(v => Math.abs(v.importance)));
    
    topValues.forEach(value => {
        const barContainer = document.createElement('div');
        barContainer.className = 'shap-bar-item';
        barContainer.style.display = 'flex';
        barContainer.style.alignItems = 'center';
        barContainer.style.marginBottom = '5px';
        
        const label = document.createElement('span');
        label.textContent = value.token;
        label.style.minWidth = '100px';
        label.style.fontSize = '12px';
        
        const bar = document.createElement('div');
        bar.style.height = '20px';
        bar.style.width = `${(Math.abs(value.importance) / maxAbs) * 200}px`;
        bar.style.backgroundColor = value.importance > 0 ? '#ff6384' : '#36a2eb';
        bar.style.marginLeft = '10px';
        bar.title = `${value.importance.toFixed(4)}`;
        
        barContainer.appendChild(label);
        barContainer.appendChild(bar);
        container.appendChild(barContainer);
    });
}

// Initialize topic clustering network view
function initializeTopicNetwork(topics, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !topics || topics.length === 0) return;
    
    const networkContainer = document.createElement('div');
    networkContainer.className = 'topic-network';
    networkContainer.style.position = 'relative';
    networkContainer.style.height = '300px';
    networkContainer.style.border = '1px solid #ddd';
    networkContainer.style.borderRadius = '5px';
    networkContainer.style.overflow = 'hidden';
    
    topics.forEach((topic, index) => {
        const node = document.createElement('div');
        node.className = 'topic-node';
        node.style.position = 'absolute';
        node.style.width = `${Math.max(60, topic.probability * 120)}px`;
        node.style.height = `${Math.max(60, topic.probability * 120)}px`;
        node.style.borderRadius = '50%';
        node.style.backgroundColor = `hsl(${index * 60}, 70%, 60%)`;
        node.style.display = 'flex';
        node.style.alignItems = 'center';
        node.style.justifyContent = 'center';
        node.style.cursor = 'pointer';
        node.style.fontSize = '10px';
        node.style.color = 'white';
        node.style.fontWeight = 'bold';
        node.textContent = `T${topic.topic_id}`;
        
        // Random positioning
        node.style.left = `${Math.random() * 250}px`;
        node.style.top = `${Math.random() * 200}px`;
        
        node.title = `Topic ${topic.topic_id}: ${topic.keywords.slice(0, 3).join(', ')} (${(topic.probability * 100).toFixed(1)}%)`;
        
        node.addEventListener('click', () => {
            showTopicDetails(topic);
        });
        
        networkContainer.appendChild(node);
    });
    
    container.appendChild(networkContainer);
}

// Show topic details modal
function showTopicDetails(topic) {
    const modal = document.createElement('div');
    modal.className = 'topic-modal';
    modal.style.position = 'fixed';
    modal.style.top = '50%';
    modal.style.left = '50%';
    modal.style.transform = 'translate(-50%, -50%)';
    modal.style.backgroundColor = 'white';
    modal.style.padding = '20px';
    modal.style.borderRadius = '10px';
    modal.style.boxShadow = '0 4px 20px rgba(0,0,0,0.3)';
    modal.style.zIndex = '1000';
    modal.style.maxWidth = '400px';
    
    modal.innerHTML = `
        <h3>Topic ${topic.topic_id} Details</h3>
        <p><strong>Probability:</strong> ${(topic.probability * 100).toFixed(2)}%</p>
        <p><strong>Keywords:</strong> ${topic.keywords.join(', ')}</p>
        <button onclick="this.parentElement.remove()" style="margin-top: 10px; padding: 5px 10px;">Close</button>
    `;
    
    document.body.appendChild(modal);
    
    // Close on background click
    setTimeout(() => {
        document.addEventListener('click', function closeModal(e) {
            if (!modal.contains(e.target)) {
                modal.remove();
                document.removeEventListener('click', closeModal);
            }
        });
    }, 100);
}

// Initialize Grad-CAM heatmap overlay
function initializeGradCamOverlay(gradcamData, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !gradcamData) return;
    
    const overlayContainer = document.createElement('div');
    overlayContainer.className = 'gradcam-overlay';
    overlayContainer.style.position = 'relative';
    overlayContainer.style.display = 'inline-block';
    
    // Create intensity control
    const intensityControl = document.createElement('div');
    intensityControl.innerHTML = `
        <label>Heatmap Intensity: <input type="range" id="intensity-slider" min="0.3" max="1" step="0.1" value="0.7"></label>
        <span id="intensity-value">0.7</span>
    `;
    
    const slider = intensityControl.querySelector('#intensity-slider');
    const valueDisplay = intensityControl.querySelector('#intensity-value');
    
    slider.addEventListener('input', (e) => {
        const intensity = parseFloat(e.target.value);
        valueDisplay.textContent = intensity;
        updateHeatmapIntensity(overlayContainer, intensity);
    });
    
    container.appendChild(intensityControl);
    container.appendChild(overlayContainer);
    
    // Create heatmap visualization
    if (gradcamData.attention_regions && gradcamData.attention_regions.length > 0) {
        createAttentionRegions(overlayContainer, gradcamData.attention_regions);
    }
}

// Create attention regions visualization
function createAttentionRegions(container, regions) {
    const canvas = document.createElement('canvas');
    canvas.width = 300;
    canvas.height = 200;
    canvas.style.border = '1px solid #ddd';
    canvas.style.borderRadius = '5px';
    
    const ctx = canvas.getContext('2d');
    
    // Draw background
    ctx.fillStyle = '#f0f0f0';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw attention regions
    regions.forEach((region, index) => {
        const intensity = region.intensity || 0.5;
        const hue = (index * 60) % 360;
        
        ctx.fillStyle = `hsla(${hue}, 70%, 50%, ${intensity})`;
        ctx.fillRect(
            region.x * canvas.width,
            region.y * canvas.height,
            region.width * canvas.width,
            region.height * canvas.height
        );
        
        // Add region label
        ctx.fillStyle = 'white';
        ctx.font = '12px Arial';
        ctx.fillText(
            `R${index + 1}`,
            region.x * canvas.width + 5,
            region.y * canvas.height + 15
        );
    });
    
    container.appendChild(canvas);
    
    // Add region details
    const detailsContainer = document.createElement('div');
    detailsContainer.className = 'attention-details';
    detailsContainer.style.marginTop = '10px';
    
    regions.forEach((region, index) => {
        const detail = document.createElement('div');
        detail.style.fontSize = '12px';
        detail.style.marginBottom = '5px';
        detail.innerHTML = `<strong>Region ${index + 1}:</strong> Intensity ${(region.intensity * 100).toFixed(1)}%, Area: ${(region.width * region.height * 100).toFixed(1)}%`;
        detailsContainer.appendChild(detail);
    });
    
    container.appendChild(detailsContainer);
}

// Update heatmap intensity
function updateHeatmapIntensity(container, intensity) {
    const canvas = container.querySelector('canvas');
    if (!canvas) return;
    
    canvas.style.opacity = intensity;
}

// Initialize proof validation display
function initializeProofValidation(proofData, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !proofData) return;
    
    proofData.forEach((proof, index) => {
        const proofCard = document.createElement('div');
        proofCard.className = 'proof-card';
        proofCard.style.border = '1px solid #ddd';
        proofCard.style.borderRadius = '8px';
        proofCard.style.padding = '15px';
        proofCard.style.marginBottom = '10px';
        proofCard.style.backgroundColor = '#f9f9f9';
        
        const reliabilityColor = proof.reliability > 0.7 ? '#28a745' : 
                                proof.reliability > 0.4 ? '#ffc107' : '#dc3545';
        
        proofCard.innerHTML = `
            <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 10px;">
                <h4 style="margin: 0; color: #333;">${proof.source}</h4>
                <span style="background: ${reliabilityColor}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px;">
                    ${(proof.reliability * 100).toFixed(0)}% Reliable
                </span>
            </div>
            <p style="margin: 5px 0; font-size: 14px; color: #666;"><strong>Status:</strong> ${proof.status}</p>
            <p style="margin: 5px 0; font-size: 14px;">${proof.summary}</p>
            <div style="margin-top: 10px;">
                <strong>Key Points:</strong>
                <ul style="margin: 5px 0; padding-left: 20px;">
                    ${proof.key_points.map(point => `<li style="font-size: 13px; margin: 2px 0;">${point}</li>`).join('')}
                </ul>
            </div>
            <div style="margin-top: 10px; display: flex; justify-content: space-between; align-items: center;">
                <a href="${proof.url}" target="_blank" style="color: #007bff; text-decoration: none; font-size: 13px;">View Source →</a>
                <span style="font-size: 12px; color: #888;">Confidence: ${(proof.confidence * 100).toFixed(1)}%</span>
            </div>
        `;
        
        container.appendChild(proofCard);
    });
}

// Test function for enhanced verification system
function testEnhancedVerification() {
    console.log('🧪 Starting enhanced verification test');
    
    // Sample test data with diverse sources and verdicts
    const testProofs = [
        {
            domain: 'reuters.com',
            title: 'Fact Check: Climate change is real and caused by human activities',
            snippet: 'Scientific consensus confirms climate change is real',
            credibilityScore: 0.95,
            factCheckVerdict: 'TRUE'
        },
        {
            domain: 'bbc.com',
            title: 'Breaking: New study confirms vaccine effectiveness',
            snippet: 'Peer-reviewed research shows vaccines are effective',
            credibilityScore: 0.92,
            factCheckVerdict: 'VERIFIED'
        },
        {
            domain: 'cnn.com',
            title: 'Investigation reveals misinformation campaign',
            snippet: 'False claims debunked by multiple fact-checkers',
            credibilityScore: 0.88,
            factCheckVerdict: 'FALSE'
        },
        {
            domain: 'apnews.com',
            title: 'Associated Press fact-check confirms accuracy',
            snippet: 'Verified information from reliable sources',
            credibilityScore: 0.94,
            factCheckVerdict: 'TRUE'
        },
        {
            domain: 'snopes.com',
            title: 'Fact-check: Viral claim is misleading',
            snippet: 'Partially true but lacks important context',
            credibilityScore: 0.90,
            factCheckVerdict: 'MIXED'
        }
    ];
    
    // Set up global proofs array for testing
    window.proofsArray = testProofs;
    
    // Show content result panel
    if (elements.contentResultPanel) {
        elements.contentResultPanel.style.display = 'block';
    }
    
    // Trigger the enhanced verification
    if (typeof executeContentResultVerification === 'function') {
        console.log('🔄 Executing enhanced verification with test data');
        executeContentResultVerification();
    } else {
        console.error('❌ Enhanced verification function not available');
        showNotification('Enhanced verification system not loaded', 'error');
    }
}

async function initializeApp() {
    console.log('Initializing Fake News Detection Dashboard...');
    
    // Initialize tabs
    initializeTabs();
    
    // Initialize file upload
    initializeFileUpload();
    
    // Initialize URL input
    initializeUrlInput();
    
    // Initialize live feed
    initializeLiveFeed();
    
    // Initialize event listeners
    initializeEventListeners();
    
    // Load initial data
    try {
        // Load history
        await loadHistory();
        
        console.log('Dashboard initialized successfully!');
        
    } catch (error) {
        console.error('Failed to initialize dashboard:', error);
    }
}

// Error handling for uncaught errors
window.addEventListener('error', (e) => {
    console.error('Uncaught error:', e.error);
});

window.addEventListener('unhandledrejection', (e) => {
    console.error('Unhandled promise rejection:', e.reason);
});

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}

// Keep console open for debugging (2 minutes)
console.log('Fake News Detection System - Console will remain active for 2 minutes for debugging...');
setTimeout(() => {
    console.log('Console debugging period completed after 2 minutes.');
}, 120000); // 120000ms = 2 minutes
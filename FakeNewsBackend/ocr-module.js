/**
 * =============================================================================
 * LIGHTWEIGHT OCR MODULE FOR FAKE NEWS VERIFICATION
 * =============================================================================
 * 
 * A high-performance, resource-efficient OCR implementation using Tesseract.js
 * Optimized for speed, low CPU/RAM usage, and seamless integration
 * 
 * SUBTASKS IMPLEMENTED:
 * 1. File Input Handling - Validate image files (JPEG/PNG, max 5MB)
 * 2. OCR Initialization - Lightweight Tesseract.js worker setup
 * 3. Preprocessing - Optional image enhancement for better OCR accuracy
 * 4. OCR Execution - Extract text with progress feedback
 * 5. Text Post-Processing - Clean and normalize extracted text
 * 6. UI Feedback - Loading states, progress, error handling
 * 7. Resource Management - Proper worker cleanup
 * 8. Integration - Connect to existing verification pipeline
 */

// =============================================================================
// SUBTASK 1: FILE INPUT HANDLING
// =============================================================================

/**
 * Validates uploaded image files for OCR processing
 * @param {File} file - The uploaded image file
 * @returns {Object} - Validation result with success flag and message
 */
function validateImageFile(file) {
    console.log('[OCR] Validating image file:', file.name);
    
    // Check if file exists
    if (!file) {
        return {
            success: false,
            message: 'No file selected. Please choose an image file.'
        };
    }
    
    // Validate file type (JPEG/PNG only)
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!allowedTypes.includes(file.type.toLowerCase())) {
        return {
            success: false,
            message: 'Invalid file type. Please upload JPEG or PNG images only.'
        };
    }
    
    // Validate file size (max 5MB)
    const maxSizeBytes = 5 * 1024 * 1024; // 5MB
    if (file.size > maxSizeBytes) {
        const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
        return {
            success: false,
            message: `File too large (${sizeMB}MB). Maximum size allowed is 5MB.`
        };
    }
    
    console.log('[OCR] File validation passed:', {
        name: file.name,
        type: file.type,
        size: `${(file.size / 1024).toFixed(2)}KB`
    });
    
    return {
        success: true,
        message: 'File validation successful'
    };
}

/**
 * Converts uploaded file to image element for processing
 * @param {File} file - The validated image file
 * @returns {Promise<HTMLImageElement>} - Promise resolving to image element
 */
function fileToImage(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            const img = new Image();
            
            img.onload = function() {
                console.log('[OCR] Image loaded successfully:', {
                    width: img.width,
                    height: img.height
                });
                resolve(img);
            };
            
            img.onerror = function() {
                reject(new Error('Failed to load image. The file may be corrupted.'));
            };
            
            img.src = e.target.result;
        };
        
        reader.onerror = function() {
            reject(new Error('Failed to read file. Please try again.'));
        };
        
        reader.readAsDataURL(file);
    });
}

// =============================================================================
// SUBTASK 2: OCR INITIALIZATION
// =============================================================================

/**
 * Global OCR worker instance for reuse
 */
let ocrWorker = null;
let isWorkerInitialized = false;

/**
 * Initializes Tesseract.js worker with lightweight configuration
 * @returns {Promise<Object>} - Promise resolving to initialized worker
 */
async function initializeOCRWorker() {
    if (isWorkerInitialized && ocrWorker) {
        console.log('[OCR] Using existing worker instance');
        return ocrWorker;
    }
    
    try {
        console.log('[OCR] Initializing Tesseract.js worker...');
        
        // Load Tesseract.js from CDN (lightweight approach)
        if (typeof Tesseract === 'undefined') {
            await loadTesseractLibrary();
        }
        
        // Create worker with English language only
        ocrWorker = await Tesseract.createWorker('eng', 1, {
            logger: m => {
                if (m.status === 'recognizing text') {
                    updateOCRProgress(Math.round(m.progress * 100));
                }
            },
            // Lightweight configuration for speed
            corePath: 'https://unpkg.com/tesseract.js-core@4.0.4/tesseract-core-simd.wasm.js',
            workerPath: 'https://unpkg.com/tesseract.js@4.1.1/dist/worker.min.js'
        });
        
        // Set OCR engine parameters for speed optimization
        await ocrWorker.setParameters({
            tessedit_pageseg_mode: Tesseract.PSM.SINGLE_BLOCK,
            tessedit_ocr_engine_mode: Tesseract.OEM.LSTM_ONLY,
            tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?;:"\'-()[]{}/@#$%^&*+=<>|\\~`'
        });
        
        isWorkerInitialized = true;
        console.log('[OCR] Worker initialized successfully');
        
        return ocrWorker;
        
    } catch (error) {
        console.error('[OCR] Worker initialization failed:', error);
        throw new Error('Failed to initialize OCR engine. Please check your internet connection.');
    }
}

/**
 * Loads Tesseract.js library dynamically
 * @returns {Promise<void>}
 */
function loadTesseractLibrary() {
    return new Promise((resolve, reject) => {
        if (typeof Tesseract !== 'undefined') {
            resolve();
            return;
        }
        
        const script = document.createElement('script');
        script.src = 'https://unpkg.com/tesseract.js@4.1.1/dist/tesseract.min.js';
        script.onload = resolve;
        script.onerror = () => reject(new Error('Failed to load Tesseract.js library'));
        document.head.appendChild(script);
    });
}

// =============================================================================
// SUBTASK 3: PREPROCESSING (OPTIONAL)
// =============================================================================

/**
 * Applies lightweight image preprocessing to improve OCR accuracy
 * @param {HTMLImageElement} img - Source image
 * @returns {HTMLCanvasElement} - Processed canvas
 */
function preprocessImage(img) {
    console.log('[OCR] Applying image preprocessing...');
    
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Set canvas dimensions
    canvas.width = img.width;
    canvas.height = img.height;
    
    // Draw original image
    ctx.drawImage(img, 0, 0);
    
    // Get image data for processing
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // Apply grayscale and contrast enhancement
    for (let i = 0; i < data.length; i += 4) {
        // Convert to grayscale
        const gray = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
        
        // Apply contrast enhancement (simple threshold)
        const enhanced = gray > 128 ? Math.min(255, gray * 1.2) : Math.max(0, gray * 0.8);
        
        data[i] = enhanced;     // Red
        data[i + 1] = enhanced; // Green
        data[i + 2] = enhanced; // Blue
        // Alpha channel remains unchanged
    }
    
    // Put processed data back to canvas
    ctx.putImageData(imageData, 0, 0);
    
    console.log('[OCR] Image preprocessing completed');
    return canvas;
}

// =============================================================================
// SUBTASK 4: OCR EXECUTION
// =============================================================================

/**
 * Executes OCR on the provided image with progress feedback
 * @param {HTMLImageElement|HTMLCanvasElement} imageSource - Image to process
 * @param {Object} options - OCR options
 * @returns {Promise<string>} - Extracted text
 */
async function executeOCR(imageSource, options = {}) {
    const {
        usePreprocessing = true,
        showProgress = true
    } = options;
    
    try {
        console.log('[OCR] Starting text extraction...');
        
        // Show loading state
        if (showProgress) {
            showOCRLoadingState('Initializing OCR engine...');
        }
        
        // Initialize worker
        const worker = await initializeOCRWorker();
        
        // Preprocess image if enabled
        let processedImage = imageSource;
        if (usePreprocessing && imageSource instanceof HTMLImageElement) {
            if (showProgress) {
                updateOCRProgress(10, 'Preprocessing image...');
            }
            processedImage = preprocessImage(imageSource);
        }
        
        // Update progress
        if (showProgress) {
            updateOCRProgress(20, 'Extracting text...');
        }
        
        // Perform OCR
        const { data: { text, confidence } } = await worker.recognize(processedImage);
        
        console.log('[OCR] Text extraction completed:', {
            textLength: text.length,
            confidence: Math.round(confidence)
        });
        
        // Update progress to completion
        if (showProgress) {
            updateOCRProgress(100, 'Text extraction complete!');
        }
        
        return text;
        
    } catch (error) {
        console.error('[OCR] Text extraction failed:', error);
        throw new Error(`OCR processing failed: ${error.message}`);
    }
}

// =============================================================================
// SUBTASK 5: TEXT POST-PROCESSING
// =============================================================================

/**
 * Cleans and normalizes extracted OCR text
 * @param {string} rawText - Raw OCR output
 * @returns {string} - Cleaned and normalized text
 */
function postProcessOCRText(rawText) {
    console.log('[OCR] Post-processing extracted text...');
    
    if (!rawText || typeof rawText !== 'string') {
        return '';
    }
    
    let cleanedText = rawText;
    
    // Remove excessive whitespace
    cleanedText = cleanedText.replace(/\s+/g, ' ');
    
    // Fix common OCR errors
    const ocrCorrections = {
        // Common character misrecognitions
        '0': 'O', // Zero to letter O in words
        '1': 'I', // One to letter I in words
        '5': 'S', // Five to letter S in words
        '8': 'B', // Eight to letter B in words
        // Add more corrections as needed
    };
    
    // Apply corrections contextually (only in word contexts)
    cleanedText = cleanedText.replace(/\b\w*\b/g, (word) => {
        let correctedWord = word;
        for (const [wrong, correct] of Object.entries(ocrCorrections)) {
            // Only replace if it makes sense in context
            if (word.includes(wrong) && /[a-zA-Z]/.test(word)) {
                correctedWord = correctedWord.replace(new RegExp(wrong, 'g'), correct);
            }
        }
        return correctedWord;
    });
    
    // Trim and clean up
    cleanedText = cleanedText.trim();
    
    // Remove lines with only special characters or very short content
    cleanedText = cleanedText
        .split('\n')
        .filter(line => line.trim().length > 2 && /[a-zA-Z0-9]/.test(line))
        .join('\n');
    
    console.log('[OCR] Text post-processing completed:', {
        originalLength: rawText.length,
        cleanedLength: cleanedText.length
    });
    
    return cleanedText;
}

// =============================================================================
// SUBTASK 6: UI FEEDBACK
// =============================================================================

/**
 * Shows OCR loading state with spinner
 * @param {string} message - Loading message
 */
function showOCRLoadingState(message = 'Processing image...') {
    console.log('[OCR UI]', message);
    
    // Create or update loading overlay
    let loadingOverlay = document.getElementById('ocrLoadingOverlay');
    if (!loadingOverlay) {
        loadingOverlay = document.createElement('div');
        loadingOverlay.id = 'ocrLoadingOverlay';
        loadingOverlay.className = 'ocr-loading-overlay';
        loadingOverlay.innerHTML = `
            <div class="ocr-loading-content">
                <div class="ocr-spinner"></div>
                <div class="ocr-loading-message" id="ocrLoadingMessage">${message}</div>
                <div class="ocr-progress-bar">
                    <div class="ocr-progress-fill" id="ocrProgressFill" style="width: 0%"></div>
                </div>
                <div class="ocr-progress-text" id="ocrProgressText">0%</div>
            </div>
        `;
        document.body.appendChild(loadingOverlay);
    } else {
        document.getElementById('ocrLoadingMessage').textContent = message;
    }
    
    loadingOverlay.style.display = 'flex';
}

/**
 * Updates OCR progress
 * @param {number} percentage - Progress percentage (0-100)
 * @param {string} message - Optional progress message
 */
function updateOCRProgress(percentage, message = null) {
    const progressFill = document.getElementById('ocrProgressFill');
    const progressText = document.getElementById('ocrProgressText');
    const loadingMessage = document.getElementById('ocrLoadingMessage');
    
    if (progressFill) {
        progressFill.style.width = `${Math.min(100, Math.max(0, percentage))}%`;
    }
    
    if (progressText) {
        progressText.textContent = `${Math.round(percentage)}%`;
    }
    
    if (message && loadingMessage) {
        loadingMessage.textContent = message;
    }
}

/**
 * Hides OCR loading state
 */
function hideOCRLoadingState() {
    const loadingOverlay = document.getElementById('ocrLoadingOverlay');
    if (loadingOverlay) {
        loadingOverlay.style.display = 'none';
    }
}

/**
 * Shows extracted text for user confirmation
 * @param {string} extractedText - The OCR extracted text
 * @returns {Promise<boolean>} - User confirmation result
 */
function showExtractedTextConfirmation(extractedText) {
    return new Promise((resolve) => {
        // Create confirmation modal
        const modal = document.createElement('div');
        modal.className = 'ocr-confirmation-modal';
        modal.innerHTML = `
            <div class="ocr-confirmation-content">
                <h3>📄 Text Extracted Successfully</h3>
                <div class="ocr-extracted-text">
                    <label>Extracted Text:</label>
                    <textarea readonly>${extractedText}</textarea>
                </div>
                <div class="ocr-confirmation-buttons">
                    <button id="ocrConfirmBtn" class="ocr-btn ocr-btn-primary">✅ Analyze This Text</button>
                    <button id="ocrCancelBtn" class="ocr-btn ocr-btn-secondary">❌ Cancel</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Handle button clicks
        document.getElementById('ocrConfirmBtn').onclick = () => {
            document.body.removeChild(modal);
            resolve(true);
        };
        
        document.getElementById('ocrCancelBtn').onclick = () => {
            document.body.removeChild(modal);
            resolve(false);
        };
    });
}

/**
 * Shows OCR error message with retry option
 * @param {string} errorMessage - Error message to display
 * @returns {Promise<boolean>} - Whether user wants to retry
 */
function showOCRError(errorMessage) {
    return new Promise((resolve) => {
        hideOCRLoadingState();
        
        const modal = document.createElement('div');
        modal.className = 'ocr-error-modal';
        modal.innerHTML = `
            <div class="ocr-error-content">
                <h3>❌ OCR Processing Failed</h3>
                <div class="ocr-error-message">${errorMessage}</div>
                <div class="ocr-error-buttons">
                    <button id="ocrRetryBtn" class="ocr-btn ocr-btn-primary">🔄 Try Again</button>
                    <button id="ocrCloseBtn" class="ocr-btn ocr-btn-secondary">Close</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        document.getElementById('ocrRetryBtn').onclick = () => {
            document.body.removeChild(modal);
            resolve(true);
        };
        
        document.getElementById('ocrCloseBtn').onclick = () => {
            document.body.removeChild(modal);
            resolve(false);
        };
    });
}

// =============================================================================
// SUBTASK 7: RESOURCE MANAGEMENT
// =============================================================================

/**
 * Properly terminates OCR worker to free memory
 */
async function terminateOCRWorker() {
    if (ocrWorker && isWorkerInitialized) {
        try {
            console.log('[OCR] Terminating worker to free memory...');
            await ocrWorker.terminate();
            ocrWorker = null;
            isWorkerInitialized = false;
            console.log('[OCR] Worker terminated successfully');
        } catch (error) {
            console.error('[OCR] Error terminating worker:', error);
        }
    }
}

/**
 * Cleans up OCR resources and UI elements
 */
function cleanupOCRResources() {
    // Hide any active UI elements
    hideOCRLoadingState();
    
    // Remove any temporary DOM elements
    const modals = document.querySelectorAll('.ocr-confirmation-modal, .ocr-error-modal');
    modals.forEach(modal => {
        if (modal.parentNode) {
            modal.parentNode.removeChild(modal);
        }
    });
    
    console.log('[OCR] Resources cleaned up');
}

// =============================================================================
// SUBTASK 8: INTEGRATION WITH EXISTING PIPELINE
// =============================================================================

/**
 * Main OCR processing function that integrates with existing verification pipeline
 * @param {File} imageFile - The uploaded image file
 * @returns {Promise<string>} - Extracted and processed text
 */
async function processImageForVerification(imageFile) {
    try {
        console.log('[OCR INTEGRATION] Starting image processing for verification...');
        
        // Step 1: Validate file
        const validation = validateImageFile(imageFile);
        if (!validation.success) {
            throw new Error(validation.message);
        }
        
        // Step 2: Convert to image
        showOCRLoadingState('Loading image...');
        const img = await fileToImage(imageFile);
        
        // Step 3: Execute OCR
        updateOCRProgress(5, 'Initializing OCR...');
        const rawText = await executeOCR(img, {
            usePreprocessing: true,
            showProgress: true
        });
        
        // Step 4: Post-process text
        updateOCRProgress(95, 'Cleaning extracted text...');
        const cleanedText = postProcessOCRText(rawText);
        
        // Step 5: Validate extracted text
        if (!cleanedText || cleanedText.trim().length < 10) {
            throw new Error('No readable text found in the image. Please try a clearer image.');
        }
        
        // Step 6: Show confirmation and get user approval
        hideOCRLoadingState();
        const userConfirmed = await showExtractedTextConfirmation(cleanedText);
        
        if (!userConfirmed) {
            throw new Error('OCR processing cancelled by user');
        }
        
        console.log('[OCR INTEGRATION] Image processing completed successfully');
        return cleanedText;
        
    } catch (error) {
        console.error('[OCR INTEGRATION] Processing failed:', error);
        
        // Show error and ask for retry
        const shouldRetry = await showOCRError(error.message);
        
        if (shouldRetry) {
            // Recursive retry
            return await processImageForVerification(imageFile);
        } else {
            throw error;
        }
    } finally {
        // Always clean up
        cleanupOCRResources();
    }
}

/**
 * Integrates OCR with existing analyze button event handlers
 * @param {File} imageFile - The image file to process
 */
async function handleImageAnalysis(imageFile) {
    try {
        // Process image and extract text
        const extractedText = await processImageForVerification(imageFile);
        
        // Check if the existing pipeline function exists
        if (typeof window.executeUniversalFactVerification === 'function') {
            console.log('[OCR INTEGRATION] Triggering universal fact verification pipeline...');
            
            // Trigger existing verification pipeline with extracted text
            await window.executeUniversalFactVerification(extractedText);
            
        } else if (typeof window.executeMultiStageRelevancePipeline === 'function') {
            console.log('[OCR INTEGRATION] Triggering multi-stage relevance pipeline...');
            
            // Trigger existing verification pipeline with extracted text
            await window.executeMultiStageRelevancePipeline(extractedText);
            
        } else if (typeof window.analyzeContentVerification === 'function') {
            console.log('[OCR INTEGRATION] Triggering legacy verification pipeline...');
            
            // Fallback to legacy pipeline
            await window.analyzeContentVerification(extractedText);
            
        } else {
            console.warn('[OCR INTEGRATION] No verification pipeline found');
            
            // Show extracted text in results area as fallback
            const resultsArea = document.getElementById('proofValidationArea') || document.getElementById('proofs-container');
            if (resultsArea) {
                resultsArea.innerHTML = `
                    <div class="ocr-results">
                        <h3>📄 Extracted Text (OCR)</h3>
                        <div class="extracted-text-display" style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; white-space: pre-wrap; font-family: monospace;">${extractedText}</div>
                        <p><em>Note: Automatic verification pipeline not available. Please copy the text above for manual analysis.</em></p>
                    </div>
                `;
            }
        }
        
    } catch (error) {
        console.error('[OCR INTEGRATION] Image analysis failed:', error);
        
        // Show error in existing error handling system
        if (typeof window.showErrorState === 'function') {
            window.showErrorState(`Image analysis failed: ${error.message}`);
        } else {
            alert(`Image analysis failed: ${error.message}`);
        }
    }
}

// =============================================================================
// GLOBAL EXPORTS AND INITIALIZATION
// =============================================================================

// =============================================================================
// PUBLIC API FUNCTIONS
// =============================================================================

/**
 * Initialize OCR Module - Public API function
 * @returns {Promise<boolean>} - Success status
 */
async function initializeOCRModule() {
    console.log('[OCR API] Initializing OCR module...');
    try {
        await initializeOCRWorker();
        console.log('[OCR API] OCR module initialized successfully');
        return true;
    } catch (error) {
        console.error('[OCR API] Failed to initialize OCR module:', error);
        return false;
    }
}

/**
 * Process Image with OCR - Public API function
 * @param {File} imageFile - The image file to process
 * @returns {Promise<string>} - Extracted text
 */
async function processImageWithOCR(imageFile) {
    console.log('[OCR API] Processing image with OCR...');
    
    // Validate the image file
    const validation = validateImageFile(imageFile);
    if (!validation.success) {
        throw new Error(validation.message);
    }
    
    try {
        // Initialize OCR worker if not already done
        if (!isWorkerInitialized) {
            await initializeOCRWorker();
        }
        
        // Convert file to image
        const img = await fileToImage(imageFile);
        
        // Execute OCR
        const result = await executeOCR(img, {
            showProgress: true,
            preprocessImage: true
        });
        
        if (result.success && result.text) {
            console.log('[OCR API] Text extraction successful');
            return result.text;
        } else {
            throw new Error(result.error || 'OCR processing failed');
        }
    } catch (error) {
        console.error('[OCR API] OCR processing error:', error);
        throw error;
    }
}

// Main OCR Pipeline execution function
async function executeOCRPipeline(imageInput, options = {}) {
    try {
        console.log('Starting OCR pipeline execution...');
        
        // Initialize OCR if not already done
        if (!window.ocrWorker) {
            await initializeOCRModule();
        }
        
        // Process the image
        const result = await processImageWithOCR(imageInput, options);
        
        // Enhanced result with pipeline metadata
        const pipelineResult = {
            ...result,
            pipeline: 'OCR',
            timestamp: new Date().toISOString(),
            processingTime: result.processingTime || 0,
            confidence: result.confidence || 0,
            wordCount: result.text ? result.text.split(/\s+/).length : 0
        };
        
        console.log('OCR pipeline completed successfully');
        return pipelineResult;
        
    } catch (error) {
        console.error('OCR pipeline execution failed:', error);
        throw new Error(`OCR Pipeline Error: ${error.message}`);
    }
}

// Export functions to global scope for external access
window.initializeOCRModule = initializeOCRModule;
window.processImageWithOCR = processImageWithOCR;
window.executeOCRPipeline = executeOCRPipeline;
window.processImageForVerification = processImageForVerification;
window.handleImageAnalysis = handleImageAnalysis;
window.validateImageFile = validateImageFile;
window.executeOCR = executeOCR;
window.terminateOCRWorker = terminateOCRWorker;

// Auto-cleanup on page unload
window.addEventListener('beforeunload', () => {
    terminateOCRWorker();
    cleanupOCRResources();
});

console.log('🔍 OCR Module Loaded Successfully');
console.log('📷 Ready for image-to-text conversion');
console.log('⚡ Optimized for speed and low resource usage');
console.log('🎯 Integrated with fake news verification pipeline');

// =============================================================================
// CSS STYLES FOR OCR UI COMPONENTS
// =============================================================================

// Inject CSS styles for OCR UI components
const ocrStyles = `
<style>
.ocr-loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.8);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 10000;
}

.ocr-loading-content {
    background: white;
    padding: 30px;
    border-radius: 10px;
    text-align: center;
    max-width: 400px;
    width: 90%;
}

.ocr-spinner {
    border: 4px solid #f3f3f3;
    border-top: 4px solid #3498db;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: ocrSpin 1s linear infinite;
    margin: 0 auto 20px;
}

@keyframes ocrSpin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.ocr-loading-message {
    font-size: 16px;
    margin-bottom: 20px;
    color: #333;
}

.ocr-progress-bar {
    width: 100%;
    height: 20px;
    background: #f0f0f0;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 10px;
}

.ocr-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #3498db, #2ecc71);
    transition: width 0.3s ease;
}

.ocr-progress-text {
    font-size: 14px;
    color: #666;
}

.ocr-confirmation-modal,
.ocr-error-modal {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.8);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 10001;
}

.ocr-confirmation-content,
.ocr-error-content {
    background: white;
    padding: 30px;
    border-radius: 10px;
    max-width: 500px;
    width: 90%;
    max-height: 80vh;
    overflow-y: auto;
}

.ocr-extracted-text textarea {
    width: 100%;
    height: 150px;
    margin: 10px 0;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 5px;
    font-family: monospace;
    resize: vertical;
}

.ocr-confirmation-buttons,
.ocr-error-buttons {
    display: flex;
    gap: 10px;
    justify-content: center;
    margin-top: 20px;
}

.ocr-btn {
    padding: 10px 20px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 14px;
    transition: background-color 0.3s;
}

.ocr-btn-primary {
    background: #3498db;
    color: white;
}

.ocr-btn-primary:hover {
    background: #2980b9;
}

.ocr-btn-secondary {
    background: #95a5a6;
    color: white;
}

.ocr-btn-secondary:hover {
    background: #7f8c8d;
}

.ocr-error-message {
    background: #ffebee;
    border: 1px solid #f44336;
    padding: 15px;
    border-radius: 5px;
    margin: 15px 0;
    color: #c62828;
}

.ocr-results {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 20px;
    margin: 20px 0;
}

.extracted-text-display {
    background: white;
    border: 1px solid #ddd;
    padding: 15px;
    border-radius: 5px;
    margin: 15px 0;
    font-family: monospace;
    white-space: pre-wrap;
    max-height: 300px;
    overflow-y: auto;
}
</style>
`;

// Inject styles into document head
if (document.head) {
    document.head.insertAdjacentHTML('beforeend', ocrStyles);
} else {
    document.addEventListener('DOMContentLoaded', () => {
        document.head.insertAdjacentHTML('beforeend', ocrStyles);
    });
}
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
// MISTRAL REFINEMENT HELPER
// =============================================================================

/**
 * Uses Mistral to correct OCR noise and extract the main claim.
 * Falls back gracefully if the API is unavailable.
 * @param {string} text - Cleaned OCR text
 * @returns {Promise<{clean_text:string, main_claim:string, quality:number}>}
 */
/**
 * Ensures `window.MISTRAL_API_KEY` is set by reading 
 * the `.env` file served from the same origin when missing.
 * WARNING: Exposes the API key to the browser if `.env` is publicly served.
 */
async function ensureMistralKeyFromEnv() {
    if (window.MISTRAL_API_KEY && window.MISTRAL_API_KEY.trim().length > 0) {
        return window.MISTRAL_API_KEY.trim();
    }
    try {
        // Try loading from same-origin .env
        const resp = await fetch('./.env', { cache: 'no-store' });
        if (!resp.ok) return '';
        const envText = await resp.text();
        const match = envText.match(/^\s*MISTRAL_API_KEY\s*=\s*(.+)\s*$/m);
        if (match && match[1]) {
            window.MISTRAL_API_KEY = match[1].trim();
            return window.MISTRAL_API_KEY;
        }
        return '';
    } catch (_) {
        return '';
    }
}

async function refineOCRTextWithMistral(text) {
    let apiKey = (window.MISTRAL_API_KEY || '').trim();
    // If input text is empty or extremely short, bail early and let caller show error
    if (!text || text.trim().length < 10) {
        throw new Error('Insufficient OCR text for refinement');
    }
    if (!apiKey) {
        apiKey = await ensureMistralKeyFromEnv();
        if (!apiKey) throw new Error('Missing MISTRAL_API_KEY');
    }

    try {
        const payload = {
            model: 'mistral-small-latest',
            temperature: 0.2,
            max_tokens: 512,
            messages: [
                {
                    role: 'system',
                    content: 'You are an assistant that cleans OCR text and extracts the single most checkable claim. Return strict JSON with keys clean_text, main_claim, and quality (0-1). Keep content concise; remove duplicates and menu/boilerplate noise.'
                },
                {
                    role: 'user',
                    content: `Please clean the OCR text, remove noise/duplicates, and extract a single most checkable factual claim suitable for verification. Return strict JSON: {"clean_text":"...","main_claim":"...","quality":0-1}.
\n\nOCR TEXT:\n\n${text}`
                }
            ]
        };

        const resp = await fetch('https://api.mistral.ai/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!resp.ok) {
            console.warn('[OCR] Mistral API error:', resp.status, await resp.text());
            return { clean_text: text, main_claim: text, quality: 0.5 };
        }

        const data = await resp.json();
        const content = data?.choices?.[0]?.message?.content || '';

        // Try parsing JSON content; if not JSON, return heuristic fallback
        let parsed;
        try {
            parsed = JSON.parse(content);
        } catch {
            // Attempt to extract JSON block from text
            const match = content.match(/\{[\s\S]*\}/);
            if (match) {
                try { parsed = JSON.parse(match[0]); } catch { /* ignore */ }
            }
        }

        if (!parsed || typeof parsed !== 'object') {
            return { clean_text: text, main_claim: text, quality: 0.6 };
        }

        // Sanitize to strip any accidental prompt echo or boilerplate
        const clean_text_raw = (parsed.clean_text || text).trim();
        const main_claim_raw = (parsed.main_claim || clean_text_raw || text).trim();

        const clean_text = stripPromptArtifacts(clean_text_raw);
        let main_claim = stripPromptArtifacts(main_claim_raw);
        // If model echoed the task or produced empty output, fall back
        if (!main_claim || /\bTask:\b/i.test(main_claim_raw)) {
            main_claim = clean_text || text;
        }

        const quality = Number(parsed.quality || 0.7);
        return { clean_text, main_claim, quality };
    } catch (error) {
        console.warn('[OCR] Mistral refinement failed:', error);
        throw error;
    }
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
 * Initializes Tesseract.js worker with proper lifecycle and CDN fallbacks
 * @returns {Promise<Object>} - Promise resolving to initialized worker
 */
async function initializeOCRWorker() {
    if (isWorkerInitialized && ocrWorker) {
        console.log('[OCR] Using existing worker instance');
        return ocrWorker;
    }

    try {
        console.log('[OCR] Initializing Tesseract.js worker...');

        // Load Tesseract.js from CDN (with fallback)
        if (typeof Tesseract === 'undefined') {
            await loadTesseractLibrary();
        }

        // Create worker with proper options and then load/initialize language
        ocrWorker = await Tesseract.createWorker({
            logger: m => {
                if (m.status === 'recognizing text') {
                    updateOCRProgress(Math.round(((m.progress || 0) * 100)));
                }
            },
            corePath: 'https://unpkg.com/tesseract.js-core@4.0.4/tesseract-core-simd.wasm.js',
            workerPath: 'https://unpkg.com/tesseract.js@4.1.1/dist/worker.min.js',
            langPath: 'https://tessdata.projectnaptha.com/4.0.0'
        });

        // Correct worker lifecycle
        await ocrWorker.load();
        await ocrWorker.loadLanguage('eng');
        await ocrWorker.initialize('eng');

        // Set OCR engine parameters for speed optimization
        await ocrWorker.setParameters({
            // Use AUTO for varied layouts; we will override per-pass when needed
            tessedit_pageseg_mode: Tesseract.PSM.AUTO,
            tessedit_ocr_engine_mode: Tesseract.OEM.LSTM_ONLY,
            preserve_interword_spaces: '1',
            user_defined_dpi: '300',
            tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?;:\"\'-()[]{}/@#$%^&*+=<>|\\~`'
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

        const tryLoad = (src, onFail) => {
            const script = document.createElement('script');
            script.src = src;
            script.async = true;
            script.onload = () => resolve();
            script.onerror = () => {
                if (onFail) onFail(); else reject(new Error('Failed to load Tesseract.js library'));
            };
            document.head.appendChild(script);
        };

        // Primary CDN (unpkg) then fallback (jsdelivr)
        tryLoad('https://unpkg.com/tesseract.js@4.1.1/dist/tesseract.min.js', () => {
            console.warn('[OCR] Primary CDN failed, attempting fallback (jsdelivr)');
            tryLoad('https://cdn.jsdelivr.net/npm/tesseract.js@4.1.1/dist/tesseract.min.js');
        });
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
    
    // Upscale small images for better OCR
    const upscaleFactor = Math.max(1, Math.min(2, 1000 / Math.max(img.width, 1)));
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = Math.round(img.width * upscaleFactor);
    canvas.height = Math.round(img.height * upscaleFactor);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // Grayscale and simple adaptive thresholding based on mean intensity
    let sum = 0;
    for (let i = 0; i < data.length; i += 4) {
        const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
        sum += gray;
    }
    const mean = sum / (data.length / 4);
    const threshold = Math.min(220, Math.max(60, mean));
    
    for (let i = 0; i < data.length; i += 4) {
        const gray = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
        const bin = gray > threshold ? 255 : 0;
        data[i] = bin;
        data[i + 1] = bin;
        data[i + 2] = bin;
        // Keep alpha channel
    }
    
    ctx.putImageData(imageData, 0, 0);
    console.log('[OCR] Image preprocessing completed');
    return canvas;
}

/**
 * Aggressive preprocessing for tough images (extra upscale and threshold)
 */
function aggressivePreprocessImage(imgOrCanvas) {
    const srcCanvas = imgOrCanvas instanceof HTMLCanvasElement ? imgOrCanvas : (() => {
        const c = document.createElement('canvas');
        const cx = c.getContext('2d');
        c.width = imgOrCanvas.width; c.height = imgOrCanvas.height;
        cx.drawImage(imgOrCanvas, 0, 0);
        return c;
    })();
    const out = document.createElement('canvas');
    const ox = out.getContext('2d');
    const factor = 2;
    out.width = srcCanvas.width * factor;
    out.height = srcCanvas.height * factor;
    ox.imageSmoothingEnabled = true;
    ox.imageSmoothingQuality = 'high';
    ox.drawImage(srcCanvas, 0, 0, out.width, out.height);
    const id = ox.getImageData(0, 0, out.width, out.height);
    const d = id.data;
    // Strong binarization
    for (let i = 0; i < d.length; i += 4) {
        const gray = (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]);
        const bin = gray > 140 ? 255 : 0; // mid threshold
        d[i] = bin; d[i + 1] = bin; d[i + 2] = bin;
    }
    ox.putImageData(id, 0, 0);
    return out;
}

/**
 * Crop bottom region (e.g., news ticker/banner) ratio of height
 */
function cropBottomRegion(imgOrCanvas, ratio = 0.35) {
    const w = imgOrCanvas.width; const h = imgOrCanvas.height;
    const startY = Math.max(0, Math.round(h * (1 - ratio)));
    const out = document.createElement('canvas');
    const ox = out.getContext('2d');
    out.width = w; out.height = Math.round(h * ratio);
    ox.drawImage(imgOrCanvas, 0, startY, w, out.height, 0, 0, w, out.height);
    return out;
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
        
        // First pass OCR
        const { data: { text: text1, confidence: conf1 } } = await worker.recognize(processedImage);
        let finalText = text1 || '';
        let finalConf = conf1 || 0;

        // Fallback passes for tough images
        if ((finalText.trim().length < 10) && imageSource) {
            // Try aggressive preprocessing
            if (showProgress) updateOCRProgress(40, 'Enhancing image (aggressive)...');
            const aggressive = aggressivePreprocessImage(processedImage);
            await worker.setParameters({ tessedit_pageseg_mode: Tesseract.PSM.SINGLE_COLUMN });
            const { data: { text: text2, confidence: conf2 } } = await worker.recognize(aggressive);
            if ((text2 || '').trim().length > finalText.trim().length) {
                finalText = text2; finalConf = conf2 || finalConf;
            }

            // Try cropping bottom banner region
            if (finalText.trim().length < 10) {
                if (showProgress) updateOCRProgress(60, 'Focusing on banner region...');
                const cropped = cropBottomRegion(processedImage);
                await worker.setParameters({ tessedit_pageseg_mode: Tesseract.PSM.SINGLE_LINE });
                const { data: { text: text3, confidence: conf3 } } = await worker.recognize(cropped);
                if ((text3 || '').trim().length > finalText.trim().length) {
                    finalText = text3; finalConf = conf3 || finalConf;
                }
            }
        }
        
        console.log('[OCR] Text extraction completed:', {
            textLength: finalText.length,
            confidence: Math.round(finalConf)
        });
        
        // Update progress to completion
        if (showProgress) {
            updateOCRProgress(100, 'Text extraction complete!');
        }
        
        return finalText;
        
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
    // Minimal normalization only; avoid aggressive character substitutions
    
    // Trim and clean up
    cleanedText = cleanedText.trim();

    // Remove any accidental prompt or UI artifacts if present
    cleanedText = stripPromptArtifacts(cleanedText);
    
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
// MISTRAL VISION OCR
// =============================================================================

/**
 * Convert a File to a data URL (base64) for API submission
 * @param {File} file
 * @returns {Promise<string>} data URL
 */
function fileToDataURL(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = (e) => reject(e);
        reader.readAsDataURL(file);
    });
}

/**
 * Uses Mistral Vision to extract text and the main claim directly from an image.
 * Requires a valid `window.MISTRAL_API_KEY`. No fallbacks are used.
 * @param {File} imageFile
 * @returns {Promise<{clean_text:string, main_claim:string, quality:number}>}
 */
async function extractTextWithMistralVision(imageFile) {
    let apiKey = (window.MISTRAL_API_KEY || '').trim();
    if (!apiKey) {
        apiKey = await ensureMistralKeyFromEnv();
        if (!apiKey) throw new Error('Missing MISTRAL_API_KEY');
    }
    if (!imageFile) throw new Error('No image provided');

    const dataUrl = await fileToDataURL(imageFile);

    const payload = {
        model: 'pixtral-large-latest',
        temperature: 0.1,
        max_tokens: 512,
        messages: [
            {
                role: 'system',
                content: 'You are a vision assistant. Read any text in the image and extract the single most checkable factual claim. Return strict JSON with keys clean_text, main_claim, and quality (0-1).'
            },
            {
                role: 'user',
                content: [
                    { type: 'text', text: 'Read the text in this image, clean noise, and return JSON.' },
                    { type: 'image_url', image_url: dataUrl }
                ]
            }
        ]
    };

    const resp = await fetch('https://api.mistral.ai/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        body: JSON.stringify(payload)
    });

    if (!resp.ok) {
        throw new Error(`Mistral Vision API error: ${resp.status} ${await resp.text()}`);
    }

    const data = await resp.json();
    const content = data?.choices?.[0]?.message?.content || '';

    let parsed;
    try { parsed = JSON.parse(content); } catch {
        const match = content.match(/\{[\s\S]*\}/);
        if (match) { try { parsed = JSON.parse(match[0]); } catch { /* ignore */ } }
    }
    if (!parsed || typeof parsed !== 'object') {
        throw new Error('Mistral Vision response parsing failed');
    }

    const clean_text = stripPromptArtifacts((parsed.clean_text || '').trim());
    const main_claim = stripPromptArtifacts((parsed.main_claim || clean_text).trim());
    const quality = Number(parsed.quality || 0.7);
    if (!main_claim || main_claim.length < 5) throw new Error('No meaningful text extracted from image');
    return { clean_text, main_claim, quality };
}

/**
 * Strips prompt/UI artifacts that may accidentally appear in outputs
 * @param {string} text
 * @returns {string}
 */
function stripPromptArtifacts(text) {
    if (!text || typeof text !== 'string') return '';
    let t = text;
    const patterns = [
        /\bOCR TEXT:\b[\s\S]*/gmi,
        /\bTask:\b[\s\S]*/gmi,
        /Extracted\s+Text:/gmi,
        /Analyze\s+This\s+Text/gmi,
        /Cancel/gmi
    ];
    patterns.forEach((p) => { t = t.replace(p, ''); });
    // Collapse whitespace after removals
    t = t.replace(/\s+/g, ' ').trim();
    return t;
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
        
        // Step 2: Use vision OCR directly (no local OCR fallback)
        showOCRLoadingState('Fetching details using the OCR...');
        updateOCRProgress(10, 'Fetching details using the OCR...');
        const refined = await extractTextWithMistralVision(imageFile);
        let finalText = stripPromptArtifacts((refined.main_claim || refined.clean_text || '').trim());

        // Step 6: Validate extracted text
        if (!finalText || finalText.trim().length < 10) {
            throw new Error('No readable text found in the image. Please try a clearer image.');
        }

        // Step 7: Show confirmation and get user approval
        hideOCRLoadingState();
        const userConfirmed = await showExtractedTextConfirmation(finalText);

        if (!userConfirmed) {
            throw new Error('OCR processing cancelled by user');
        }

        console.log('[OCR INTEGRATION] Image processing completed successfully');
        return finalText;
        
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
        
        // Preferred integration: populate input and trigger Dashboard analysis
        const contentInput = document.getElementById('content-input');
        if (contentInput) {
            contentInput.value = extractedText;
        }

        if (typeof window.analyzeContent === 'function') {
            console.log('[OCR INTEGRATION] Calling analyzeContent with OCR-refined text');
            await window.analyzeContent();
            return;
        }

        // Fallbacks to other pipelines if present
        if (typeof window.executeUniversalFactVerification === 'function') {
            console.log('[OCR INTEGRATION] Triggering universal fact verification pipeline...');
            await window.executeUniversalFactVerification(extractedText);
            return;
        }
        if (typeof window.executeMultiStageRelevancePipeline === 'function') {
            console.log('[OCR INTEGRATION] Triggering multi-stage relevance pipeline...');
            await window.executeMultiStageRelevancePipeline(extractedText);
            return;
        }
        if (typeof window.analyzeContentVerification === 'function') {
            console.log('[OCR INTEGRATION] Triggering legacy verification pipeline...');
            await window.analyzeContentVerification(extractedText);
            return;
        }

        console.warn('[OCR INTEGRATION] No verification pipeline found');
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
        const text = await executeOCR(img, {
            showProgress: true,
            preprocessImage: true
        });
        
        if (text && text.trim().length > 0) {
            console.log('[OCR API] Text extraction successful');
            return text;
        }
        throw new Error('No text extracted from image');
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
        const text = await processImageWithOCR(imageInput, options);
        
        // Enhanced result with pipeline metadata
        const pipelineResult = {
            text,
            pipeline: 'OCR',
            timestamp: new Date().toISOString(),
            processingTime: 0,
            confidence: 0,
            wordCount: text ? text.split(/\s+/).length : 0
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
# Authentication System Architecture
## Hybrid Deep Learning Fake News Detection Application

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Authentication Flow](#authentication-flow)
3. [Security Architecture](#security-architecture)
4. [User Interface Design](#user-interface-design)
5. [Session Management](#session-management)
6. [Implementation Guidelines](#implementation-guidelines)
7. [Security Validation Procedures](#security-validation-procedures)
8. [Responsive Design Specifications](#responsive-design-specifications)
9. [Error Handling](#error-handling)
10. [Accessibility Considerations](#accessibility-considerations)

---

## System Overview

### Technology Stack
- **Frontend**: Vanilla HTML5, CSS3, JavaScript (ES6+)
- **Session Management**: Browser sessionStorage/localStorage
- **Communication**: Fetch API with HTTPS
- **Validation**: Client-side and server-side validation

### Color Scheme
- **Primary Background**: Obsidian (#0d1b2a)
- **Accent Color**: Sapphire (#00c4cc)
- **Highlight Color**: Gold (#f1c40f)
- **Text Colors**: 
  - Primary: #ffffff
  - Secondary: #b8c5d1
  - Error: #e74c3c
  - Success: #27ae60

### Responsive Breakpoints
- **Mobile**: < 768px
- **Tablet**: 768px - 1024px
- **Desktop**: > 1024px

---

## Authentication Flow

### 1. User Registration Flow
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Access   │───▶│  Registration    │───▶│   Validation    │
│   /signup       │    │     Form         │    │   (Client)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Redirect to   │◀───│   Session        │◀───│   Server        │
│   Main Page     │    │   Creation       │    │   Validation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 2. User Login Flow
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Access   │───▶│    Login Form    │───▶│   Validation    │
│   /login        │    │                  │    │   (Client)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Main Page     │◀───│   Session        │◀───│   Credential    │
│   with Profile  │    │   Management     │    │   Verification  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 3. Session Validation Flow
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Page Access   │───▶│   Check Session  │───▶│   Valid?        │
│                 │    │   Token          │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                        ┌─────────────────┐             ▼
                        │   Redirect to   │◀───No───┌─────────┐
                        │   Login Page    │         │  Valid  │
                        └─────────────────┘         └─────────┘
                                                         │
                                                        Yes
                                                         ▼
                                                ┌─────────────────┐
                                                │   Allow Access  │
                                                │   Load Content  │
                                                └─────────────────┘
```

---

## Security Architecture

### 1. Password Security
```javascript
// Password Requirements
const PASSWORD_REQUIREMENTS = {
    minLength: 8,
    maxLength: 128,
    requireUppercase: true,
    requireLowercase: true,
    requireNumbers: true,
    requireSpecialChars: true,
    preventCommonPasswords: true
};

// Password Validation Function
function validatePassword(password) {
    const validations = {
        length: password.length >= 8 && password.length <= 128,
        uppercase: /[A-Z]/.test(password),
        lowercase: /[a-z]/.test(password),
        numbers: /\d/.test(password),
        specialChars: /[!@#$%^&*(),.?":{}|<>]/.test(password),
        notCommon: !COMMON_PASSWORDS.includes(password.toLowerCase())
    };
    
    return {
        isValid: Object.values(validations).every(v => v),
        validations
    };
}
```

### 2. Input Sanitization
```javascript
// XSS Prevention
function sanitizeInput(input) {
    const div = document.createElement('div');
    div.textContent = input;
    return div.innerHTML;
}

// SQL Injection Prevention (for server-side)
function escapeSQL(input) {
    return input.replace(/["'\\]/g, '\\$&');
}

// Email Validation
function validateEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email) && email.length <= 254;
}
```

### 3. Session Security
```javascript
// Secure Session Token Generation
function generateSecureToken() {
    const array = new Uint8Array(32);
    crypto.getRandomValues(array);
    return Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');
}

// Session Management
class SecureSession {
    constructor() {
        this.tokenKey = 'auth_token';
        this.userKey = 'user_data';
        this.expiryKey = 'session_expiry';
    }
    
    createSession(userData, expiryHours = 24) {
        const token = generateSecureToken();
        const expiry = Date.now() + (expiryHours * 60 * 60 * 1000);
        
        sessionStorage.setItem(this.tokenKey, token);
        sessionStorage.setItem(this.userKey, JSON.stringify(userData));
        sessionStorage.setItem(this.expiryKey, expiry.toString());
        
        return token;
    }
    
    validateSession() {
        const token = sessionStorage.getItem(this.tokenKey);
        const expiry = sessionStorage.getItem(this.expiryKey);
        
        if (!token || !expiry) return false;
        if (Date.now() > parseInt(expiry)) {
            this.destroySession();
            return false;
        }
        
        return true;
    }
    
    destroySession() {
        sessionStorage.removeItem(this.tokenKey);
        sessionStorage.removeItem(this.userKey);
        sessionStorage.removeItem(this.expiryKey);
    }
}
```

---

## User Interface Design

### 1. Login/Signup Form Structure
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Authentication - Fake News Detection</title>
    <link rel="stylesheet" href="auth-styles.css">
</head>
<body>
    <div class="auth-container">
        <div class="auth-card">
            <div class="auth-header">
                <h1 class="auth-title">Fake News Detection</h1>
                <p class="auth-subtitle">Secure Authentication Portal</p>
            </div>
            
            <div class="auth-tabs">
                <button class="tab-btn active" data-tab="login">Login</button>
                <button class="tab-btn" data-tab="signup">Sign Up</button>
            </div>
            
            <!-- Login Form -->
            <form id="loginForm" class="auth-form active">
                <div class="form-group">
                    <label for="loginEmail">Email Address</label>
                    <input type="email" id="loginEmail" name="email" required 
                           autocomplete="email" aria-describedby="emailError">
                    <span class="error-message" id="emailError"></span>
                </div>
                
                <div class="form-group">
                    <label for="loginPassword">Password</label>
                    <div class="password-input">
                        <input type="password" id="loginPassword" name="password" 
                               required autocomplete="current-password" 
                               aria-describedby="passwordError">
                        <button type="button" class="toggle-password" 
                                aria-label="Toggle password visibility">
                            <span class="eye-icon">👁️</span>
                        </button>
                    </div>
                    <span class="error-message" id="passwordError"></span>
                </div>
                
                <div class="form-options">
                    <label class="checkbox-label">
                        <input type="checkbox" id="rememberMe">
                        <span class="checkmark"></span>
                        Remember me
                    </label>
                    <a href="#" class="forgot-password">Forgot Password?</a>
                </div>
                
                <button type="submit" class="auth-btn primary">
                    <span class="btn-text">Sign In</span>
                    <span class="btn-loader hidden">🔄</span>
                </button>
            </form>
            
            <!-- Signup Form -->
            <form id="signupForm" class="auth-form">
                <div class="form-row">
                    <div class="form-group half">
                        <label for="firstName">First Name</label>
                        <input type="text" id="firstName" name="firstName" required 
                               autocomplete="given-name">
                        <span class="error-message"></span>
                    </div>
                    <div class="form-group half">
                        <label for="lastName">Last Name</label>
                        <input type="text" id="lastName" name="lastName" required 
                               autocomplete="family-name">
                        <span class="error-message"></span>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="signupEmail">Email Address</label>
                    <input type="email" id="signupEmail" name="email" required 
                           autocomplete="email">
                    <span class="error-message"></span>
                </div>
                
                <div class="form-group">
                    <label for="signupPassword">Password</label>
                    <div class="password-input">
                        <input type="password" id="signupPassword" name="password" 
                               required autocomplete="new-password">
                        <button type="button" class="toggle-password">
                            <span class="eye-icon">👁️</span>
                        </button>
                    </div>
                    <div class="password-strength">
                        <div class="strength-bar">
                            <div class="strength-fill"></div>
                        </div>
                        <span class="strength-text">Password Strength</span>
                    </div>
                    <span class="error-message"></span>
                </div>
                
                <div class="form-group">
                    <label for="confirmPassword">Confirm Password</label>
                    <input type="password" id="confirmPassword" name="confirmPassword" 
                           required autocomplete="new-password">
                    <span class="error-message"></span>
                </div>
                
                <div class="form-group">
                    <label class="checkbox-label">
                        <input type="checkbox" id="agreeTerms" required>
                        <span class="checkmark"></span>
                        I agree to the <a href="#">Terms of Service</a> and 
                        <a href="#">Privacy Policy</a>
                    </label>
                </div>
                
                <button type="submit" class="auth-btn primary">
                    <span class="btn-text">Create Account</span>
                    <span class="btn-loader hidden">🔄</span>
                </button>
            </form>
        </div>
    </div>
</body>
</html>
```

### 2. CSS Styling (auth-styles.css)
```css
/* Reset and Base Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 100%);
    color: #ffffff;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
}

/* Authentication Container */
.auth-container {
    width: 100%;
    max-width: 450px;
    margin: 0 auto;
}

.auth-card {
    background: rgba(13, 27, 42, 0.95);
    border-radius: 16px;
    padding: 40px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(0, 196, 204, 0.2);
    backdrop-filter: blur(10px);
}

/* Header Styles */
.auth-header {
    text-align: center;
    margin-bottom: 30px;
}

.auth-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00c4cc, #f1c40f);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
}

.auth-subtitle {
    color: #b8c5d1;
    font-size: 0.95rem;
}

/* Tab Navigation */
.auth-tabs {
    display: flex;
    margin-bottom: 30px;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid rgba(0, 196, 204, 0.3);
}

.tab-btn {
    flex: 1;
    padding: 12px 20px;
    background: transparent;
    border: none;
    color: #b8c5d1;
    font-size: 0.95rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
}

.tab-btn.active {
    background: #00c4cc;
    color: #0d1b2a;
    font-weight: 600;
}

.tab-btn:hover:not(.active) {
    background: rgba(0, 196, 204, 0.1);
    color: #00c4cc;
}

/* Form Styles */
.auth-form {
    display: none;
}

.auth-form.active {
    display: block;
}

.form-row {
    display: flex;
    gap: 15px;
}

.form-group {
    margin-bottom: 20px;
}

.form-group.half {
    flex: 1;
}

.form-group label {
    display: block;
    margin-bottom: 6px;
    color: #ffffff;
    font-weight: 500;
    font-size: 0.9rem;
}

.form-group input {
    width: 100%;
    padding: 12px 16px;
    background: rgba(27, 38, 59, 0.8);
    border: 1px solid rgba(0, 196, 204, 0.3);
    border-radius: 8px;
    color: #ffffff;
    font-size: 0.95rem;
    transition: all 0.3s ease;
}

.form-group input:focus {
    outline: none;
    border-color: #00c4cc;
    box-shadow: 0 0 0 3px rgba(0, 196, 204, 0.1);
    background: rgba(27, 38, 59, 1);
}

.form-group input::placeholder {
    color: #6c7b7f;
}

/* Password Input */
.password-input {
    position: relative;
}

.toggle-password {
    position: absolute;
    right: 12px;
    top: 50%;
    transform: translateY(-50%);
    background: none;
    border: none;
    color: #b8c5d1;
    cursor: pointer;
    padding: 4px;
    border-radius: 4px;
    transition: color 0.3s ease;
}

.toggle-password:hover {
    color: #00c4cc;
}

/* Password Strength Indicator */
.password-strength {
    margin-top: 8px;
}

.strength-bar {
    height: 4px;
    background: rgba(27, 38, 59, 0.8);
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 4px;
}

.strength-fill {
    height: 100%;
    width: 0%;
    transition: all 0.3s ease;
    border-radius: 2px;
}

.strength-fill.weak {
    width: 25%;
    background: #e74c3c;
}

.strength-fill.fair {
    width: 50%;
    background: #f39c12;
}

.strength-fill.good {
    width: 75%;
    background: #f1c40f;
}

.strength-fill.strong {
    width: 100%;
    background: #27ae60;
}

.strength-text {
    font-size: 0.8rem;
    color: #b8c5d1;
}

/* Form Options */
.form-options {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 25px;
}

.checkbox-label {
    display: flex;
    align-items: center;
    cursor: pointer;
    font-size: 0.9rem;
    color: #b8c5d1;
}

.checkbox-label input[type="checkbox"] {
    display: none;
}

.checkmark {
    width: 18px;
    height: 18px;
    border: 2px solid rgba(0, 196, 204, 0.5);
    border-radius: 4px;
    margin-right: 8px;
    position: relative;
    transition: all 0.3s ease;
}

.checkbox-label input[type="checkbox"]:checked + .checkmark {
    background: #00c4cc;
    border-color: #00c4cc;
}

.checkbox-label input[type="checkbox"]:checked + .checkmark::after {
    content: '✓';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: #0d1b2a;
    font-weight: bold;
    font-size: 12px;
}

.forgot-password {
    color: #00c4cc;
    text-decoration: none;
    font-size: 0.9rem;
    transition: color 0.3s ease;
}

.forgot-password:hover {
    color: #f1c40f;
}

/* Button Styles */
.auth-btn {
    width: 100%;
    padding: 14px 20px;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.auth-btn.primary {
    background: linear-gradient(135deg, #00c4cc, #0099a3);
    color: #ffffff;
}

.auth-btn.primary:hover {
    background: linear-gradient(135deg, #0099a3, #007a82);
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0, 196, 204, 0.3);
}

.auth-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
}

.btn-loader {
    animation: spin 1s linear infinite;
}

@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.hidden {
    display: none;
}

/* Error Messages */
.error-message {
    display: block;
    color: #e74c3c;
    font-size: 0.8rem;
    margin-top: 4px;
    min-height: 1rem;
}

.form-group.error input {
    border-color: #e74c3c;
    box-shadow: 0 0 0 3px rgba(231, 76, 60, 0.1);
}

/* Success Messages */
.success-message {
    background: rgba(39, 174, 96, 0.1);
    border: 1px solid #27ae60;
    color: #27ae60;
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 20px;
    font-size: 0.9rem;
}

/* Responsive Design */
@media (max-width: 768px) {
    .auth-card {
        padding: 30px 20px;
        margin: 10px;
    }
    
    .auth-title {
        font-size: 1.75rem;
    }
    
    .form-row {
        flex-direction: column;
        gap: 0;
    }
    
    .form-group.half {
        flex: none;
    }
    
    .form-options {
        flex-direction: column;
        gap: 15px;
        align-items: flex-start;
    }
}

@media (max-width: 480px) {
    body {
        padding: 10px;
    }
    
    .auth-card {
        padding: 25px 15px;
    }
    
    .auth-title {
        font-size: 1.5rem;
    }
}

/* Focus Indicators for Accessibility */
.tab-btn:focus,
.auth-btn:focus,
.toggle-password:focus {
    outline: 2px solid #f1c40f;
    outline-offset: 2px;
}

.form-group input:focus {
    outline: 2px solid #f1c40f;
    outline-offset: 2px;
}

/* High Contrast Mode Support */
@media (prefers-contrast: high) {
    .auth-card {
        border: 2px solid #ffffff;
    }
    
    .form-group input {
        border: 2px solid #ffffff;
    }
    
    .auth-btn.primary {
        border: 2px solid #ffffff;
    }
}

/* Reduced Motion Support */
@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
```

### 3. Main Page Header with User Profile
```html
<!-- Main Page Header -->
<header class="main-header">
    <div class="header-container">
        <div class="logo-section">
            <h1 class="app-title">Fake News Detection</h1>
        </div>
        
        <nav class="main-nav">
            <ul class="nav-list">
                <li><a href="#dashboard" class="nav-link">Dashboard</a></li>
                <li><a href="#analyze" class="nav-link">Analyze</a></li>
                <li><a href="#history" class="nav-link">History</a></li>
            </ul>
        </nav>
        
        <div class="user-profile">
            <div class="profile-info">
                <span class="user-name" id="userName">John Doe</span>
                <span class="user-email" id="userEmail">john@example.com</span>
            </div>
            <div class="profile-avatar">
                <img src="#" alt="User Avatar" id="userAvatar" class="avatar-img">
                <div class="avatar-placeholder" id="avatarPlaceholder">
                    <span id="userInitials">JD</span>
                </div>
            </div>
            <div class="profile-dropdown">
                <button class="dropdown-toggle" aria-label="User menu">
                    <span class="dropdown-icon">▼</span>
                </button>
                <div class="dropdown-menu">
                    <a href="#profile" class="dropdown-item">Profile Settings</a>
                    <a href="#preferences" class="dropdown-item">Preferences</a>
                    <hr class="dropdown-divider">
                    <button class="dropdown-item logout-btn" id="logoutBtn">
                        Logout
                    </button>
                </div>
            </div>
        </div>
    </div>
</header>
```

---

## Session Management

### 1. Session Storage Strategy
```javascript
class SessionManager {
    constructor() {
        this.storageType = 'sessionStorage'; // or 'localStorage' for persistent sessions
        this.tokenKey = 'auth_token';
        this.userKey = 'user_data';
        this.expiryKey = 'session_expiry';
        this.refreshKey = 'refresh_token';
    }
    
    // Create new session
    createSession(userData, options = {}) {
        const {
            expiryHours = 24,
            persistent = false,
            refreshToken = null
        } = options;
        
        const token = this.generateSecureToken();
        const expiry = Date.now() + (expiryHours * 60 * 60 * 1000);
        const storage = persistent ? localStorage : sessionStorage;
        
        const sessionData = {
            token,
            user: this.sanitizeUserData(userData),
            expiry,
            created: Date.now(),
            lastActivity: Date.now()
        };
        
        storage.setItem(this.tokenKey, token);
        storage.setItem(this.userKey, JSON.stringify(sessionData.user));
        storage.setItem(this.expiryKey, expiry.toString());
        
        if (refreshToken) {
            storage.setItem(this.refreshKey, refreshToken);
        }
        
        this.startSessionMonitoring();
        return token;
    }
    
    // Validate current session
    validateSession() {
        const token = this.getStorageItem(this.tokenKey);
        const expiry = this.getStorageItem(this.expiryKey);
        const userData = this.getStorageItem(this.userKey);
        
        if (!token || !expiry || !userData) {
            return { valid: false, reason: 'missing_data' };
        }
        
        const expiryTime = parseInt(expiry);
        const currentTime = Date.now();
        
        if (currentTime > expiryTime) {
            this.destroySession();
            return { valid: false, reason: 'expired' };
        }
        
        // Update last activity
        this.updateLastActivity();
        
        return {
            valid: true,
            token,
            user: JSON.parse(userData),
            timeRemaining: expiryTime - currentTime
        };
    }
    
    // Refresh session
    async refreshSession() {
        const refreshToken = this.getStorageItem(this.refreshKey);
        if (!refreshToken) {
            throw new Error('No refresh token available');
        }
        
        try {
            const response = await fetch('/api/auth/refresh', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${refreshToken}`
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to refresh session');
            }
            
            const data = await response.json();
            this.createSession(data.user, {
                expiryHours: 24,
                refreshToken: data.refreshToken
            });
            
            return data;
        } catch (error) {
            this.destroySession();
            throw error;
        }
    }
    
    // Destroy session
    destroySession() {
        [sessionStorage, localStorage].forEach(storage => {
            storage.removeItem(this.tokenKey);
            storage.removeItem(this.userKey);
            storage.removeItem(this.expiryKey);
            storage.removeItem(this.refreshKey);
        });
        
        this.stopSessionMonitoring();
    }
    
    // Session monitoring
    startSessionMonitoring() {
        // Check session every 5 minutes
        this.monitoringInterval = setInterval(() => {
            const session = this.validateSession();
            if (!session.valid) {
                this.handleSessionExpiry(session.reason);
            } else if (session.timeRemaining < 300000) { // 5 minutes
                this.showSessionWarning(session.timeRemaining);
            }
        }, 300000);
        
        // Listen for storage changes (multi-tab support)
        window.addEventListener('storage', this.handleStorageChange.bind(this));
        
        // Listen for user activity
        this.setupActivityListeners();
    }
    
    stopSessionMonitoring() {
        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
        }
        window.removeEventListener('storage', this.handleStorageChange.bind(this));
        this.removeActivityListeners();
    }
    
    // Utility methods
    generateSecureToken() {
        const array = new Uint8Array(32);
        crypto.getRandomValues(array);
        return Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');
    }
    
    sanitizeUserData(userData) {
        const { password, ...safeData } = userData;
        return safeData;
    }
    
    getStorageItem(key) {
        return sessionStorage.getItem(key) || localStorage.getItem(key);
    }
    
    updateLastActivity() {
        const userData = this.getStorageItem(this.userKey);
        if (userData) {
            const user = JSON.parse(userData);
            user.lastActivity = Date.now();
            const storage = sessionStorage.getItem(this.userKey) ? sessionStorage : localStorage;
            storage.setItem(this.userKey, JSON.stringify(user));
        }
    }
}
```

---

## Implementation Guidelines

### 1. Authentication JavaScript (auth.js)
```javascript
// Authentication Manager
class AuthenticationManager {
    constructor() {
        this.sessionManager = new SessionManager();
        this.apiBaseUrl = '/api/auth';
        this.rateLimiter = new RateLimiter();
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.checkAuthStatus();
    }
    
    setupEventListeners() {
        // Form submissions
        document.getElementById('loginForm')?.addEventListener('submit', this.handleLogin.bind(this));
        document.getElementById('signupForm')?.addEventListener('submit', this.handleSignup.bind(this));
        
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', this.switchTab.bind(this));
        });
        
        // Password visibility toggle
        document.querySelectorAll('.toggle-password').forEach(btn => {
            btn.addEventListener('click', this.togglePasswordVisibility.bind(this));
        });
        
        // Password strength checking
        document.getElementById('signupPassword')?.addEventListener('input', this.checkPasswordStrength.bind(this));
        
        // Real-time validation
        this.setupRealTimeValidation();
    }
    
    async handleLogin(event) {
        event.preventDefault();
        
        const form = event.target;
        const formData = new FormData(form);
        const credentials = {
            email: formData.get('email'),
            password: formData.get('password'),
            rememberMe: formData.get('rememberMe') === 'on'
        };
        
        // Rate limiting check
        if (!this.rateLimiter.canAttempt('login')) {
            this.showError('Too many login attempts. Please try again later.');
            return;
        }
        
        // Client-side validation
        const validation = this.validateLoginForm(credentials);
        if (!validation.isValid) {
            this.displayValidationErrors(validation.errors);
            return;
        }
        
        this.setLoadingState(form, true);
        
        try {
            const response = await this.makeAuthRequest('/login', credentials);
            
            if (response.success) {
                this.sessionManager.createSession(response.user, {
                    expiryHours: credentials.rememberMe ? 168 : 24, // 7 days or 24 hours
                    persistent: credentials.rememberMe,
                    refreshToken: response.refreshToken
                });
                
                this.redirectToMain();
            } else {
                this.showError(response.message || 'Login failed');
                this.rateLimiter.recordAttempt('login');
            }
        } catch (error) {
            this.showError('Network error. Please try again.');
            console.error('Login error:', error);
        } finally {
            this.setLoadingState(form, false);
        }
    }
    
    async handleSignup(event) {
        event.preventDefault();
        
        const form = event.target;
        const formData = new FormData(form);
        const userData = {
            firstName: formData.get('firstName'),
            lastName: formData.get('lastName'),
            email: formData.get('email'),
            password: formData.get('password'),
            confirmPassword: formData.get('confirmPassword')
        };
        
        // Client-side validation
        const validation = this.validateSignupForm(userData);
        if (!validation.isValid) {
            this.displayValidationErrors(validation.errors);
            return;
        }
        
        this.setLoadingState(form, true);
        
        try {
            const response = await this.makeAuthRequest('/register', userData);
            
            if (response.success) {
                this.showSuccess('Account created successfully! Please log in.');
                this.switchTab({ target: { dataset: { tab: 'login' } } });
            } else {
                this.showError(response.message || 'Registration failed');
            }
        } catch (error) {
            this.showError('Network error. Please try again.');
            console.error('Signup error:', error);
        } finally {
            this.setLoadingState(form, false);
        }
    }
    
    validateLoginForm(credentials) {
        const errors = {};
        
        if (!this.validateEmail(credentials.email)) {
            errors.email = 'Please enter a valid email address';
        }
        
        if (!credentials.password || credentials.password.length < 1) {
            errors.password = 'Password is required';
        }
        
        return {
            isValid: Object.keys(errors).length === 0,
            errors
        };
    }
    
    validateSignupForm(userData) {
        const errors = {};
        
        // Name validation
        if (!userData.firstName || userData.firstName.trim().length < 2) {
            errors.firstName = 'First name must be at least 2 characters';
        }
        
        if (!userData.lastName || userData.lastName.trim().length < 2) {
            errors.lastName = 'Last name must be at least 2 characters';
        }
        
        // Email validation
        if (!this.validateEmail(userData.email)) {
            errors.email = 'Please enter a valid email address';
        }
        
        // Password validation
        const passwordValidation = this.validatePassword(userData.password);
        if (!passwordValidation.isValid) {
            errors.password = 'Password does not meet requirements';
        }
        
        // Confirm password
        if (userData.password !== userData.confirmPassword) {
            errors.confirmPassword = 'Passwords do not match';
        }
        
        return {
            isValid: Object.keys(errors).length === 0,
            errors
        };
    }
    
    async makeAuthRequest(endpoint, data) {
        const response = await fetch(`${this.apiBaseUrl}${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
    
    checkAuthStatus() {
        const session = this.sessionManager.validateSession();
        if (session.valid && window.location.pathname === '/login') {
            this.redirectToMain();
        } else if (!session.valid && window.location.pathname !== '/login') {
            this.redirectToLogin();
        }
    }
    
    redirectToMain() {
        window.location.href = '/main';
    }
    
    redirectToLogin() {
        window.location.href = '/login';
    }
    
    // Utility methods for UI management
    switchTab(event) {
        const tabName = event.target.dataset.tab;
        
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        event.target.classList.add('active');
        
        // Update forms
        document.querySelectorAll('.auth-form').forEach(form => {
            form.classList.remove('active');
        });
        document.getElementById(`${tabName}Form`).classList.add('active');
    }
    
    togglePasswordVisibility(event) {
        const button = event.target.closest('.toggle-password');
        const input = button.parentElement.querySelector('input');
        const icon = button.querySelector('.eye-icon');
        
        if (input.type === 'password') {
            input.type = 'text';
            icon.textContent = '🙈';
        } else {
            input.type = 'password';
            icon.textContent = '👁️';
        }
    }
    
    checkPasswordStrength(event) {
        const password = event.target.value;
        const strengthBar = document.querySelector('.strength-fill');
        const strengthText = document.querySelector('.strength-text');
        
        const strength = this.calculatePasswordStrength(password);
        
        strengthBar.className = `strength-fill ${strength.level}`;
        strengthText.textContent = `Password Strength: ${strength.text}`;
    }
    
    calculatePasswordStrength(password) {
        let score = 0;
        
        if (password.length >= 8) score++;
        if (password.length >= 12) score++;
        if (/[a-z]/.test(password)) score++;
        if (/[A-Z]/.test(password)) score++;
        if (/\d/.test(password)) score++;
        if (/[^\w\s]/.test(password)) score++;
        
        const levels = {
            0: { level: 'weak', text: 'Very Weak' },
            1: { level: 'weak', text: 'Weak' },
            2: { level: 'weak', text: 'Weak' },
            3: { level: 'fair', text: 'Fair' },
            4: { level: 'good', text: 'Good' },
            5: { level: 'strong', text: 'Strong' },
            6: { level: 'strong', text: 'Very Strong' }
        };
        
        return levels[score] || levels[0];
    }
}

// Rate Limiter for login attempts
class RateLimiter {
    constructor() {
        this.attempts = new Map();
        this.maxAttempts = 5;
        this.windowMs = 15 * 60 * 1000; // 15 minutes
    }
    
    canAttempt(key) {
        const now = Date.now();
        const attempts = this.attempts.get(key) || [];
        
        // Remove old attempts
        const validAttempts = attempts.filter(time => now - time < this.windowMs);
        this.attempts.set(key, validAttempts);
        
        return validAttempts.length < this.maxAttempts;
    }
    
    recordAttempt(key) {
        const attempts = this.attempts.get(key) || [];
        attempts.push(Date.now());
        this.attempts.set(key, attempts);
    }
}

// Initialize authentication when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AuthenticationManager();
});
```

### 2. Main Page User Profile Management
```javascript
// User Profile Manager
class UserProfileManager {
    constructor() {
        this.sessionManager = new SessionManager();
        this.init();
    }
    
    init() {
        this.loadUserProfile();
        this.setupEventListeners();
    }
    
    loadUserProfile() {
        const session = this.sessionManager.validateSession();
        if (!session.valid) {
            window.location.href = '/login';
            return;
        }
        
        const user = session.user;
        this.displayUserInfo(user);
    }
    
    displayUserInfo(user) {
        // Update user name and email
        document.getElementById('userName').textContent = `${user.firstName} ${user.lastName}`;
        document.getElementById('userEmail').textContent = user.email;
        
        // Update user initials
        const initials = `${user.firstName.charAt(0)}${user.lastName.charAt(0)}`.toUpperCase();
        document.getElementById('userInitials').textContent = initials;
        
        // Handle avatar if available
        if (user.avatar) {
            document.getElementById('userAvatar').src = user.avatar;
            document.getElementById('userAvatar').style.display = 'block';
            document.getElementById('avatarPlaceholder').style.display = 'none';
        } else {
            document.getElementById('userAvatar').style.display = 'none';
            document.getElementById('avatarPlaceholder').style.display = 'flex';
        }
    }
    
    setupEventListeners() {
        // Logout button
        document.getElementById('logoutBtn').addEventListener('click', this.handleLogout.bind(this));
        
        // Dropdown toggle
        document.querySelector('.dropdown-toggle').addEventListener('click', this.toggleDropdown.bind(this));
        
        // Close dropdown when clicking outside
        document.addEventListener('click', (event) => {
            if (!event.target.closest('.user-profile')) {
                this.closeDropdown();
            }
        });
    }
    
    toggleDropdown() {
        const dropdown = document.querySelector('.dropdown-menu');
        dropdown.classList.toggle('show');
    }
    
    closeDropdown() {
        document.querySelector('.dropdown-menu').classList.remove('show');
    }
    
    async handleLogout() {
        try {
            // Call logout API
            await fetch('/api/auth/logout', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.sessionManager.getStorageItem('auth_token')}`
                }
            });
        } catch (error) {
            console.error('Logout API error:', error);
        } finally {
            // Always destroy local session
            this.sessionManager.destroySession();
            window.location.href = '/login';
        }
    }
}

// Initialize profile manager
document.addEventListener('DOMContentLoaded', () => {
    new UserProfileManager();
});
```

---

## Security Validation Procedures

### 1. Input Validation and Sanitization
```javascript
// Security Validator
class SecurityValidator {
    static validateEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email) && email.length <= 254;
    }
    
    static validatePassword(password) {
        const requirements = {
            minLength: password.length >= 8,
            maxLength: password.length <= 128,
            hasUppercase: /[A-Z]/.test(password),
            hasLowercase: /[a-z]/.test(password),
            hasNumbers: /\d/.test(password),
            hasSpecialChars: /[!@#$%^&*(),.?":{}|<>]/.test(password),
            notCommon: !this.isCommonPassword(password)
        };
        
        return {
            isValid: Object.values(requirements).every(req => req),
            requirements
        };
    }
    
    static sanitizeInput(input) {
        if (typeof input !== 'string') return input;
        
        return input
            .replace(/[<>"'&]/g, (match) => {
                const entities = {
                    '<': '&lt;',
                    '>': '&gt;',
                    '"': '&quot;',
                    "'": '&#x27;',
                    '&': '&amp;'
                };
                return entities[match];
            })
            .trim();
    }
    
    static validateName(name) {
        const nameRegex = /^[a-zA-Z\s'-]{2,50}$/;
        return nameRegex.test(name.trim());
    }
    
    static isCommonPassword(password) {
        const commonPasswords = [
            'password', '123456', '123456789', 'qwerty', 'abc123',
            'password123', 'admin', 'letmein', 'welcome', 'monkey'
        ];
        return commonPasswords.includes(password.toLowerCase());
    }
    
    static validateCSRFToken(token) {
        // Implement CSRF token validation
        const storedToken = sessionStorage.getItem('csrf_token');
        return token === storedToken;
    }
    
    static checkRateLimit(identifier, maxAttempts = 5, windowMs = 900000) {
        const key = `rate_limit_${identifier}`;
        const attempts = JSON.parse(localStorage.getItem(key) || '[]');
        const now = Date.now();
        
        // Remove old attempts
        const validAttempts = attempts.filter(time => now - time < windowMs);
        
        if (validAttempts.length >= maxAttempts) {
            return {
                allowed: false,
                resetTime: validAttempts[0] + windowMs
            };
        }
        
        validAttempts.push(now);
        localStorage.setItem(key, JSON.stringify(validAttempts));
        
        return { allowed: true };
    }
}
```

### 2. XSS Protection
```javascript
// XSS Protection Utilities
class XSSProtection {
    static escapeHTML(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
    
    static sanitizeURL(url) {
        try {
            const urlObj = new URL(url);
            // Only allow http and https protocols
            if (!['http:', 'https:'].includes(urlObj.protocol)) {
                return '#';
            }
            return urlObj.href;
        } catch {
            return '#';
        }
    }
    
    static createSafeElement(tagName, attributes = {}, textContent = '') {
        const element = document.createElement(tagName);
        
        // Safely set attributes
        Object.entries(attributes).forEach(([key, value]) => {
            if (key === 'href') {
                element.setAttribute(key, this.sanitizeURL(value));
            } else {
                element.setAttribute(key, this.escapeHTML(value));
            }
        });
        
        // Safely set text content
        element.textContent = textContent;
        
        return element;
    }
    
    static setupCSP() {
        // Content Security Policy headers should be set server-side
        // This is a client-side fallback for additional protection
        const meta = document.createElement('meta');
        meta.httpEquiv = 'Content-Security-Policy';
        meta.content = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self';";
        document.head.appendChild(meta);
    }
}
```

---

## Responsive Design Specifications

### 1. Mobile-First CSS Architecture
```css
/* Mobile-First Responsive Design */

/* Base styles (Mobile < 768px) */
.auth-container {
    padding: 15px;
    width: 100%;
}

.auth-card {
    padding: 25px 20px;
    border-radius: 12px;
}

.auth-title {
    font-size: 1.5rem;
    margin-bottom: 6px;
}

.form-row {
    flex-direction: column;
    gap: 0;
}

.form-group.half {
    flex: none;
    margin-bottom: 20px;
}

.form-options {
    flex-direction: column;
    gap: 12px;
    align-items: flex-start;
}

.main-header {
    padding: 15px;
    flex-direction: column;
    gap: 15px;
}

.main-nav {
    order: 3;
    width: 100%;
}

.nav-list {
    flex-direction: column;
    gap: 10px;
}

.user-profile {
    order: 2;
    justify-content: center;
}

/* Tablet styles (768px - 1024px) */
@media (min-width: 768px) {
    .auth-container {
        padding: 30px;
        max-width: 500px;
    }
    
    .auth-card {
        padding: 40px 35px;
        border-radius: 16px;
    }
    
    .auth-title {
        font-size: 1.75rem;
        margin-bottom: 8px;
    }
    
    .form-row {
        flex-direction: row;
        gap: 15px;
    }
    
    .form-group.half {
        flex: 1;
        margin-bottom: 20px;
    }
    
    .form-options {
        flex-direction: row;
        justify-content: space-between;
        align-items: center;
    }
    
    .main-header {
        flex-direction: row;
        padding: 20px 30px;
        align-items: center;
    }
    
    .main-nav {
        order: 2;
        width: auto;
    }
    
    .nav-list {
        flex-direction: row;
        gap: 25px;
    }
    
    .user-profile {
        order: 3;
        justify-content: flex-end;
    }
}

/* Desktop styles (> 1024px) */
@media (min-width: 1024px) {
    .auth-container {
        max-width: 450px;
        padding: 40px;
    }
    
    .auth-card {
        padding: 50px 40px;
    }
    
    .auth-title {
        font-size: 2rem;
    }
    
    .main-header {
        padding: 25px 40px;
    }
    
    .nav-list {
        gap: 30px;
    }
}

/* Large Desktop styles (> 1440px) */
@media (min-width: 1440px) {
    .auth-container {
        max-width: 500px;
    }
    
    .main-header {
        padding: 30px 60px;
    }
}

/* Touch-friendly interactions */
@media (hover: none) and (pointer: coarse) {
    .auth-btn,
    .tab-btn,
    .nav-link {
        min-height: 44px;
        padding: 12px 20px;
    }
    
    .toggle-password {
        min-width: 44px;
        min-height: 44px;
    }
    
    .dropdown-toggle {
        min-width: 44px;
        min-height: 44px;
    }
}

/* High DPI displays */
@media (-webkit-min-device-pixel-ratio: 2), (min-resolution: 192dpi) {
    .auth-card {
        border-width: 0.5px;
    }
    
    .form-group input {
        border-width: 0.5px;
    }
}

/* Landscape orientation on mobile */
@media (max-width: 768px) and (orientation: landscape) {
    .auth-card {
        padding: 20px;
        max-height: 90vh;
        overflow-y: auto;
    }
    
    .auth-title {
        font-size: 1.25rem;
        margin-bottom: 15px;
    }
    
    .form-group {
        margin-bottom: 15px;
    }
}
```

### 2. Responsive JavaScript Utilities
```javascript
// Responsive Utilities
class ResponsiveManager {
    constructor() {
        this.breakpoints = {
            mobile: 768,
            tablet: 1024,
            desktop: 1440
        };
        
        this.currentBreakpoint = this.getCurrentBreakpoint();
        this.setupResizeListener();
    }
    
    getCurrentBreakpoint() {
        const width = window.innerWidth;
        
        if (width < this.breakpoints.mobile) return 'mobile';
        if (width < this.breakpoints.tablet) return 'tablet';
        if (width < this.breakpoints.desktop) return 'desktop';
        return 'large-desktop';
    }
    
    setupResizeListener() {
        let resizeTimer;
        
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(() => {
                const newBreakpoint = this.getCurrentBreakpoint();
                if (newBreakpoint !== this.currentBreakpoint) {
                    this.handleBreakpointChange(this.currentBreakpoint, newBreakpoint);
                    this.currentBreakpoint = newBreakpoint;
                }
            }, 250);
        });
    }
    
    handleBreakpointChange(oldBreakpoint, newBreakpoint) {
        document.body.classList.remove(`bp-${oldBreakpoint}`);
        document.body.classList.add(`bp-${newBreakpoint}`);
        
        // Trigger custom event
        window.dispatchEvent(new CustomEvent('breakpointChange', {
            detail: { oldBreakpoint, newBreakpoint }
        }));
    }
    
    isMobile() {
        return this.currentBreakpoint === 'mobile';
    }
    
    isTablet() {
        return this.currentBreakpoint === 'tablet';
    }
    
    isDesktop() {
        return ['desktop', 'large-desktop'].includes(this.currentBreakpoint);
    }
    
    adaptFormLayout() {
        const forms = document.querySelectorAll('.auth-form');
        forms.forEach(form => {
            if (this.isMobile()) {
                form.classList.add('mobile-layout');
            } else {
                form.classList.remove('mobile-layout');
            }
        });
    }
}
```

---

## Error Handling

### 1. User-Friendly Error Messages
```javascript
// Error Handler
class ErrorHandler {
    static displayError(message, type = 'error', duration = 5000) {
        const errorContainer = this.createErrorContainer();
        const errorElement = this.createErrorElement(message, type);
        
        errorContainer.appendChild(errorElement);
        
        // Auto-remove after duration
        setTimeout(() => {
            if (errorElement.parentNode) {
                errorElement.remove();
            }
        }, duration);
        
        return errorElement;
    }
    
    static createErrorContainer() {
        let container = document.getElementById('error-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'error-container';
            container.className = 'error-container';
            document.body.appendChild(container);
        }
        return container;
    }
    
    static createErrorElement(message, type) {
        const element = document.createElement('div');
        element.className = `alert alert-${type}`;
        element.innerHTML = `
            <span class="alert-icon">${this.getIcon(type)}</span>
            <span class="alert-message">${this.escapeHTML(message)}</span>
            <button class="alert-close" onclick="this.parentElement.remove()">&times;</button>
        `;
        return element;
    }
    
    static getIcon(type) {
        const icons = {
            error: '❌',
            warning: '⚠️',
            success: '✅',
            info: 'ℹ️'
        };
        return icons[type] || icons.info;
    }
    
    static escapeHTML(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
    
    static handleAuthError(error) {
        const errorMessages = {
            'invalid_credentials': 'Invalid email or password. Please try again.',
            'account_locked': 'Account temporarily locked due to multiple failed attempts.',
            'email_not_verified': 'Please verify your email address before logging in.',
            'session_expired': 'Your session has expired. Please log in again.',
            'network_error': 'Network connection error. Please check your internet connection.',
            'server_error': 'Server error occurred. Please try again later.',
            'rate_limit_exceeded': 'Too many requests. Please wait before trying again.'
        };
        
        const message = errorMessages[error.code] || error.message || 'An unexpected error occurred.';
        this.displayError(message, 'error');
    }
}
```

### 2. Form Validation Error Display
```css
/* Error Styling */
.error-container {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 10000;
    max-width: 400px;
}

.alert {
    display: flex;
    align-items: center;
    padding: 12px 16px;
    margin-bottom: 10px;
    border-radius: 8px;
    font-size: 0.9rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    animation: slideIn 0.3s ease-out;
}

.alert-error {
    background: rgba(231, 76, 60, 0.1);
    border: 1px solid #e74c3c;
    color: #e74c3c;
}

.alert-success {
    background: rgba(39, 174, 96, 0.1);
    border: 1px solid #27ae60;
    color: #27ae60;
}

.alert-warning {
    background: rgba(243, 156, 18, 0.1);
    border: 1px solid #f39c12;
    color: #f39c12;
}

.alert-icon {
    margin-right: 8px;
    font-size: 1.1rem;
}

.alert-message {
    flex: 1;
}

.alert-close {
    background: none;
    border: none;
    color: inherit;
    font-size: 1.2rem;
    cursor: pointer;
    padding: 0 4px;
    margin-left: 8px;
}

@keyframes slideIn {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

/* Form field errors */
.form-group.error input {
    border-color: #e74c3c;
    box-shadow: 0 0 0 3px rgba(231, 76, 60, 0.1);
}

.error-message {
    display: block;
    color: #e74c3c;
    font-size: 0.8rem;
    margin-top: 4px;
    min-height: 1rem;
}

.error-message:empty {
    display: none;
}
```

---

## Accessibility Considerations

### 1. WCAG 2.1 Compliance
```html
<!-- Accessible Form Structure -->
<form id="loginForm" class="auth-form" role="form" aria-labelledby="login-heading">
    <h2 id="login-heading" class="sr-only">Login Form</h2>
    
    <div class="form-group">
        <label for="loginEmail" class="required">
            Email Address
            <span class="sr-only">required</span>
        </label>
        <input 
            type="email" 
            id="loginEmail" 
            name="email" 
            required 
            autocomplete="email"
            aria-describedby="emailError emailHelp"
            aria-invalid="false"
        >
        <div id="emailHelp" class="help-text">Enter your registered email address</div>
        <span class="error-message" id="emailError" role="alert" aria-live="polite"></span>
    </div>
    
    <div class="form-group">
        <label for="loginPassword" class="required">
            Password
            <span class="sr-only">required</span>
        </label>
        <div class="password-input">
            <input 
                type="password" 
                id="loginPassword" 
                name="password" 
                required 
                autocomplete="current-password"
                aria-describedby="passwordError passwordHelp"
                aria-invalid="false"
            >
            <button 
                type="button" 
                class="toggle-password" 
                aria-label="Show password"
                aria-pressed="false"
            >
                <span class="eye-icon" aria-hidden="true">👁️</span>
            </button>
        </div>
        <div id="passwordHelp" class="help-text">Enter your account password</div>
        <span class="error-message" id="passwordError" role="alert" aria-live="polite"></span>
    </div>
    
    <button type="submit" class="auth-btn primary" aria-describedby="loginStatus">
        <span class="btn-text">Sign In</span>
        <span class="btn-loader hidden" aria-hidden="true">🔄</span>
    </button>
    
    <div id="loginStatus" class="sr-only" aria-live="polite" aria-atomic="true"></div>
</form>
```

### 2. Accessibility CSS
```css
/* Screen Reader Only Content */
.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
}

/* Focus Indicators */
*:focus {
    outline: 2px solid #f1c40f;
    outline-offset: 2px;
}

/* High Contrast Mode */
@media (prefers-contrast: high) {
    .auth-card {
        border: 3px solid #ffffff;
        background: #000000;
    }
    
    .form-group input {
        border: 2px solid #ffffff;
        background: #000000;
        color: #ffffff;
    }
    
    .auth-btn.primary {
        background: #ffffff;
        color: #000000;
        border: 2px solid #ffffff;
    }
}

/* Reduced Motion */
@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

/* Color Contrast Requirements */
.auth-title {
    /* Ensure 4.5:1 contrast ratio */
    color: #ffffff;
    background: transparent;
}

.form-group label {
    /* Ensure 4.5:1 contrast ratio */
    color: #ffffff;
    font-weight: 600;
}

.error-message {
    /* Ensure 4.5:1 contrast ratio for errors */
    color: #ff6b6b;
    font-weight: 500;
}

/* Help Text */
.help-text {
    font-size: 0.8rem;
    color: #b8c5d1;
    margin-top: 4px;
}

.required::after {
    content: ' *';
    color: #e74c3c;
    font-weight: bold;
}
```

### 3. Keyboard Navigation
```javascript
// Keyboard Navigation Manager
class KeyboardNavigationManager {
    constructor() {
        this.setupKeyboardHandlers();
        this.setupFocusManagement();
    }
    
    setupKeyboardHandlers() {
        document.addEventListener('keydown', (event) => {
            switch (event.key) {
                case 'Escape':
                    this.handleEscape(event);
                    break;
                case 'Tab':
                    this.handleTab(event);
                    break;
                case 'Enter':
                    this.handleEnter(event);
                    break;
            }
        });
    }
    
    handleEscape(event) {
        // Close dropdowns, modals, etc.
        const openDropdown = document.querySelector('.dropdown-menu.show');
        if (openDropdown) {
            openDropdown.classList.remove('show');
            event.preventDefault();
        }
    }
    
    handleTab(event) {
        // Ensure proper tab order
        const focusableElements = this.getFocusableElements();
        const currentIndex = focusableElements.indexOf(document.activeElement);
        
        if (event.shiftKey) {
            // Shift+Tab (backward)
            if (currentIndex === 0) {
                focusableElements[focusableElements.length - 1].focus();
                event.preventDefault();
            }
        } else {
            // Tab (forward)
            if (currentIndex === focusableElements.length - 1) {
                focusableElements[0].focus();
                event.preventDefault();
            }
        }
    }
    
    handleEnter(event) {
        // Handle Enter key on buttons and links
        if (event.target.matches('button, [role="button"]')) {
            event.target.click();
        }
    }
    
    getFocusableElements() {
        const selector = 'input, button, select, textarea, a[href], [tabindex]:not([tabindex="-1"])';
        return Array.from(document.querySelectorAll(selector))
            .filter(el => !el.disabled && el.offsetParent !== null);
    }
    
    setupFocusManagement() {
        // Announce form errors to screen readers
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'childList') {
                    mutation.addedNodes.forEach((node) => {
                        if (node.classList && node.classList.contains('error-message')) {
                            this.announceError(node.textContent);
                        }
                    });
                }
            });
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }
    
    announceError(message) {
        const announcement = document.createElement('div');
        announcement.setAttribute('aria-live', 'assertive');
        announcement.setAttribute('aria-atomic', 'true');
        announcement.className = 'sr-only';
        announcement.textContent = `Error: ${message}`;
        
        document.body.appendChild(announcement);
        
        setTimeout(() => {
            document.body.removeChild(announcement);
        }, 1000);
    }
}
```

---

## Implementation Checklist

### Security Implementation
- [ ] HTTPS enforcement
- [ ] Input validation (client and server-side)
- [ ] XSS protection
- [ ] CSRF protection
- [ ] Rate limiting
- [ ] Secure session management
- [ ] Password strength requirements
- [ ] SQL injection prevention
- [ ] Content Security Policy
- [ ] Secure headers implementation

### Authentication Features
- [ ] User registration
- [ ] User login
- [ ] Session validation
- [ ] Password reset
- [ ] Remember me functionality
- [ ] Logout functionality
- [ ] Multi-tab session sync
- [ ] Session timeout warnings
- [ ] Account lockout protection

### User Interface
- [ ] Responsive design (mobile, tablet, desktop)
- [ ] Accessible forms (WCAG 2.1)
- [ ] Loading states
- [ ] Error handling
- [ ] Success feedback
- [ ] Password strength indicator
- [ ] Form validation feedback
- [ ] User profile display
- [ ] Dropdown menus

### Code Quality
- [ ] ES6+ JavaScript
- [ ] Modular architecture
- [ ] Error handling
- [ ] Code documentation
- [ ] Performance optimization
- [ ] Browser compatibility
- [ ] Testing implementation
- [ ] Code minification
- [ ] Asset optimization

---

## Deployment Considerations

### 1. Environment Configuration
```javascript
// config/environment.js
const config = {
    development: {
        apiUrl: 'http://localhost:3000/api',
        sessionTimeout: 24 * 60 * 60 * 1000, // 24 hours
        enableDebug: true,
        csrfProtection: false
    },
    production: {
        apiUrl: 'https://api.fakenewsdetection.com',
        sessionTimeout: 2 * 60 * 60 * 1000, // 2 hours
        enableDebug: false,
        csrfProtection: true
    }
};

const environment = process.env.NODE_ENV || 'development';
export default config[environment];
```

### 2. Security Headers
```javascript
// Security headers to implement server-side
const securityHeaders = {
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block',
    'Referrer-Policy': 'strict-origin-when-cross-origin',
    'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:;"
};
```

### 3. Performance Optimization
```javascript
// Lazy loading for non-critical components
const lazyLoadComponents = {
    passwordStrengthChecker: () => import('./components/PasswordStrengthChecker.js'),
    profileManager: () => import('./components/ProfileManager.js'),
    sessionMonitor: () => import('./components/SessionMonitor.js')
};

// Service Worker for caching
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js')
        .then(registration => console.log('SW registered:', registration))
        .catch(error => console.log('SW registration failed:', error));
}
```

---

## Conclusion

This comprehensive authentication architecture provides:

1. **Security-First Design**: Multiple layers of protection against common web vulnerabilities
2. **User-Centric Experience**: Intuitive interfaces with clear feedback and error handling
3. **Accessibility Compliance**: WCAG 2.1 standards for inclusive design
4. **Responsive Implementation**: Mobile-first approach with progressive enhancement
5. **Scalable Architecture**: Modular design for easy maintenance and extension
6. **Performance Optimization**: Efficient loading and resource management

The implementation follows modern web standards and best practices while maintaining the specified color scheme and design requirements. All components are designed to work seamlessly together while providing robust security and excellent user experience across all devices and accessibility needs.
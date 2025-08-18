// DOM Elements
document.addEventListener('DOMContentLoaded', function() {
    // Tab Functionality
    initTabs();
    
    // Theme Toggle
    initThemeToggle();
    
    // Mobile Sidebar Toggle
    initMobileSidebar();
    
    // File Upload Preview
    initFileUpload();
    
    // Visualization Tabs initialization removed
    
    // Initialize Animations
    initAnimations();
    
    // Initialize Mock Data
    initMockData();
});

// Initialize Tab Functionality
function initTabs() {
    const tabBtns = document.querySelectorAll('.input-tabs .tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active class from all buttons and contents
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            // Add active class to clicked button and corresponding content
            btn.classList.add('active');
            const target = btn.getAttribute('data-target');
            document.getElementById(target).classList.add('active');
        });
    });
}

// Initialize Theme Toggle
function initThemeToggle() {
    const themeToggle = document.querySelector('.theme-toggle');
    const body = document.body;
    
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'light') {
        body.classList.add('light-theme');
        themeToggle.classList.remove('dark');
    } else {
        body.classList.remove('light-theme');
        themeToggle.classList.add('dark');
    }
    
    themeToggle.addEventListener('click', () => {
        body.classList.toggle('light-theme');
        themeToggle.classList.toggle('dark');
        
        // Save theme preference
        const currentTheme = body.classList.contains('light-theme') ? 'light' : 'dark';
        localStorage.setItem('theme', currentTheme);
    });
}

// Initialize Mobile Sidebar Toggle
function initMobileSidebar() {
    const mobileToggle = document.querySelector('.mobile-toggle');
    const sidebar = document.querySelector('.sidebar');
    
    mobileToggle.addEventListener('click', () => {
        sidebar.classList.toggle('expanded');
    });
    
    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 768 && 
            !sidebar.contains(e.target) && 
            !mobileToggle.contains(e.target) && 
            sidebar.classList.contains('expanded')) {
            sidebar.classList.remove('expanded');
        }
    });
}

// Initialize File Upload Preview
function initFileUpload() {
    const fileInput = document.querySelector('.file-upload input');
    const previewContainer = document.querySelector('.preview-container');
    const previewImage = document.querySelector('.preview-container img');
    
    if (fileInput && previewContainer && previewImage) {
        fileInput.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    previewImage.src = e.target.result;
                    previewContainer.style.display = 'block';
                }
                
                reader.readAsDataURL(this.files[0]);
            }
        });
    }
}

// Visualization Tabs function removed

// Initialize Animations
function initAnimations() {
    // Add entrance animations to cards
    const cards = document.querySelectorAll('.result-card, .metrics-card, .live-feed-card, .history-card');
    
    cards.forEach((card, index) => {
        setTimeout(() => {
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, 100 * index);
    });
    
    // Add floating animation to certain elements
    document.querySelectorAll('.floating-element').forEach(el => {
        el.classList.add('floating');
    });
    
    // Add glow effect to important buttons
    document.querySelector('.detect-btn').classList.add('glow');
}

// Initialize Mock Data for demonstration
function initMockData() {
    // Set confidence meter width
    const meterFill = document.querySelector('.meter-fill');
    if (meterFill) {
        meterFill.style.width = '78%';
    }
    
    // Token highlighting removed as per requirements
    
    // Initialize charts if Chart.js is available
    if (typeof Chart !== 'undefined') {
        initCharts();
    }
    
    // Add ripple effect to buttons
    document.querySelectorAll('button').forEach(button => {
        button.classList.add('ripple');
    });
}

// Token highlighting function removed as per requirements

// Initialize Charts for metrics and visualizations
function initCharts() {
    // Performance metrics chart
    const metricsCtx = document.getElementById('metrics-chart');
    if (metricsCtx) {
        new Chart(metricsCtx, {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                datasets: [{
                    label: 'Accuracy',
                    data: [92, 93, 94, 93, 95, 97],
                    borderColor: '#4caf50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    tension: 0.4,
                    fill: true
                }, {
                    label: 'Precision',
                    data: [88, 90, 89, 91, 92, 94],
                    borderColor: '#2196f3',
                    backgroundColor: 'rgba(33, 150, 243, 0.1)',
                    tension: 0.4,
                    fill: true
                }, {
                    label: 'Recall',
                    data: [85, 87, 86, 88, 90, 93],
                    borderColor: '#ff9800',
                    backgroundColor: 'rgba(255, 152, 0, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#f0f0f0'
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 80,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#f0f0f0'
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#f0f0f0'
                        }
                    }
                }
            }
        });
    }
    
    // Topic modeling chart removed as the section has been removed from the UI
}

// Detect Button Click Handler
document.addEventListener('click', function(e) {
    if (e.target.classList.contains('detect-btn') || e.target.closest('.detect-btn')) {
        // Show loading state
        const resultCard = document.querySelector('.result-card');
        if (resultCard) {
            resultCard.classList.add('loading');
            
            // Simulate API call delay
            setTimeout(() => {
                resultCard.classList.remove('loading');
                
                // Update prediction result (for demo purposes)
                const predictionBadge = document.querySelector('.prediction-badge');
                if (predictionBadge) {
                    // Randomly choose between real, fake, or uncertain
                    const outcomes = ['real', 'fake', 'uncertain'];
                    const outcome = outcomes[Math.floor(Math.random() * outcomes.length)];
                    
                    // Remove all classes and add the new one
                    predictionBadge.classList.remove('real', 'fake', 'uncertain');
                    predictionBadge.classList.add(outcome);
                    
                    // Update text
                    predictionBadge.innerHTML = `<i class="fas fa-${outcome === 'real' ? 'check-circle' : outcome === 'fake' ? 'times-circle' : 'question-circle'}"></i> ${outcome.charAt(0).toUpperCase() + outcome.slice(1)} News`;
                    
                    // Update confidence meter
                    const meterFill = document.querySelector('.meter-fill');
                    if (meterFill) {
                        const confidence = Math.floor(Math.random() * 30) + 70; // 70-99%
                        meterFill.style.width = `${confidence}%`;
                        document.querySelector('.meter-label').textContent = `Confidence: ${confidence}%`;
                    }
                    
                    // Highlight the result card
                    resultCard.classList.add('highlight-pulse');
                    setTimeout(() => {
                        resultCard.classList.remove('highlight-pulse');
                    }, 2000);
                }
                
                // Update visualizations
                updateVisualizations();
                
                // Add to history
                addToHistory();
            }, 1500);
        }
    }
});

// Update visualizations function removed
function updateVisualizations() {
    // This function is kept as a placeholder but its functionality has been removed
    // as the visualizations section has been removed from the UI
    console.log('Visualizations update skipped - section removed');
}

// Add entry to detection history
function addToHistory() {
    const historyContent = document.querySelector('.history-content');
    if (historyContent) {
        // Create new history item
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        
        // Randomly choose between real and fake for demo
        const isFake = Math.random() > 0.5;
        
        // Get current time
        const now = new Date();
        const timeStr = now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        
        // Create content
        historyItem.innerHTML = `
            <div class="history-badge ${isFake ? 'fake' : 'real'}"></div>
            <div class="history-text">${isFake ? 'Fake' : 'Real'} news detected: "${getRandomHeadline()}"</div>
            <div class="history-time">${timeStr}</div>
        `;
        
        // Add to history (at the top)
        historyContent.insertBefore(historyItem, historyContent.firstChild);
        
        // Add entrance animation
        historyItem.style.opacity = '0';
        historyItem.style.transform = 'translateY(-10px)';
        setTimeout(() => {
            historyItem.style.opacity = '1';
            historyItem.style.transform = 'translateY(0)';
        }, 10);
    }
}

// Get random headline for demo purposes
function getRandomHeadline() {
    const headlines = [
        "New study reveals breakthrough in AI research",
        "Government announces major policy change",
        "Scientists discover potential cure for common disease",
        "Tech company launches revolutionary product",
        "Global leaders meet to discuss climate change",
        "Stock market reaches record high amid economic growth",
        "Sports team wins championship in dramatic fashion",
        "Celebrity announces surprising career change"
    ];
    
    return headlines[Math.floor(Math.random() * headlines.length)];
}

// Clear history button handler
document.addEventListener('click', function(e) {
    if (e.target.classList.contains('clear-btn') || e.target.closest('.clear-btn')) {
        const historyContent = document.querySelector('.history-content');
        if (historyContent) {
            // Animate items out
            const items = historyContent.querySelectorAll('.history-item');
            items.forEach((item, index) => {
                setTimeout(() => {
                    item.style.opacity = '0';
                    item.style.transform = 'translateY(-10px)';
                }, 50 * index);
            });
            
            // Clear after animations
            setTimeout(() => {
                historyContent.innerHTML = '';
            }, 50 * items.length + 300);
        }
    }
});

// Refresh metrics button handler
document.addEventListener('click', function(e) {
    if (e.target.classList.contains('refresh-btn') || e.target.closest('.refresh-btn')) {
        const metricsCard = document.querySelector('.metrics-card');
        if (metricsCard) {
            metricsCard.classList.add('loading');
            
            // Simulate refresh delay
            setTimeout(() => {
                metricsCard.classList.remove('loading');
                
                // Update metrics values (for demo)
                document.querySelectorAll('.metric-value').forEach(value => {
                    // Generate a random number between 90 and 99
                    const newValue = Math.floor(Math.random() * 10) + 90;
                    value.textContent = newValue + '%';
                    
                    // Add highlight effect
                    value.classList.add('highlight-pulse');
                    setTimeout(() => {
                        value.classList.remove('highlight-pulse');
                    }, 2000);
                });
                
                // If using Chart.js, update chart data
                if (typeof Chart !== 'undefined') {
                    const chart = Chart.getChart('metrics-chart');
                    if (chart) {
                        // Update with slightly different values
                        chart.data.datasets.forEach(dataset => {
                            dataset.data = dataset.data.map(value => {
                                // Random adjustment between -2 and +2
                                return Math.max(80, Math.min(99, value + (Math.random() * 4 - 2)));
                            });
                        });
                        chart.update();
                    }
                }
            }, 1000);
        }
    }
});

// Export button handler
document.addEventListener('click', function(e) {
    if (e.target.classList.contains('export-btn') || e.target.closest('.export-btn')) {
        // Create a mock export object
        const exportData = {
            timestamp: new Date().toISOString(),
            prediction: document.querySelector('.prediction-badge').classList.contains('fake') ? 'FAKE' : 'REAL',
            confidence: document.querySelector('.meter-label').textContent.replace('Confidence: ', ''),
            content: document.querySelector('.content-preview p').textContent,
            metrics: {
                accuracy: '97%',
                precision: '94%',
                recall: '93%',
                f1Score: '93.5%'
            }
        };
        
        // Convert to JSON string
        const jsonStr = JSON.stringify(exportData, null, 2);
        
        // Create a blob and download link
        const blob = new Blob([jsonStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        // Create download link and trigger click
        const a = document.createElement('a');
        a.href = url;
        a.download = 'fake-news-detection-' + new Date().getTime() + '.json';
        document.body.appendChild(a);
        a.click();
        
        // Clean up
        setTimeout(() => {
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, 100);
    }
});

// Feed toggle handler
document.addEventListener('change', function(e) {
    if (e.target.classList.contains('feed-toggle-input')) {
        const feedStatus = document.querySelector('.feed-status');
        if (feedStatus) {
            feedStatus.textContent = e.target.checked ? 'Live Feed: Active' : 'Live Feed: Paused';
            
            // If activated, simulate incoming feed items
            if (e.target.checked) {
                simulateLiveFeed();
            }
        }
    }
});

// Simulate live feed updates
function simulateLiveFeed() {
    const feedContent = document.querySelector('.feed-content');
    const feedToggle = document.querySelector('.feed-toggle-input');
    
    if (feedContent && feedToggle && feedToggle.checked) {
        // Create new feed item
        const feedItem = document.createElement('div');
        feedItem.className = 'feed-item';
        
        // Randomly choose between real, fake, or uncertain
        const outcomes = ['real', 'fake', 'uncertain'];
        const outcome = outcomes[Math.floor(Math.random() * outcomes.length)];
        
        // Get random headline
        const headline = getRandomHeadline();
        
        // Create content
        feedItem.innerHTML = `
            <div class="feed-badge ${outcome}">
                <i class="fas fa-${outcome === 'real' ? 'check-circle' : outcome === 'fake' ? 'times-circle' : 'question-circle'}"></i>
                ${outcome.charAt(0).toUpperCase() + outcome.slice(1)}
            </div>
            <div class="feed-title">${headline}</div>
            <div class="feed-meta">
                <span>Source: News Agency</span>
                <span>Just now</span>
            </div>
        `;
        
        // Add to feed (at the top)
        feedContent.insertBefore(feedItem, feedContent.firstChild);
        
        // Add entrance animation
        feedItem.style.opacity = '0';
        feedItem.style.transform = 'translateY(-10px)';
        setTimeout(() => {
            feedItem.style.opacity = '1';
            feedItem.style.transform = 'translateY(0)';
        }, 10);
        
        // Schedule next update if still active
        const delay = Math.floor(Math.random() * 5000) + 5000; // 5-10 seconds
        setTimeout(simulateLiveFeed, delay);
    }
}

// Initialize live feed if toggle is checked on page load
document.addEventListener('DOMContentLoaded', function() {
    const feedToggle = document.querySelector('.feed-toggle-input');
    if (feedToggle && feedToggle.checked) {
        simulateLiveFeed();
    }
});
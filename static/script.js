document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewSection = document.getElementById('preview-section');
    const imagePreview = document.getElementById('image-preview');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resetBtn = document.getElementById('reset-btn');
    const newScanBtn = document.getElementById('new-scan-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const resultCard = document.getElementById('result-card');
    const errorBanner = document.getElementById('error-banner');
    const errorMessage = document.getElementById('error-message');

    let currentFile = null;

    // Click to upload
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    // File Input Change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and Drop Effects
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            handleFile(files[0]);
        }
    });

    // Process File
    function handleFile(file) {
        if (!file.type.match('image.*')) {
            showError("Please upload a valid image file (JPG, PNG).");
            return;
        }

        currentFile = file;
        const reader = new FileReader();
        
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.style.display = 'none';
            previewSection.style.display = 'block';
            resultCard.style.display = 'none';
            errorBanner.style.display = 'none';
            // ensure scanning animation is off initially
            previewSection.classList.remove('scanning'); 
        }
        
        reader.readAsDataURL(file);
    }

    // Reset Flow
    function resetApp() {
        currentFile = null;
        fileInput.value = '';
        previewSection.style.display = 'none';
        previewSection.classList.remove('scanning');
        resultCard.style.display = 'none';
        dropZone.style.display = 'block';
        imagePreview.src = '';
    }

    resetBtn.addEventListener('click', resetApp);
    newScanBtn.addEventListener('click', resetApp);

    // Analyze API Call
    analyzeBtn.addEventListener('click', () => runAnalysis());

    async function runAnalysis(retryCount = 0) {
        if (!currentFile) return;

        const MAX_RETRIES = 5;
        const RETRY_DELAY = 5000; // 5 seconds between retries

        // UI State: Scanning/Loading
        analyzeBtn.disabled = true;
        resetBtn.disabled = true;
        previewSection.classList.add('scanning');
        setTimeout(() => {
            previewSection.style.display = 'none';
            loadingOverlay.style.display = 'block';
            if (retryCount > 0) {
                loadingOverlay.querySelector('p').textContent = `Server is waking up... retrying (${retryCount}/${MAX_RETRIES})`;
            } else {
                loadingOverlay.querySelector('p').textContent = 'Running Neural Network Analysis...';
            }
        }, retryCount === 0 ? 1500 : 0);

        errorBanner.style.display = 'none';
        resultCard.style.display = 'none';

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            let data;
            const text = await response.text();
            if (!text || text.trim() === '') {
                if (retryCount < MAX_RETRIES) {
                    setTimeout(() => runAnalysis(retryCount + 1), RETRY_DELAY);
                    return;
                }
                throw new Error("Server is unavailable. Please try again later.");
            }
            try {
                data = JSON.parse(text);
            } catch {
                if (retryCount < MAX_RETRIES) {
                    setTimeout(() => runAnalysis(retryCount + 1), RETRY_DELAY);
                    return;
                }
                throw new Error("Server is unavailable. Please try again later.");
            }

            if (!response.ok) {
                throw new Error(data.error || "Server returned an error");
            }

            // Populate Results
            document.getElementById('res-plant').textContent = data.plant;
            
            const diseaseEl = document.getElementById('res-disease');
            diseaseEl.textContent = data.disease;
            
            // Dynamic Icons/Colors based on status
            const iconWrap = document.getElementById('disease-icon-wrap');
            if (data.disease.toLowerCase().includes('healthy')) {
                diseaseEl.className = 'result-value target-disease success';
                iconWrap.innerHTML = '<i data-lucide="check-circle"></i>';
            } else {
                diseaseEl.className = 'result-value target-disease danger';
                iconWrap.innerHTML = '<i data-lucide="bug"></i>';
            }
            
            // Re-initialize any newly injected Lucide icons
            if (window.lucide) {
                lucide.createIcons();
            }
            
            document.getElementById('res-confidence').textContent = data.confidence + '% Match';
            document.getElementById('res-solution').textContent = data.solution;

            // Show Results after a slight delay for dramatic effect
            setTimeout(() => {
                loadingOverlay.style.display = 'none';
                resultCard.style.display = 'block';
                analyzeBtn.disabled = false;
                resetBtn.disabled = false;
            }, 1000);

        } catch (error) {
            setTimeout(() => {
                showError(error.message);
                loadingOverlay.style.display = 'none';
                loadingOverlay.querySelector('p').textContent = 'Running Neural Network Analysis...';
                previewSection.style.display = 'block';
                previewSection.classList.remove('scanning');
                analyzeBtn.disabled = false;
                resetBtn.disabled = false;
            }, retryCount === 0 ? 1000 : 0);
        }
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorBanner.style.display = 'flex';
    }

    // Interactive Rainbow Border logic
    document.querySelectorAll('.glass-panel').forEach(panel => {
        panel.addEventListener('mousemove', e => {
            const rect = panel.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            panel.style.setProperty('--mouse-x', `${x}px`);
            panel.style.setProperty('--mouse-y', `${y}px`);
        });
    });
});

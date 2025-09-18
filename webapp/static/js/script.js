// 다양한 CTA 문구 배열
const ctaMessages = [
    "음악을 찾아 분류해보세요! 🎵",
    "당신의 음악을 분석해드릴게요! 🎼",
    "음악 파일을 업로드하고 분류해보세요! 🎶",
    "AI가 당신의 음악을 분류해드립니다! 🤖",
    "음악의 비밀을 파헤쳐보세요! 🔍",
    "당신만의 음악을 분류해보세요! ⭐",
    "음악 파일을 선택하고 분석을 시작해보세요! 🚀",
    "AI 음악 분류의 마법을 경험해보세요! ✨",
    "당신의 음악 취향을 알아보세요! 🎯",
    "음악을 업로드하고 분류 결과를 확인하세요! 📊"
];

// 랜덤 CTA 문구 선택
function setRandomCTA() {
    const ctaElement = document.getElementById('random-cta');
    if (ctaElement) {
        const randomIndex = Math.floor(Math.random() * ctaMessages.length);
        ctaElement.textContent = ctaMessages[randomIndex];
    }
}

function updateFileName() {
    const input = document.getElementById('audio_file');
    const fileName = input.files.length > 0 ? input.files[0].name : '';
    document.getElementById('file-name').textContent = fileName;
}

function toggleMode() {
    const body = document.body;
    const btn = document.querySelector('.mode-toggle button');
    if (body.classList.contains('dark-mode')) {
        body.classList.remove('dark-mode');
        body.classList.add('white-mode');
        btn.textContent = '다크 모드';
    } else {
        body.classList.remove('white-mode');
        body.classList.add('dark-mode');
        btn.textContent = '화이트 모드';
    }
}

// URL 검증 함수
async function validateUrl() {
    const urlInput = document.getElementById('url_input');
    const validationDiv = document.getElementById('url-validation');
    const url = urlInput.value.trim();
    
    if (!url) {
        validationDiv.textContent = '';
        validationDiv.className = 'validation-message';
        return;
    }
    
    try {
        const response = await fetch('/api/validate_url', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: url })
        });
        
        const data = await response.json();
        
        if (data.valid) {
            validationDiv.textContent = `✅ ${data.message}`;
            validationDiv.className = 'validation-message valid';
        } else {
            validationDiv.textContent = `❌ ${data.message}`;
            validationDiv.className = 'validation-message invalid';
        }
    } catch (error) {
        validationDiv.textContent = '❌ URL 검증 중 오류가 발생했습니다.';
        validationDiv.className = 'validation-message invalid';
    }
}

// 링크 미리보기 함수 (간단한 정보만 표시)
function getLinkPreview() {
    const urlInput = document.getElementById('url_input');
    const previewDiv = document.getElementById('link-preview');
    const platformBadge = document.getElementById('preview-platform');
    const previewInfo = document.getElementById('preview-info');
    const url = urlInput.value.trim();
    
    if (!url) {
        hideLinkPreview();
        return;
    }
    
    // URL이 유효한지 먼저 확인
    const validationDiv = document.getElementById('url-validation');
    if (validationDiv.classList.contains('invalid')) {
        hideLinkPreview();
        return;
    }
    
    // 간단한 정보만 표시 (API 호출 없음)
    previewDiv.style.display = 'block';
    
    // URL에서 플랫폼 감지
    let platform = '음악 링크';
    if (url.includes('youtube.com') || url.includes('youtu.be')) {
        platform = 'YouTube';
    } else if (url.includes('soundcloud.com')) {
        platform = 'SoundCloud';
    } else if (url.includes('spotify.com')) {
        platform = 'Spotify';
    } else if (url.includes('.mp3') || url.includes('.wav') || url.includes('.m4a')) {
        platform = '오디오 파일';
    }
    
    platformBadge.textContent = platform;
    
    // 간단한 정보만 표시
    let infoHTML = `
        <h4>🔗 링크 정보</h4>
        <p><strong>URL:</strong> <a href="${url}" target="_blank" style="color: #4a90e2; word-break: break-all;">${url}</a></p>
        <p><strong>플랫폼:</strong> ${platform}</p>
        <p><strong>상태:</strong> ✅ 유효한 링크</p>
        <div class="preview-note">
            <p><em>💡 이 링크로 음악 분류를 진행할 수 있습니다.</em></p>
        </div>
    `;
    
    previewInfo.innerHTML = infoHTML;
}

// 링크 미리보기 숨기기
function hideLinkPreview() {
    const previewDiv = document.getElementById('link-preview');
    previewDiv.style.display = 'none';
}

// URL 폼 제출 전 검증
function validateUrlForm() {
    const urlInput = document.getElementById('url_input');
    const url = urlInput.value.trim();
    
    if (!url) {
        alert('URL을 입력해주세요.');
        return false;
    }
    
    const validationDiv = document.getElementById('url-validation');
    if (validationDiv.classList.contains('invalid')) {
        alert('유효하지 않은 URL입니다. 다시 확인해주세요.');
        return false;
    }
    
    return true;
}

// 일괄 분류 폼 제출 전 검증
function validateBatchForm() {
    const urlsInput = document.getElementById('urls_input');
    const urls = urlsInput.value.trim();
    
    if (!urls) {
        alert('URL 목록을 입력해주세요.');
        return false;
    }
    
    // URL 개수 확인 (최대 20개)
    const urlList = urls.split(/[\n,]/).filter(url => url.trim());
    if (urlList.length > 20) {
        alert('최대 20개의 URL만 처리할 수 있습니다.');
        return false;
    }
    
    return true;
}



// 로딩 상태 표시 함수 (개선된 버전)
function showLoading(formId, loadingText = '처리 중...') {
    const form = document.getElementById(formId);
    if (form) {
        const submitBtn = form.querySelector('button[type="submit"]');
        if (submitBtn) {
            // 원본 텍스트 저장
            if (!submitBtn.getAttribute('data-original-text')) {
                submitBtn.setAttribute('data-original-text', submitBtn.textContent);
            }
            
            submitBtn.disabled = true;
            submitBtn.innerHTML = `
                <span class="loading-spinner"></span>
                ${loadingText}
            `;
            
            // 로딩 오버레이 추가
            addLoadingOverlay(formId);
        }
    }
}

// 로딩 상태 해제 함수
function hideLoading(formId) {
    const form = document.getElementById(formId);
    if (form) {
        const submitBtn = form.querySelector('button[type="submit"]');
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.textContent = submitBtn.getAttribute('data-original-text') || '분류하기';
        }
        
        // 로딩 오버레이 제거
        removeLoadingOverlay(formId);
    }
}

// 로딩 오버레이 추가
function addLoadingOverlay(formId) {
    const form = document.getElementById(formId);
    if (form && !form.querySelector('.loading-overlay')) {
        const overlay = document.createElement('div');
        overlay.className = 'loading-overlay';
        overlay.innerHTML = `
            <div class="loading-content">
                <div class="loading-spinner-large"></div>
                <div class="loading-text">
                    <h3>음악을 분석하고 있습니다...</h3>
                    <p id="loading-progress">링크에서 음악을 다운로드하는 중...</p>
                    <div class="loading-steps">
                        <div class="step active" id="step-download">
                            <span class="step-icon">📥</span>
                            <span class="step-text">다운로드 중</span>
                        </div>
                        <div class="step" id="step-process">
                            <span class="step-icon">🔍</span>
                            <span class="step-text">분석 중</span>
                        </div>
                        <div class="step" id="step-complete">
                            <span class="step-icon">✅</span>
                            <span class="step-text">완료</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
        form.appendChild(overlay);
        
        // 단계별 진행 시뮬레이션
        simulateLoadingProgress();
    }
}

// 로딩 오버레이 제거
function removeLoadingOverlay(formId) {
    const form = document.getElementById(formId);
    if (form) {
        const overlay = form.querySelector('.loading-overlay');
        if (overlay) {
            overlay.remove();
        }
    }
}

// 로딩 진행 시뮬레이션
function simulateLoadingProgress() {
    const progressText = document.getElementById('loading-progress');
    const steps = ['step-download', 'step-process', 'step-complete'];
    const messages = [
        '링크에서 음악을 다운로드하는 중...',
        'AI가 음악을 분석하는 중...',
        '분류 결과를 준비하는 중...'
    ];
    
    let currentStep = 0;
    
    const interval = setInterval(() => {
        if (currentStep < steps.length) {
            // 이전 단계 비활성화
            if (currentStep > 0) {
                const prevStep = document.getElementById(steps[currentStep - 1]);
                if (prevStep) {
                    prevStep.classList.remove('active');
                    prevStep.classList.add('completed');
                }
            }
            
            // 현재 단계 활성화
            const currentStepEl = document.getElementById(steps[currentStep]);
            if (currentStepEl) {
                currentStepEl.classList.add('active');
            }
            
            // 메시지 업데이트
            if (progressText) {
                progressText.textContent = messages[currentStep];
            }
            
            currentStep++;
        } else {
            clearInterval(interval);
        }
    }, 2000); // 2초마다 단계 변경
}

// 페이지 로드 시 랜덤 CTA 설정
window.addEventListener('load', setRandomCTA);

// 업로드 폴더 정보 조회
async function refreshUploadsInfo() {
    const statsDiv = document.getElementById('uploads-stats');
    const filesDiv = document.getElementById('uploads-files');
    const clearBtn = document.querySelector('.clear-btn');
    const refreshBtn = document.querySelector('.refresh-btn');
    
    // 로딩 상태 표시
    statsDiv.innerHTML = '<p><span class="loading"></span>업로드 폴더 정보를 불러오는 중...</p>';
    refreshBtn.disabled = true;
    
    try {
        const response = await fetch('/get_uploads_info');
        const data = await response.json();
        
        if (data.success) {
            // 통계 정보 업데이트
            let statsHTML = '';
            if (data.file_count === 0) {
                statsHTML = '<p>📁 업로드된 파일이 없습니다.</p>';
                filesDiv.style.display = 'none';
                clearBtn.disabled = true;
            } else {
                statsHTML = `
                    <p><strong>📁 총 파일 수:</strong> ${data.file_count}개</p>
                    <p><strong>💾 총 용량:</strong> ${data.total_size_formatted}</p>
                `;
                
                // 파일 목록 표시
                let filesHTML = '<h4>📋 파일 목록:</h4>';
                data.files.forEach(file => {
                    filesHTML += `
                        <div class="file-item">
                            <span class="file-name">${file.name}</span>
                            <span class="file-size">${file.size_formatted}</span>
                        </div>
                    `;
                });
                filesDiv.innerHTML = filesHTML;
                filesDiv.style.display = 'block';
                clearBtn.disabled = false;
            }
            
            statsDiv.innerHTML = statsHTML;
        } else {
            statsDiv.innerHTML = `<p style="color: #dc3545;">❌ ${data.message}</p>`;
            filesDiv.style.display = 'none';
            clearBtn.disabled = true;
        }
    } catch (error) {
        statsDiv.innerHTML = '<p style="color: #dc3545;">❌ 폴더 정보 조회 중 오류가 발생했습니다.</p>';
        filesDiv.style.display = 'none';
        clearBtn.disabled = true;
    } finally {
        refreshBtn.disabled = false;
    }
}

// 업로드 폴더 비우기
async function clearUploads() {
    if (!confirm('정말로 업로드 폴더의 모든 파일을 삭제하시겠습니까?\n이 작업은 되돌릴 수 없습니다.')) {
        return;
    }
    
    const clearBtn = document.querySelector('.clear-btn');
    const originalText = clearBtn.textContent;
    
    // 로딩 상태 표시
    clearBtn.disabled = true;
    clearBtn.textContent = '🗑️ 삭제 중...';
    
    try {
        const response = await fetch('/clear_uploads', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            alert('✅ ' + data.message);
            // 폴더 정보 새로고침
            refreshUploadsInfo();
        } else {
            alert('❌ ' + data.message);
        }
    } catch (error) {
        alert('❌ 폴더 정리 중 오류가 발생했습니다.');
    } finally {
        clearBtn.disabled = false;
        clearBtn.textContent = originalText;
    }
}

// 페이지 로드 시 이벤트 리스너 설정
document.addEventListener('DOMContentLoaded', function() {
    // URL 폼 제출 검증 및 로딩 표시
    const urlForm = document.getElementById('url-form');
    if (urlForm) {
        urlForm.addEventListener('submit', function(e) {
            if (!validateUrlForm()) {
                e.preventDefault();
                return;
            }
            
            // 로딩 표시 활성화
            showLoading('url-form', '링크 분석 중...');
        });
    }
    
    // 일괄 분류 폼 제출 검증 및 로딩 표시
    const batchForm = document.getElementById('batch-form');
    if (batchForm) {
        batchForm.addEventListener('submit', function(e) {
            if (!validateBatchForm()) {
                e.preventDefault();
                return;
            }
            
            // 로딩 표시 활성화
            showLoading('batch-form', '일괄 분석 중...');
        });
    }
    
    // 파일 업로드 폼에도 로딩 표시 추가
    const fileForm = document.querySelector('form[action*="classify"]');
    if (fileForm) {
        fileForm.addEventListener('submit', function(e) {
            showLoading('file-form', '파일 분석 중...');
        });
    }
    
    // URL 입력 시 실시간 검증
    const urlInput = document.getElementById('url_input');
    if (urlInput) {
        urlInput.addEventListener('input', validateUrl);
        urlInput.addEventListener('blur', validateUrl);
        // 링크 미리보기 자동 호출 비활성화 (코드는 유지)
        // urlInput.addEventListener('blur', getLinkPreview);
    }
    
    // 페이지 로드 시 업로드 폴더 정보 조회
    refreshUploadsInfo();
}); 
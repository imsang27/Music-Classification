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

// 다중 파일 업로드 관련 함수들
function updateMultipleFileNames() {
    const input = document.getElementById('audio_files');
    const files = input.files;
    const fileNamesElement = document.getElementById('multiple-file-names');
    const selectedFilesList = document.getElementById('selected-files-list');
    const filesContainer = document.getElementById('files-container');
    
    if (files.length === 0) {
        fileNamesElement.textContent = '';
        selectedFilesList.style.display = 'none';
        return;
    }
    
    // 파일 개수 표시
    if (files.length === 1) {
        fileNamesElement.textContent = files[0].name;
    } else {
        fileNamesElement.textContent = `${files.length}개 파일 선택됨`;
    }
    
    // 선택된 파일 목록 표시
    selectedFilesList.style.display = 'block';
    
    // 파일 목록 HTML 생성
    let filesHTML = '';
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const fileSize = formatFileSize(file.size);
        filesHTML += `
            <div class="file-item" data-index="${i}">
                <div class="file-info">
                    <span class="file-icon">🎵</span>
                    <div class="file-details">
                        <span class="file-name">${file.name}</span>
                        <span class="file-size">${fileSize}</span>
                    </div>
                </div>
                <button type="button" class="remove-file-btn" onclick="removeFile(${i})" title="파일 제거">×</button>
            </div>
        `;
    }
    
    filesContainer.innerHTML = filesHTML;
}

function removeFile(index) {
    const input = document.getElementById('audio_files');
    const dt = new DataTransfer();
    
    // 선택된 인덱스가 아닌 파일들만 유지
    for (let i = 0; i < input.files.length; i++) {
        if (i !== index) {
            dt.items.add(input.files[i]);
        }
    }
    
    input.files = dt.files;
    updateMultipleFileNames();
}

function clearSelectedFiles() {
    const input = document.getElementById('audio_files');
    input.value = '';
    updateMultipleFileNames();
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
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

// 기본 링크 정보 표시 (자동)
function showBasicLinkInfo() {
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
    
    // 기본 정보 표시
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
    
    // 기본 정보만 표시
    let infoHTML = `
        <h4>🔗 링크 정보</h4>
        <p><strong>URL:</strong> <a href="${url}" target="_blank" style="color: #4a90e2; word-break: break-all;">${url}</a></p>
        <p><strong>플랫폼:</strong> ${platform}</p>
        <p><strong>상태:</strong> ✅ 유효한 링크</p>
        <div class="preview-note">
            <p><em>💡 이 링크로 음악 분류를 진행할 수 있습니다.</em></p>
            <p><em>🔍 상세 정보를 보려면 👁️ 버튼을 클릭하세요.</em></p>
        </div>
    `;
    
    previewInfo.innerHTML = infoHTML;
}

// 상세 링크 미리보기 함수 (버튼 클릭 시)
async function getLinkPreview() {
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
    
    try {
        // 로딩 상태 표시
        previewDiv.style.display = 'block';
        platformBadge.textContent = '로딩 중...';
        previewInfo.innerHTML = '<p>상세 링크 정보를 가져오는 중...</p>';
        
        const response = await fetch('/api/link_preview', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: url })
        });
        
        const data = await response.json();
        
        if (data.error) {
            platformBadge.textContent = '오류';
            previewInfo.innerHTML = `<p style="color: #e74c3c;">❌ ${data.error}</p>`;
            return;
        }
        
        // 플랫폼 배지 설정
        if (data.platform) {
            platformBadge.textContent = data.platform;
        } else if (data.title) {
            platformBadge.textContent = 'YouTube';
        } else {
            platformBadge.textContent = '음악 링크';
        }
        
        // 상세 미리보기 정보 구성
        let infoHTML = '';
        
        if (data.title) {
            infoHTML += `<h4>${data.title}</h4>`;
        }
        
        if (data.uploader) {
            infoHTML += `<p><strong>업로더:</strong> ${data.uploader}</p>`;
        }
        
        if (data.duration) {
            const minutes = Math.floor(data.duration / 60);
            const seconds = data.duration % 60;
            infoHTML += `<p><strong>길이:</strong> ${minutes}:${seconds.toString().padStart(2, '0')}</p>`;
        }
        
        if (data.view_count) {
            infoHTML += `<p><strong>조회수:</strong> ${data.view_count.toLocaleString()}</p>`;
        }
        
        if (data.like_count) {
            infoHTML += `<p><strong>좋아요:</strong> ${data.like_count.toLocaleString()}</p>`;
        }
        
        if (data.description) {
            infoHTML += `<p><strong>설명:</strong> ${data.description}</p>`;
        }
        
        if (data.content_type) {
            infoHTML += `<p><strong>파일 형식:</strong> ${data.content_type}</p>`;
        }
        
        if (data.file_size && data.file_size !== '알 수 없음') {
            const sizeMB = (data.file_size / (1024 * 1024)).toFixed(2);
            infoHTML += `<p><strong>파일 크기:</strong> ${sizeMB} MB</p>`;
        }
        
        if (data.tags && data.tags.length > 0) {
            infoHTML += `<p><strong>태그:</strong> ${data.tags.join(', ')}</p>`;
        }
        
        // 통계 정보
        if (data.view_count || data.like_count || data.duration) {
            infoHTML += '<div class="preview-stats">';
            if (data.duration) {
                const minutes = Math.floor(data.duration / 60);
                const seconds = data.duration % 60;
                infoHTML += `<span>⏱️ ${minutes}:${seconds.toString().padStart(2, '0')}</span>`;
            }
            if (data.view_count) {
                infoHTML += `<span>👁️ ${data.view_count.toLocaleString()}</span>`;
            }
            if (data.like_count) {
                infoHTML += `<span>👍 ${data.like_count.toLocaleString()}</span>`;
            }
            infoHTML += '</div>';
        }
        
        previewInfo.innerHTML = infoHTML;
        
    } catch (error) {
        platformBadge.textContent = '오류';
        previewInfo.innerHTML = `<p style="color: #e74c3c;">❌ 상세 정보 가져오기 중 오류가 발생했습니다.</p>`;
    }
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
        urlsInput.focus();
        return false;
    }
    
    // URL 목록 파싱 및 검증
    const urlList = urls.split(/[\n,]/).map(url => url.trim()).filter(url => url);
    
    if (urlList.length === 0) {
        alert('유효한 URL이 없습니다.');
        urlsInput.focus();
        return false;
    }
    
    // URL 개수 확인 (최대 20개)
    if (urlList.length > 20) {
        if (!confirm(`많은 수의 URL이 입력되었습니다 (${urlList.length}개). 계속 진행하시겠습니까?\n처리 시간이 오래 걸릴 수 있습니다.`)) {
            return false;
        }
    }
    
    // 각 URL 형식 검증
    const urlPattern = /^https?:\/\/.+/;
    for (let i = 0; i < urlList.length; i++) {
        if (!urlPattern.test(urlList[i])) {
            alert(`올바르지 않은 URL 형식입니다 (${i + 1}번째): ${urlList[i]}\nhttp:// 또는 https://로 시작하는 URL을 입력해주세요.`);
            urlsInput.focus();
            return false;
        }
    }
    
    return true;
}

// 다중 파일 폼 제출 전 검증
function validateMultipleFileForm() {
    const filesInput = document.getElementById('audio_files');
    const files = filesInput.files;
    
    if (files.length === 0) {
        alert('분류할 음악 파일을 선택해주세요.');
        filesInput.focus();
        return false;
    }
    
    // 파일 개수 제한 (최대 10개)
    if (files.length > 10) {
        if (!confirm(`많은 수의 파일이 선택되었습니다 (${files.length}개). 계속 진행하시겠습니까?\n처리 시간이 오래 걸릴 수 있습니다.`)) {
            return false;
        }
    }
    
    // 파일 크기 제한 (각 파일 최대 50MB)
    const maxFileSize = 50 * 1024 * 1024; // 50MB
    for (let i = 0; i < files.length; i++) {
        if (files[i].size > maxFileSize) {
            alert(`파일 크기가 너무 큽니다 (${i + 1}번째): ${files[i].name}\n최대 50MB까지 업로드 가능합니다.`);
            return false;
        }
    }
    
    // 파일 형식 검증
    const allowedTypes = ['audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/m4a', 'audio/ogg', 'audio/flac'];
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const isValidType = allowedTypes.some(type => file.type === type) || 
                           /\.(mp3|wav|m4a|ogg|flac)$/i.test(file.name);
        
        if (!isValidType) {
            alert(`지원하지 않는 파일 형식입니다 (${i + 1}번째): ${file.name}\n지원 형식: MP3, WAV, M4A, OGG, FLAC`);
            return false;
        }
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
            
            // 버튼을 비활성화하지 않고 텍스트만 변경
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

// 모델 상태 확인 및 UI 업데이트
async function checkModelStatus() {
    const statusElement = document.querySelector('.status-value');
    const statusNote = document.querySelector('.status-note');
    const modelNameElement = document.getElementById('model-name');
    
    if (!statusElement) return;
    
    try {
        const response = await fetch('/api/model_status');
        const data = await response.json();
        
        if (data.success) {
            const modelInfo = data.model_status;
            
            // 상태 텍스트 업데이트 - "로드됨" 또는 "로드되지 않음" 유지
            if (modelInfo.model_loaded) {
                statusElement.textContent = '로드됨';
            } else {
                statusElement.textContent = '로드되지 않음';
            }
            
            // CSS 클래스 업데이트
            statusElement.className = `status-value ${modelInfo.model_loaded ? 'status-success' : 'status-error'}`;
            
            // 모델 이름 표시
            if (modelNameElement) {
                if (modelInfo.model_loaded) {
                    const modelName = modelInfo.model_type || modelInfo.model_name || 'music_genres_classification';
                    console.log('JavaScript - 받은 모델 정보:', modelInfo);
                    console.log('JavaScript - 사용할 모델명:', modelName);
                    modelNameElement.textContent = modelName;
                    modelNameElement.style.display = 'inline';
                } else {
                    modelNameElement.style.display = 'none';
                }
            }
            
            // 추가 정보 표시
            if (statusNote) {
                if (modelInfo.model_loaded) {
                    statusNote.textContent = '(AI 모델 사용 가능)';
                    statusNote.className = 'status-note success';
                } else {
                    statusNote.textContent = '(규칙 기반 분류만 사용 가능)';
                    statusNote.className = 'status-note error';
                }
            }
            
            // 지원 장르 정보 표시 (선택사항)
            if (modelInfo.supported_genres && modelInfo.supported_genres.length > 0) {
                console.log('지원되는 장르:', modelInfo.supported_genres);
            }
            
            // GPU 사용 가능 여부 표시 (선택사항)
            if (modelInfo.gpu_available) {
                console.log('GPU 사용 가능');
            }
            
        } else {
            // API 오류 시 기본 상태로 설정
            statusElement.textContent = '확인 불가';
            statusElement.className = 'status-value status-error';
            if (statusNote) {
                statusNote.textContent = '(상태 확인 중 오류 발생)';
                statusNote.className = 'status-note error';
            }
        }
    } catch (error) {
        console.error('모델 상태 확인 중 오류:', error);
        // 네트워크 오류 시 기본 상태로 설정
        statusElement.textContent = '확인 불가';
        statusElement.className = 'status-value status-error';
        if (statusNote) {
            statusNote.textContent = '(네트워크 오류)';
            statusNote.className = 'status-note error';
        }
    }
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
    // 페이지 로드 시 모델 상태 확인
    checkModelStatus();
    
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
            
            // 로딩 표시 활성화 (배치 분류는 진행률 페이지로 이동하므로 간단한 로딩만)
            showLoading('batch-form', '배치 분류를 시작합니다...');
        });
    }
    
    // 파일 업로드 폼에도 로딩 표시 추가
    const fileForm = document.querySelector('form[action*="classify"]');
    if (fileForm) {
        fileForm.addEventListener('submit', function(e) {
            // 폼 제출을 막지 않고 로딩 표시만 활성화
            showLoading('file-form', '파일 분석 중...');
            // 폼이 정상적으로 제출되도록 함
        });
    }
    
    // 다중 파일 업로드 폼 제출 검증 및 로딩 표시
    const multipleFileForm = document.getElementById('multiple-file-form');
    if (multipleFileForm) {
        multipleFileForm.addEventListener('submit', function(e) {
            if (!validateMultipleFileForm()) {
                e.preventDefault();
                return;
            }
            
            // 로딩 표시 활성화
            showLoading('multiple-file-form', '여러 파일 분석 중...');
        });
    }
    
    // URL 입력 시 실시간 검증 및 기본 정보 표시
    const urlInput = document.getElementById('url_input');
    if (urlInput) {
        urlInput.addEventListener('input', function() {
            validateUrl();
            showBasicLinkInfo();
        });
        urlInput.addEventListener('blur', function() {
            validateUrl();
            showBasicLinkInfo();
        });
    }
    
    // 페이지 로드 시 업로드 폴더 정보 조회
    refreshUploadsInfo();
}); 
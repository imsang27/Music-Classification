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

// 링크 미리보기 함수
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
        previewInfo.innerHTML = '<p>링크 정보를 가져오는 중...</p>';
        
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
        
        // 미리보기 정보 구성
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
        previewInfo.innerHTML = `<p style="color: #e74c3c;">❌ 미리보기 생성 중 오류가 발생했습니다.</p>`;
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



// 로딩 상태 표시 함수
function showLoading(formId) {
    const form = document.getElementById(formId);
    if (form) {
        const submitBtn = form.querySelector('button[type="submit"]');
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.textContent = '처리 중...';
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
    // URL 폼 제출 검증
    const urlForm = document.getElementById('url-form');
    if (urlForm) {
        urlForm.addEventListener('submit', function(e) {
            if (!validateUrlForm()) {
                e.preventDefault();
            }
        });
    }
    
    // 일괄 분류 폼 제출 검증
    const batchForm = document.getElementById('batch-form');
    if (batchForm) {
        batchForm.addEventListener('submit', function(e) {
            if (!validateBatchForm()) {
                e.preventDefault();
            }
        });
    }
    
    // URL 입력 시 실시간 검증
    const urlInput = document.getElementById('url_input');
    if (urlInput) {
        urlInput.addEventListener('input', validateUrl);
        urlInput.addEventListener('blur', validateUrl);
    }
    
    // 페이지 로드 시 업로드 폴더 정보 조회
    refreshUploadsInfo();
}); 
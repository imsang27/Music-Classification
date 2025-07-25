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

// 폼 제출 이벤트 리스너 추가
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
});

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
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
    const randomIndex = Math.floor(Math.random() * ctaMessages.length);
    ctaElement.textContent = ctaMessages[randomIndex];
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

// 페이지 로드 시 랜덤 CTA 설정
window.addEventListener('load', setRandomCTA); 
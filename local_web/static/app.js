'use strict';
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="app-token"]').content;
let mode = 'files', busy = false, ready = false, youtubeState = null;
const api = async (url, options = {}) => {
  const response = await fetch(url, {...options, headers: {'X-App-Token': token, ...options.headers}});
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(data.error || '요청을 처리하지 못했습니다.'); error.status = response.status; throw error;
  }
  return data;
};
function message(text, kind = '') { $('status').textContent = text; $('status').dataset.kind = kind; }
function controls() {
  $('start').disabled = busy || !ready;
  $('start').firstChild.textContent = busy ? '분류 중입니다 ' : '분류 시작 ';
  for (const id of ['file-tab', 'youtube-tab', 'audio-files', 'youtube-url']) $(id).disabled = busy;
}
function chooseMode(value) {
  if (busy) return;
  mode = value;
  $('file-panel').hidden = mode !== 'files'; $('youtube-panel').hidden = mode !== 'youtube';
  $('file-tab').setAttribute('aria-selected', String(mode === 'files'));
  $('youtube-tab').setAttribute('aria-selected', String(mode === 'youtube'));
  if (mode === 'youtube' && youtubeState && !youtubeState.ready) message(youtubeState.message, 'error');
}
$('file-tab').addEventListener('click', () => chooseMode('files'));
$('youtube-tab').addEventListener('click', () => chooseMode('youtube'));
function listFiles() {
  $('file-list').replaceChildren();
  for (const file of $('audio-files').files) {
    const li = document.createElement('li');
    li.textContent = `${file.name} · ${(file.size / 1048576).toFixed(1)} MiB`;
    $('file-list').append(li);
  }
}
$('audio-files').addEventListener('change', listFiles);
$('drop-zone').addEventListener('dragover', event => {event.preventDefault(); if (!busy) $('drop-zone').classList.add('dragging');});
$('drop-zone').addEventListener('dragleave', () => $('drop-zone').classList.remove('dragging'));
$('drop-zone').addEventListener('drop', event => {
  event.preventDefault(); $('drop-zone').classList.remove('dragging');
  if (!busy && event.dataTransfer.files.length) {$('audio-files').files = event.dataTransfer.files; listFiles();}
});
function showResults(job) {
  $('results').replaceChildren(); $('downloads').replaceChildren();
  const rows = job.result.results;
  $('result-count').textContent = `${rows.length}곡`;
  for (const row of rows) {
    const card = document.createElement('article'); card.className = 'song';
    const title = document.createElement('h3');
    title.textContent = row.display_name || '음원'; card.append(title);
    if (!row.labels) {
      const error = document.createElement('p'); error.className = 'error-note';
      error.textContent = '이 파일은 분석하지 못했습니다. 정상적인 음원 파일인지 확인해주세요.'; card.append(error);
    } else {
      for (const [axis, name] of [['genre','장르'],['emotion','감정'],['mood','분위기']]) {
        const line = document.createElement('div'); line.className = 'axis';
        const label = document.createElement('span'); label.className = 'axis-label'; label.textContent = name;
        const chips = document.createElement('div'); chips.className = 'chips';
        const candidates = axis === 'genre' ? row.candidates[axis].slice(0,1) : row.candidates[axis].slice(0,3);
        for (const candidate of candidates) {
          const chip = document.createElement('span'); chip.className = 'chip' + (candidate.representative ? ' primary-chip' : '');
          const rank = document.createElement('small'); rank.textContent = candidate.representative ? '대표' : `${candidate.rank}순위`;
          chip.append(rank, document.createTextNode(candidate.name)); chips.append(chip);
        }
        line.append(label, chips); card.append(line);
      }
    }
    $('results').append(card);
  }
  for (const [name, label] of [['태깅결과.md','Markdown 저장'],['tags.json','태그 JSON 저장'],['classification.json','상세 결과 저장']]) {
    const link = document.createElement('a'); link.href = job.downloads[name]; link.textContent = label; link.setAttribute('download', name); $('downloads').append(link);
  }
  $('downloads').hidden = false;
}
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
async function poll(id) {
  busy = true; controls(); $('empty-state').hidden = true;
  try {
    while (true) {
      const job = await api(`/api/jobs/${encodeURIComponent(id)}`);
      message(job.message, ['preparing','running'].includes(job.status) ? 'busy' : job.status === 'completed' ? '' : 'error');
      if (['completed','partial'].includes(job.status)) {showResults(job); break;}
      if (job.status === 'failed') break;
      await sleep(1200);
    }
  } catch (error) {
    if (error.status === 404) {
      sessionStorage.removeItem('music-tagging-job'); $('empty-state').hidden = false;
      message('이전 작업은 현재 서버에 없습니다. 저장된 결과는 outputs/local-web 폴더에 남아 있습니다. 새 음원을 선택해주세요.');
    } else {
      message('서버 연결을 확인해주세요. 서버가 실행 중이면 화면을 새로고침하여 작업을 다시 확인할 수 있습니다. ' + error.message, 'error');
    }
  } finally {busy = false; controls();}
}
$('tagging-form').addEventListener('submit', async event => {
  event.preventDefault(); if (busy || !ready) return;
  let body, options;
  if (mode === 'files') {
    const files = [...$('audio-files').files];
    if (!files.length || files.length > 20) {message('음원 파일을 1~20개 선택해주세요.', 'error'); return;}
    if (files.some(file => !file.size || file.size > 100*1048576) || files.reduce((n,f) => n+f.size,0) > 500*1048576) {
      message('빈 파일을 제외하고 파일당 100MiB, 합계 500MiB 이하로 선택해주세요.', 'error'); return;
    }
    body = new FormData(); for (const file of files) body.append('audio', file);
    options = {method:'POST', body};
  } else {
    const url = $('youtube-url').value.trim();
    if (!url) {message('YouTube 영상 한 곡의 링크를 입력해주세요.', 'error'); return;}
    options = {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({url})};
  }
  busy = true; controls(); message('입력을 준비하고 있습니다.', 'busy');
  try {
    const job = await api(`/api/jobs/${mode}`, options);
    sessionStorage.setItem('music-tagging-job', job.id);
    $('results').replaceChildren(); $('downloads').hidden = true; $('result-count').textContent = '';
    await poll(job.id);
  } catch (error) {message(error.message, 'error');}
  finally {busy = false; controls();}
});
async function initialize() {
  try {
    const state = await api('/api/status'); ready = state.model.ready; youtubeState = state.youtube;
    message(state.model.message, ready ? '' : 'error'); $('model-help').hidden = ready; controls();
    const id = state.active_job || sessionStorage.getItem('music-tagging-job');
    if (id) await poll(id);
  } catch (error) {message('서버에 연결하지 못했습니다. 실행 창을 확인한 뒤 새로고침해주세요.', 'error');}
}
initialize();

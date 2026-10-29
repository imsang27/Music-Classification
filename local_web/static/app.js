'use strict';
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="app-token"]').content;
const ui = window.MusicUI;
let mode = 'files', busy = false, ready = false, youtubeState = null;
let method = 'familiar', models = {};
const resultByMethod = {};
function selectMethod(value, announce = true, force = false) {
  if (busy && !force) return;
  ui.hideTip();
  method = value;
  for (const [id, key] of [['familiar-tab','familiar'],['legacy-tab','wav2vec2']]) {
    $(id).setAttribute('aria-selected', String(method === key)); $(id).tabIndex = method === key ? 0 : -1;
  }
  $('ai-panel').setAttribute('aria-labelledby', method === 'familiar' ? 'familiar-tab' : 'legacy-tab');
  $('method-description').textContent = method === 'familiar' ? 'MuQ · 장르 25개, 감정 9개, 분위기 11개를 전곡에서 비교합니다.' : 'Wav2Vec2 · 앞부분 최대 30초로 기존 10장르를 분류합니다. 감정·분위기는 미지원입니다.';
  const emptyCopy = $('empty-copy') || $('empty-state').querySelector('p');
  emptyCopy.textContent = method === 'familiar' ? '분류가 끝나면 감정과 분위기를 각각 세 가지 후보로 보여드릴게요.' : '분류가 끝나면 기존 10장르 중 가장 가까운 장르를 보여드립니다. 감정·분위기는 제공하지 않습니다.';
  ready = Boolean(models[method]?.ready); controls();
  $('model-help').hidden = ready || method === 'wav2vec2';
  if (announce) message(models[method]?.message || '실행 환경을 확인하고 있습니다.', ready ? '' : 'error');
  if (resultByMethod[method]) showResults(resultByMethod[method]);
  else {$('results').replaceChildren(); $('downloads').hidden = true; $('result-count').textContent = ''; $('empty-state').hidden = false;}
  if (announce) ui.enter($('ai-panel'));
}
for (const id of ['familiar-tab','legacy-tab']) {
  $(id).addEventListener('click', () => selectMethod($(id).dataset.method));
  $(id).addEventListener('keydown', event => {
    if (busy || !['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 'familiar' : event.key === 'End' ? 'wav2vec2' : method === 'familiar' ? 'wav2vec2' : 'familiar';
    selectMethod(next); $(next === 'familiar' ? 'familiar-tab' : 'legacy-tab').focus();
  });
}
const api = async (url, options = {}) => {
  const response = await fetch(url, {...options, headers: {'X-App-Token': token, ...options.headers}});
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(data.error || '요청을 처리하지 못했습니다.'); error.status = response.status; throw error;
  }
  return data;
};
function message(text, kind = '') {
  if ($('status').textContent !== text) $('status').textContent = text;
  $('status').dataset.kind = kind;
}
function feedback(text, kind = 'error') {message(text,kind === 'error' ? 'error' : ''); ui.toast(text,kind);}
function controls() {
  $('start').disabled = busy || !ready;
  $('start').firstChild.textContent = busy ? '분류 중입니다 ' : '분류 시작 ';
  $('start').setAttribute('aria-busy',String(busy)); ui.loading(busy);
  if (busy) {$('downloads').hidden = true; ui.hideTip();}
  for (const id of ['file-tab', 'youtube-tab', 'audio-files', 'youtube-url','familiar-tab','legacy-tab']) $(id).disabled = busy;
}
function chooseMode(value) {
  if (busy) return;
  mode = value;
  $('file-panel').hidden = mode !== 'files'; $('youtube-panel').hidden = mode !== 'youtube';
  $('file-tab').setAttribute('aria-selected', String(mode === 'files'));
  $('youtube-tab').setAttribute('aria-selected', String(mode === 'youtube'));
  ui.enter($(mode === 'files' ? 'file-panel' : 'youtube-panel'));
  if (mode === 'youtube' && youtubeState && !youtubeState.ready) feedback(youtubeState.message);
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
  const count = $('audio-files').files.length;
  if (count) {ui.toast(`${count}개 음원을 선택했습니다.`); ui.enter($('file-list'));}
}
$('audio-files').addEventListener('change', listFiles);
$('drop-zone').addEventListener('dragover', event => {event.preventDefault(); if (!busy) $('drop-zone').classList.add('dragging');});
$('drop-zone').addEventListener('dragleave', () => $('drop-zone').classList.remove('dragging'));
$('drop-zone').addEventListener('drop', event => {
  event.preventDefault(); $('drop-zone').classList.remove('dragging');
  if (!busy && event.dataTransfer.files.length) {$('audio-files').files = event.dataTransfer.files; listFiles();}
});
function showResults(job) {
  ui.loading(false);
  resultByMethod[job.method || 'familiar'] = job;
  $('empty-state').hidden = true;
  $('results').replaceChildren(); $('downloads').replaceChildren();
  const rows = job.result.results;
  const sourceLabel = document.createElement('p'); sourceLabel.className = 'hint';
  sourceLabel.textContent = job.method === 'wav2vec2' ? '기존 AI · Wav2Vec2 · 앞부분 최대 30초' : '현재 AI · MuQ · 전곡 구간 종합';
  $('results').append(sourceLabel);
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
        if (!candidates.length) {const note = document.createElement('span'); note.className = 'unsupported'; note.textContent = '이 방식에서는 미지원'; chips.append(note);}
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
    const link = document.createElement('a'); link.href = job.downloads[name]; link.textContent = label; link.setAttribute('download', name);
    link.dataset.tooltip = name === '태깅결과.md' ? '읽기 쉬운 표 형식의 문서입니다. 노트에 적용하기 전 후보를 확인하세요.' : name === 'tags.json' ? '대표값과 순위별 후보를 프로그램에서 재사용할 수 있는 파일입니다.' : '모델 응답과 분석 범위를 포함한 상세 기록입니다.';
    link.addEventListener('click',() => ui.toast('결과 파일 다운로드를 요청했습니다.'));
    $('downloads').append(link);
  }
  $('downloads').hidden = false;
  ui.enter($('results'));
}
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
async function poll(id, announce = true) {
  busy = true; controls(); $('empty-state').hidden = true;
  let sawRunning = false;
  try {
    while (true) {
      const job = await api(`/api/jobs/${encodeURIComponent(id)}`);
      if (job.method && job.method !== method) selectMethod(job.method, false, true);
      message(job.message, ['preparing','running'].includes(job.status) ? 'busy' : job.status === 'completed' ? '' : 'error');
      if (['completed','partial'].includes(job.status)) {
        showResults(job);
        if (announce || sawRunning) ui.toast(job.message,job.status === 'completed' ? 'success' : 'info');
        break;
      }
      if (job.status === 'failed') {feedback(job.message); break;}
      sawRunning = true;
      await sleep(1200);
    }
  } catch (error) {
    if (error.status === 404) {
      sessionStorage.removeItem('music-tagging-job'); $('empty-state').hidden = false;
      feedback('이전 작업은 현재 서버에 없습니다. 저장된 결과는 outputs/local-web 폴더에 남아 있습니다. 새 음원을 선택해주세요.','info');
    } else {
      feedback('서버 연결을 확인해주세요. 서버가 실행 중이면 화면을 새로고침하여 작업을 다시 확인할 수 있습니다. ' + error.message);
    }
  } finally {busy = false; controls();}
}
$('tagging-form').addEventListener('submit', async event => {
  event.preventDefault(); if (busy || !ready) return;
  let body, options;
  if (mode === 'files') {
    const files = [...$('audio-files').files];
    if (!files.length || files.length > 20) {feedback('음원 파일을 1~20개 선택해주세요.'); return;}
    if (files.some(file => !file.size || file.size > 100*1048576) || files.reduce((n,f) => n+f.size,0) > 500*1048576) {
      feedback('빈 파일을 제외하고 파일당 100MiB, 합계 500MiB 이하로 선택해주세요.'); return;
    }
    body = new FormData(); body.append('method',method); for (const file of files) body.append('audio', file);
    options = {method:'POST', body};
  } else {
    const url = $('youtube-url').value.trim();
    if (!url) {feedback('YouTube 영상 한 곡의 링크를 입력해주세요.'); return;}
    options = {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({url,method})};
  }
  busy = true; controls(); message('입력을 준비하고 있습니다.', 'busy');
  $('results').replaceChildren(); $('result-count').textContent = '';
  try {
    const job = await api(`/api/jobs/${mode}`, options);
    sessionStorage.setItem('music-tagging-job', job.id);
    $('results').replaceChildren(); $('downloads').hidden = true; $('result-count').textContent = '';
    await poll(job.id);
  } catch (error) {feedback(error.message);}
  finally {busy = false; controls();}
});
async function initialize() {
  try {
    const state = await api('/api/status'); models = state.models; youtubeState = state.youtube; selectMethod('familiar',false);
    message(state.model.message, ready ? '' : 'error'); $('model-help').hidden = ready; controls();
    const id = state.active_job || sessionStorage.getItem('music-tagging-job');
    if (id) await poll(id,false);
  } catch (error) {feedback('서버에 연결하지 못했습니다. 실행 창을 확인한 뒤 새로고침해주세요.');}
  finally {busy = false; controls();}
}
initialize();

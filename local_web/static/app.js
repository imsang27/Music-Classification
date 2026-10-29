'use strict';
const $ = id => document.getElementById(id);
const ui = window.MusicUI;
const token = document.querySelector('meta[name="app-token"]').content;
const aiFootnote = document.querySelector('footer').textContent;
let mode='files', method='familiar', section='ai', aiMethod='familiar', busy=false, ready=false;
let models={}, youtubeState=null, uploadState=null, previewSequence=0;
const resultByMethod={};
const descriptions={
  familiar:'MuQ · 장르 25개, 감정 9개, 분위기 11개를 전곡에서 비교합니다.',
  wav2vec2:'Wav2Vec2 · 앞부분 최대 30초로 기존 10장르를 분류합니다. 감정·분위기는 미지원입니다.',
  rule:'기존 템포 규칙의 참고 결과입니다. 장르·감정의 정확한 판정으로 해석하지 마세요.',
  manual:'곡 이름과 장르·감정·분위기를 직접 선택하여 기록합니다. 오디오는 분석하지 않습니다.'
};
const api=async(url,options={})=>{
  const response=await fetch(url,{...options,headers:{'X-App-Token':token,...options.headers}});
  const data=await response.json();
  if(!response.ok){const error=new Error(data.error||'요청을 처리하지 못했습니다.');error.status=response.status;throw error;}
  return data;
};
const post=(url,data)=>api(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
function message(text,kind=''){
  if($('status').textContent!==text)$('status').textContent=text;
  $('status').dataset.kind=kind;
}
function feedback(text,kind='error'){message(text,kind==='error'?'error':'');ui.toast(text,kind);}
function controls(){
  $('start').disabled=busy||!ready;
  $('start').firstChild.textContent=busy?(method==='manual'?'저장 중입니다 ':'분류 중입니다 '):(method==='manual'?'선택한 분류 저장 ':'분류 시작 ');
  $('start').setAttribute('aria-busy',String(busy));ui.loading(busy);
  for(const id of ['file-tab','youtube-tab','links-tab','audio-files','youtube-url','batch-urls','familiar-tab','legacy-tab','preview-link','refresh-models','manual-title','manual-genre'])$(id).disabled=busy;
  for(const node of document.querySelectorAll('[data-section],#manual-panel input'))node.disabled=busy;
  if(busy){$('downloads').hidden=true;ui.hideTip();}
}
function paintSection(value){
  section=value;
  for(const button of document.querySelectorAll('[data-section]')){
    const selected=button.dataset.section===value;button.setAttribute('aria-selected',String(selected));button.tabIndex=selected?0:-1;
  }
  $('ai-model-navigation').hidden=value!=='ai';
  $('ai-method-guide').hidden=value!=='ai';
  $('ai-panel').hidden=['uploads','other'].includes(value);
  $('uploads-panel').hidden=value!=='uploads';$('other-panel').hidden=value!=='other';
  if(['uploads','other'].includes(value))document.querySelector('footer').textContent='원본 노트는 수정하지 않습니다. 새 분석용 오디오는 처리 후 삭제합니다.';
  $('input-tabs').hidden=value==='manual';$('manual-panel').hidden=value!=='manual';
  for(const [id,key] of [['file-panel','files'],['youtube-panel','youtube'],['links-panel','links']])$(id).hidden=value==='manual'||mode!==key;
}
function selectMethod(value,announce=true,force=false){
  if(busy&&!force)return;
  ui.hideTip();method=value;
  if(['familiar','wav2vec2'].includes(value)){aiMethod=value;paintSection('ai');}else paintSection(value);
  for(const [id,key] of [['familiar-tab','familiar'],['legacy-tab','wav2vec2']]){
    $(id).setAttribute('aria-selected',String(value===key));$(id).tabIndex=value===key?0:-1;
  }
  $('ai-panel').setAttribute('aria-labelledby',value==='familiar'?'familiar-tab':value==='wav2vec2'?'legacy-tab':`${value}-nav`);
  $('method-description').textContent=descriptions[value];
  document.querySelector('.privacy').textContent=value==='manual'?'직접 선택한 분류만 저장합니다. 오디오 분석·모델 학습·원본 수정을 하지 않습니다.':'원본 음원과 Obsidian 노트는 수정하지 않습니다. 가사 의미를 해석하지 않고 소리의 인상을 분석합니다.';
  $('empty-copy').textContent=value==='manual'?'직접 선택한 분류가 여기에 표시됩니다.':value==='rule'?'템포에 따른 참고 결과를 표시합니다. 분위기는 미지원입니다.':value==='wav2vec2'?'기존 10장르 중 가까운 장르를 보여드립니다.':'감정과 분위기를 각각 세 가지 후보로 보여드립니다.';
  document.querySelector('footer').textContent=value==='manual'?'직접 기록한 선택은 모델 예측이나 학습용 정답 인증이 아닙니다.':value==='rule'?'템포 규칙은 참고 기능이며 정확한 장르·감정이나 확률을 제공하지 않습니다.':aiFootnote;
  ready=Boolean(models[value]?.ready);$('model-help').hidden=ready||value!=='familiar';controls();
  if(announce)message(models[value]?.message||'실행 환경을 확인하고 있습니다.',ready?'':'error');
  if(!busy){
    if(resultByMethod[value])showResults(resultByMethod[value]);
    else{$('results').replaceChildren();$('downloads').hidden=true;$('result-count').textContent='';$('empty-state').hidden=false;}
  }
  if(announce)ui.enter($('ai-panel'));
}
function selectSection(value){
  if(busy)return;
  paintSection(value);
  if(value==='ai')selectMethod(aiMethod);
  else if(['rule','manual'].includes(value))selectMethod(value);
  else if(value==='uploads')refreshUploads();
  else ui.enter($('other-panel'));
}
function tabKeyboard(event,buttons,select){
  if(busy||!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
  event.preventDefault();const index=buttons.indexOf(event.currentTarget);
  const next=event.key==='Home'?0:event.key==='End'?buttons.length-1:(index+(event.key==='ArrowRight'?1:-1)+buttons.length)%buttons.length;
  select(buttons[next]);buttons[next].focus();
}
const mainTabs=[...document.querySelectorAll('[data-section]')];
for(const button of mainTabs){button.addEventListener('click',()=>selectSection(button.dataset.section));button.addEventListener('keydown',e=>tabKeyboard(e,mainTabs,b=>selectSection(b.dataset.section)));}
const modelTabs=[$('familiar-tab'),$('legacy-tab')];
for(const button of modelTabs){button.addEventListener('click',()=>selectMethod(button.dataset.method));button.addEventListener('keydown',e=>tabKeyboard(e,modelTabs,b=>selectMethod(b.dataset.method)));}
function chooseMode(value){
  if(busy)return;mode=value;paintSection(section);
  for(const [id,key] of [['file-tab','files'],['youtube-tab','youtube'],['links-tab','links']])$(id).setAttribute('aria-selected',String(value===key));
  ui.enter($(value==='files'?'file-panel':value==='youtube'?'youtube-panel':'links-panel'));
}
for(const [id,key] of [['file-tab','files'],['youtube-tab','youtube'],['links-tab','links']])$(id).addEventListener('click',()=>chooseMode(key));
function listFiles(){
  $('file-list').replaceChildren();
  for(const file of $('audio-files').files){const li=document.createElement('li');li.textContent=`${file.name} · ${(file.size/1048576).toFixed(1)} MiB`;$('file-list').append(li);}
  const count=$('audio-files').files.length;
  if(count){ui.toast(`${count}개 음원을 선택했습니다.`);ui.enter($('file-list'));}
}
$('audio-files').addEventListener('change',listFiles);
$('drop-zone').addEventListener('dragover',event=>{event.preventDefault();if(!busy)$('drop-zone').classList.add('dragging');});
$('drop-zone').addEventListener('dragleave',()=>$('drop-zone').classList.remove('dragging'));
$('drop-zone').addEventListener('drop',event=>{event.preventDefault();$('drop-zone').classList.remove('dragging');if(!busy&&event.dataTransfer.files.length){$('audio-files').files=event.dataTransfer.files;listFiles();}});
function showResults(job){
  resultByMethod[job.method||'familiar']=job;ui.loading(false);$('empty-state').hidden=true;$('results').replaceChildren();$('downloads').replaceChildren();
  const rows=job.result.results,source=document.createElement('p');source.className='hint';
  source.textContent={familiar:'현재 AI · MuQ · 전곡 구간 종합',wav2vec2:'기존 AI · Wav2Vec2 · 앞부분 최대 30초',rule:'기존 규칙 · 템포 기반 · 앞부분 최대 60초',manual:'직접 분류 · 사용자 선택 기록'}[job.method||'familiar'];
  $('results').append(source);$('result-count').textContent=`${rows.length}곡`;
  for(const row of rows){
    const card=document.createElement('article');card.className='song';const title=document.createElement('h3');title.textContent=row.display_name||'음원';card.append(title);
    if(!row.labels){const error=document.createElement('p');error.className='error-note';error.textContent='이 입력은 처리하지 못했습니다. 정상적인 음원 또는 공개 링크인지 확인해주세요.';card.append(error);}
    else for(const [axis,name] of [['genre','장르'],['emotion','감정'],['mood','분위기']]){
      const line=document.createElement('div');line.className='axis';const label=document.createElement('span');label.className='axis-label';label.textContent=name;
      const chips=document.createElement('div');chips.className='chips';const candidates=(row.candidates[axis]||[]).slice(0,axis==='genre'?1:3);
      if(!candidates.length){const note=document.createElement('span');note.className='unsupported';note.textContent='이 방식에서는 미지원';chips.append(note);}
      for(const candidate of candidates){const chip=document.createElement('span');chip.className='chip'+(candidate.representative?' primary-chip':'');const rank=document.createElement('small');rank.textContent=job.method==='manual'?'선택':job.method==='rule'?'규칙':candidate.representative?'대표':`${candidate.rank}순위`;chip.append(rank,document.createTextNode(candidate.name));chips.append(chip);}
      line.append(label,chips);card.append(line);
    }
    if(row.tempo_bpm){const tempo=document.createElement('p');tempo.className='hint';tempo.textContent=`추정 템포 ${row.tempo_bpm.toFixed(1)} BPM · 규칙 참고용`;card.append(tempo);}
    $('results').append(card);
  }
  for(const [name,label] of [['태깅결과.md','Markdown 저장'],['tags.json','태그 JSON 저장'],['classification.json','상세 결과 저장']]){
    const link=document.createElement('a');link.href=job.downloads[name];link.textContent=label;link.setAttribute('download',name);
    link.dataset.tooltip=name==='태깅결과.md'?'읽기 쉬운 표 형식의 문서입니다. 노트에 적용하기 전 내용을 확인하세요.':name==='tags.json'?'분류 항목과 후보를 재사용할 수 있는 JSON 파일입니다.':'모델 응답·분석 범위 또는 직접 선택한 내용의 상세 기록입니다.';
    link.addEventListener('click',()=>ui.toast('결과 파일 다운로드를 요청했습니다.'));$('downloads').append(link);
  }
  $('downloads').hidden=false;ui.enter($('results'));
}
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function poll(id,announce=true){
  busy=true;controls();let sawRunning=false;
  try{
    while(true){
      const job=await api(`/api/jobs/${encodeURIComponent(id)}`);
      if(job.method&&job.method!==method)selectMethod(job.method,false,true);
      message(job.message,['preparing','running'].includes(job.status)?'busy':job.status==='completed'?'':'error');
      if(['completed','partial'].includes(job.status)){showResults(job);if(announce||sawRunning)ui.toast(job.message,job.status==='completed'?'success':'info');break;}
      if(job.status==='failed'){feedback(job.message);break;}
      sawRunning=true;await sleep(1200);
    }
  }catch(error){
    if(error.status===404){sessionStorage.removeItem('music-tagging-job');feedback('이전 작업은 현재 서버에 없습니다. 저장된 결과는 outputs/local-web 폴더에 남아 있습니다.','info');}
    else feedback('서버 연결을 확인하고 새로고침해주세요. '+error.message);
  }finally{busy=false;controls();}
}
$('tagging-form').addEventListener('submit',async event=>{
  event.preventDefault();if(busy||!ready)return;
  let options,endpoint=mode;
  if(method==='manual'){
    const labels={genre:[$('manual-genre').value],emotion:[...$('manual-emotion').querySelectorAll('input:checked')].map(i=>i.value),mood:[...$('manual-mood').querySelectorAll('input:checked')].map(i=>i.value)};
    const title=$('manual-title').value.trim();
    if(!title||!labels.genre[0]||!labels.emotion.length||!labels.mood.length){feedback('곡 이름과 장르·감정·분위기를 선택해주세요.');return;}
    endpoint='manual';options={method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({title,labels})};
  }else if(mode==='files'){
    const files=[...$('audio-files').files];
    if(!files.length||files.length>20){feedback('음원 파일을 1~20개 선택해주세요.');return;}
    if(files.some(file=>!file.size||file.size>100*1048576)||files.reduce((n,f)=>n+f.size,0)>500*1048576){feedback('파일당 100MiB, 합계 500MiB 이하의 음원을 선택해주세요.');return;}
    const body=new FormData();body.append('method',method);for(const file of files)body.append('audio',file);options={method:'POST',body};
  }else if(mode==='links'){
    const urls=$('batch-urls').value.split(/\r?\n|,(?=\s*https?:\/\/)/).map(v=>v.trim()).filter(Boolean);
    if(!urls.length||urls.length>10){feedback('링크를 한 줄에 하나씩, 최대 10개 입력해주세요.');return;}
    options={method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({urls,method})};
  }else{
    const url=$('youtube-url').value.trim();if(!url){feedback('YouTube 또는 HTTPS 음원 링크를 입력해주세요.');return;}
    options={method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url,method})};
  }
  busy=true;controls();message('입력을 준비하고 있습니다.','busy');$('results').replaceChildren();$('result-count').textContent='';
  try{const job=await api(`/api/jobs/${endpoint}`,options);sessionStorage.setItem('music-tagging-job',job.id);await poll(job.id);}
  catch(error){feedback(error.message);}finally{busy=false;controls();}
});
function showModelStatus(){
  $('model-overview').textContent=`MuQ: ${models.familiar?.ready?'준비됨':'설정 필요'} · Wav2Vec2: ${models.wav2vec2?.ready?'준비됨':'설정 필요'}`;
}
$('refresh-models').addEventListener('click',async()=>{
  $('refresh-models').disabled=true;$('model-overview').classList.add('preview-skeleton');
  try{const state=await api('/api/status');models=state.models;youtubeState=state.youtube;showModelStatus();if(!['uploads','other'].includes(section))selectMethod(method,false);ui.toast('모델 준비 상태를 새로고침했습니다.');}
  catch(error){feedback(error.message);}finally{$('refresh-models').disabled=busy;$('model-overview').classList.remove('preview-skeleton');}
});
async function loadVocabulary(){
  const vocabulary=await api('/api/vocabulary');
  for(const name of vocabulary.genre){const option=document.createElement('option');option.value=name;option.textContent=name;$('manual-genre').append(option);}
  for(const axis of ['emotion','mood'])for(const name of vocabulary[axis]){
    const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.value=name;
    input.addEventListener('change',()=>{if($(`manual-${axis}`).querySelectorAll('input:checked').length>3){input.checked=false;ui.toast('각 항목은 최대 3개 선택할 수 있습니다.');}});
    label.append(input,document.createTextNode(name));$(`manual-${axis}`).append(label);
  }
}
$('youtube-url').addEventListener('input',()=>{previewSequence++;$('link-preview').hidden=true;});
$('preview-link').addEventListener('click',async()=>{
  const url=$('youtube-url').value.trim();if(!url){feedback('미리 볼 링크를 입력해주세요.');return;}
  const sequence=++previewSequence,preview=$('link-preview');$('preview-link').disabled=true;preview.hidden=false;preview.textContent='';preview.classList.add('preview-skeleton');
  try{const result=await post('/api/link_preview',{url});if(sequence===previewSequence){preview.textContent=[result.title||result.message,result.channel,result.duration_seconds?`${Math.round(result.duration_seconds)}초`:null,result.url].filter(Boolean).join(' · ');ui.enter(preview);}}
  catch(error){if(sequence===previewSequence){preview.textContent=error.message;feedback(error.message);}}
  finally{if(sequence===previewSequence)preview.classList.remove('preview-skeleton');$('preview-link').disabled=busy;}
});
function applyTheme(dark){document.body.classList.toggle('dark-mode',dark);$('theme-toggle').setAttribute('aria-pressed',String(dark));$('theme-toggle').textContent=dark?'라이트 모드':'다크 모드';try{localStorage.setItem('music-theme',dark?'dark':'light');}catch{}}
try{applyTheme(localStorage.getItem('music-theme')==='dark');}catch{applyTheme(false);}
$('theme-toggle').addEventListener('click',()=>applyTheme(!document.body.classList.contains('dark-mode')));
async function refreshUploads(announce=false){
  $('uploads-loading').hidden=false;$('refresh-uploads').disabled=true;$('recycle-uploads').disabled=true;
  try{
    uploadState=await api('/api/uploads');$('uploads-list').replaceChildren();
    $('uploads-summary').textContent=`보관 파일 ${uploadState.count}개 · ${(uploadState.bytes/1048576).toFixed(1)} MiB`+(uploadState.active_job?' · 현재 분류 진행 중':'');
    for(const row of uploadState.files){const li=document.createElement('li');li.textContent=`${row.name} · ${(row.bytes/1048576).toFixed(1)} MiB`;$('uploads-list').append(li);}
    if(!uploadState.count){const li=document.createElement('li');li.textContent='보관된 업로드 파일이 없습니다. 새 분석용 오디오는 처리 후 삭제됩니다.';$('uploads-list').append(li);}
    $('recycle-uploads').disabled=!uploadState.count||Boolean(uploadState.active_job);
    $('restore-uploads').hidden=!sessionStorage.getItem('music-upload-restore');
    if(announce)ui.toast('업로드 목록을 새로고침했습니다.');
  }catch(error){feedback(error.message);$('uploads-summary').textContent='목록을 불러오지 못했습니다.';}
  finally{$('uploads-loading').hidden=true;$('refresh-uploads').disabled=false;}
}
$('refresh-uploads').addEventListener('click',()=>refreshUploads(true));
$('recycle-uploads').addEventListener('click',()=>$('recycle-dialog').showModal());
$('cancel-recycle').addEventListener('click',()=>$('recycle-dialog').close());
$('confirm-recycle').addEventListener('click',async()=>{
  $('confirm-recycle').disabled=true;
  try{const result=await post('/api/uploads/recycle',{version:uploadState.version});sessionStorage.setItem('music-upload-restore',result.id);$('recycle-dialog').close();ui.toast(result.message,'success');await refreshUploads();}
  catch(error){feedback(error.message);$('recycle-dialog').close();await refreshUploads();}
  finally{$('confirm-recycle').disabled=false;}
});
$('restore-uploads').addEventListener('click',async()=>{
  $('restore-uploads').disabled=true;
  try{const result=await post('/api/uploads/restore',{id:sessionStorage.getItem('music-upload-restore')});sessionStorage.removeItem('music-upload-restore');ui.toast(result.message,'success');await refreshUploads();}
  catch(error){feedback(error.message);}finally{$('restore-uploads').disabled=false;}
});
async function initialize(){
  try{const state=await api('/api/status');models=state.models;youtubeState=state.youtube;showModelStatus();selectMethod('familiar',false);loadVocabulary().catch(error=>feedback(error.message));message(state.model.message,ready?'':'error');
    const query=new URLSearchParams(location.search);if(['files','youtube','links'].includes(query.get('input')))chooseMode(query.get('input'));if(['manual','rule','uploads','other'].includes(query.get('section')))selectSection(query.get('section'));
    const id=state.active_job||sessionStorage.getItem('music-tagging-job');if(id)await poll(id,false);}
  catch(error){feedback('서버에 연결하지 못했습니다. 실행 창을 확인한 뒤 새로고침해주세요.');}
  finally{busy=false;controls();}
}
initialize();

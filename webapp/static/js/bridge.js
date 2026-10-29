/* Preserve the original home screen; AI buttons navigate to the dedicated page. */
'use strict';
const byId=id=>document.getElementById(id);
const appToken=document.querySelector('meta[name="app-token"]').content;
let working=false,inventory=null,noticeTimer;
async function classicApi(url,options={}){
  const response=await fetch(url,{...options,headers:{'X-App-Token':appToken,...options.headers}});
  const data=await response.json();if(!response.ok)throw new Error(data.error||'요청을 처리하지 못했습니다.');return data;
}
const classicPost=(url,data)=>classicApi(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
function notify(text){const node=byId('classic-notice');node.textContent=text;node.hidden=false;clearTimeout(noticeTimer);noticeTimer=setTimeout(()=>node.hidden=true,8000);}
function setWorking(value){
  working=value;
  for(const button of document.querySelectorAll('button'))if(!['ml','lyrics','hybrid'].includes(button.value))button.disabled=value;
  if(!value)byId('classic-recycle').disabled=!inventory?.count;
}
for(const button of document.querySelectorAll('[data-open-ai]'))button.addEventListener('click',()=>{
  const form=button.closest('form');const input=form.id==='batch-form'?'links':form.id==='url-form'?'youtube':'files';
  window.location.assign('/ai?input='+input);
});
for(const button of document.querySelectorAll('[data-open-manual]'))button.addEventListener('click',()=>window.location.assign('/ai?section=manual'));
byId('random-cta').textContent='음악을 선택하고 원하는 분류 방식을 이용하세요.';
byId('audio_file').addEventListener('change',()=>{byId('file-name').textContent=byId('audio_file').files[0]?.name||'';});
function setTheme(white){
  document.body.classList.toggle('white-mode',white);document.body.classList.toggle('dark-mode',!white);
  byId('classic-theme').textContent=white?'다크 모드':'화이트 모드';
  try{localStorage.setItem('music-theme',white?'light':'dark');}catch{}
}
let initialWhite=false;try{initialWhite=localStorage.getItem('music-theme')==='light';}catch{}
setTheme(initialWhite);byId('classic-theme').addEventListener('click',()=>setTheme(!document.body.classList.contains('white-mode')));
async function modelState(){
  try{const state=await classicApi('/api/status');const names=[];if(state.models.familiar.ready)names.push('MuQ');if(state.models.wav2vec2.ready)names.push('Wav2Vec2');
    byId('model-name').textContent=names.join(' · ');byId('model-name').hidden=!names.length;
    const label=document.querySelector('.status-value');label.textContent=names.length?'준비됨':'설정 필요';label.classList.toggle('status-success',Boolean(names.length));label.classList.toggle('status-error',!names.length);
  }catch(error){notify(error.message);}
}
byId('classic-model-refresh').addEventListener('click',()=>modelState());
byId('classic-preview').addEventListener('click',async()=>{
  const url=byId('url_input').value.trim();if(!url){notify('미리 볼 링크를 입력해주세요.');return;}
  byId('classic-preview').disabled=true;byId('link-preview').hidden=false;byId('preview-info').textContent='영상 정보를 확인하고 있습니다.';
  try{const result=await classicPost('/api/link_preview',{url});byId('preview-platform').textContent=result.kind==='youtube'?'YouTube':'음원 링크';byId('preview-info').textContent=[result.title||result.message,result.channel,result.duration_seconds?`${Math.round(result.duration_seconds)}초`:null].filter(Boolean).join(' · ');}
  catch(error){byId('preview-info').textContent=error.message;notify(error.message);}finally{byId('classic-preview').disabled=working;}
});
byId('classic-preview-close').addEventListener('click',()=>byId('link-preview').hidden=true);
async function uploadsState(){
  try{inventory=await classicApi('/api/uploads');byId('uploads-stats').textContent=`보관 파일 ${inventory.count}개 · ${(inventory.bytes/1048576).toFixed(1)} MiB`;
    const list=byId('uploads-files');list.replaceChildren();for(const row of inventory.files){const line=document.createElement('p');line.textContent=row.name;list.append(line);}list.hidden=!inventory.count;
    byId('classic-recycle').disabled=working||!inventory.count||Boolean(inventory.active_job);byId('classic-restore').hidden=!sessionStorage.getItem('music-upload-restore');
  }catch(error){byId('uploads-stats').textContent='목록을 가져오지 못했습니다.';notify(error.message);}
}
byId('classic-uploads-refresh').addEventListener('click',()=>uploadsState());
byId('classic-recycle').addEventListener('click',async()=>{
  if(!inventory?.count||!confirm('현재 목록의 보관 음원을 복구 가능한 보관함으로 이동할까요? 원본 노트와 분석 결과는 변경하지 않습니다.'))return;
  try{const result=await classicPost('/api/uploads/recycle',{version:inventory.version});sessionStorage.setItem('music-upload-restore',result.id);notify(result.message);await uploadsState();}catch(error){notify(error.message);}
});
byId('classic-restore').addEventListener('click',async()=>{
  try{const result=await classicPost('/api/uploads/restore',{id:sessionStorage.getItem('music-upload-restore')});sessionStorage.removeItem('music-upload-restore');notify(result.message);await uploadsState();}catch(error){notify(error.message);}
});
function renderClassic(job){
  const body=byId('classic-result-body');body.replaceChildren();
  for(const row of job.result.results){const card=document.createElement('div');card.className='result-group';const title=document.createElement('h3');title.textContent=row.display_name||'음원';card.append(title);
    const summary=document.createElement('p');summary.textContent=row.labels?`장르: ${row.labels.genre.join(' · ')} / 감정: ${row.labels.emotion.join(' · ')} / 분위기: 미지원`:'이 입력을 처리하지 못했습니다.';card.append(summary);
    if(row.tempo_bpm){const detail=document.createElement('p');detail.textContent=`추정 템포: ${row.tempo_bpm.toFixed(1)} BPM · 기존 규칙의 참고 결과`;card.append(detail);}body.append(card);
  }
  for(const [name,label] of [['태깅결과.md','Markdown 저장'],['tags.json','JSON 저장']]){const link=document.createElement('a');link.href=job.downloads[name];link.textContent=label;link.className='classic-download';link.download=name;body.append(link);}
}
async function runRule(options){
  if(working)return;setWorking(true);byId('classic-results').hidden=false;byId('classic-result-body').replaceChildren();byId('classic-result-body').classList.add('classic-skeleton');byId('classic-status').textContent='기존 템포 규칙으로 분류하고 있습니다.';
  try{const created=await classicApi(options.url,options.request);while(true){const job=await classicApi('/api/jobs/'+created.id);byId('classic-status').textContent=job.message;
      if(['completed','partial'].includes(job.status)){renderClassic(job);notify(job.message);break;}
      if(job.status==='failed'){notify(job.message);break;}await new Promise(resolve=>setTimeout(resolve,1200));}
  }catch(error){byId('classic-status').textContent=error.message;notify(error.message);}finally{byId('classic-result-body').classList.remove('classic-skeleton');setWorking(false);}
}
byId('file-form').addEventListener('submit',event=>{
  event.preventDefault();const file=byId('audio_file').files[0];if(!file){notify('음원 파일을 선택해주세요.');return;}
  const body=new FormData();body.append('audio',file);body.append('method','rule');runRule({url:'/api/jobs/files',request:{method:'POST',body}});
});
byId('url-form').addEventListener('submit',event=>{
  event.preventDefault();const url=byId('url_input').value.trim();if(!url){notify('음원 링크를 입력해주세요.');return;}
  runRule({url:'/api/jobs/youtube',request:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url,method:'rule'})}});
});
byId('batch-form').addEventListener('submit',event=>{event.preventDefault();window.location.assign('/ai?input=links');});
modelState();uploadsState();

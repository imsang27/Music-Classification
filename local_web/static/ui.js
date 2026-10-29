'use strict';
(() => {
  const reducedMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
  const stack = document.getElementById('toast-stack');
  const announcer = document.getElementById('toast-announcer');
  const notices = new Map();
  function toast(text, kind = 'info') {
    const key = `${kind}:${text}`;
    if (notices.has(key)) return;
    const card = document.createElement('div'); card.className = `toast toast-${kind}`;
    const icon = document.createElement('span'); icon.className = 'toast-icon'; icon.setAttribute('aria-hidden','true');
    icon.textContent = kind === 'success' ? '✓' : kind === 'error' ? '!' : 'i';
    const content = document.createElement('p'); content.textContent = text;
    const close = document.createElement('button'); close.type = 'button'; close.textContent = '×'; close.setAttribute('aria-label','알림 닫기');
    card.append(icon, content, close); stack.append(card); notices.set(key,card);
    announcer.textContent = text;
    let timer, dismissed = false;
    const remove = () => {
      if (dismissed) return;
      dismissed = true; clearTimeout(timer); notices.delete(key);
      if (card.contains(document.activeElement)) document.getElementById('start').focus();
      card.classList.add('toast-leaving');
      setTimeout(() => card.remove(), reducedMotion() ? 0 : 180);
    };
    const pause = () => clearTimeout(timer);
    const resume = () => {
      pause();
      if (!dismissed && !card.matches(':hover') && !card.contains(document.activeElement)) timer = setTimeout(remove, kind === 'error' ? 12000 : 6000);
    };
    close.addEventListener('click',remove);
    card.addEventListener('pointerenter',pause); card.addEventListener('pointerleave',resume);
    card.addEventListener('focusin',pause); card.addEventListener('focusout',resume);
    resume();
    if (notices.size > 3) {
      const oldest = [...notices.entries()].find(([,node]) => !node.contains(document.activeElement));
      if (oldest) {oldest[1].querySelector('button').click();}
    }
  }
  function enter(node) {
    if (!node || reducedMotion()) return;
    node.getAnimations().forEach(animation => animation.cancel());
    node.animate([{opacity:0.35,transform:'translateY(5px)'},{opacity:1,transform:'translateY(0)'}],
      {duration:180,easing:'ease-out'});
  }
  function loading(value) {
    const skeleton = document.getElementById('loading-skeleton');
    skeleton.hidden = !value;
    const results = document.getElementById('results'); results.hidden = value;
    document.querySelector('.result-panel').setAttribute('aria-busy',String(value));
    if (value) document.getElementById('empty-state').hidden = true;
    else if (!results.childElementCount) document.getElementById('empty-state').hidden = false;
  }

  const tip = document.getElementById('ui-tooltip');
  let trigger = null, hideTimer;
  function hideTip() {
    clearTimeout(hideTimer);
    if (trigger) {
      const ids = (trigger.getAttribute('aria-describedby') || '').split(/\s+/).filter(id => id && id !== tip.id);
      if (ids.length) trigger.setAttribute('aria-describedby',ids.join(' ')); else trigger.removeAttribute('aria-describedby');
    }
    trigger = null; tip.hidden = true;
  }
  function showTip(target) {
    clearTimeout(hideTimer);
    if (trigger !== target) hideTip();
    trigger = target; tip.textContent = target.dataset.tooltip; tip.hidden = false;
    const ids = new Set((target.getAttribute('aria-describedby') || '').split(/\s+/).filter(Boolean)); ids.add(tip.id);
    target.setAttribute('aria-describedby',[...ids].join(' '));
    const rect = target.getBoundingClientRect(), box = tip.getBoundingClientRect();
    tip.style.left = `${Math.max(12,Math.min(innerWidth-box.width-12,rect.left+rect.width/2-box.width/2))}px`;
    tip.style.top = `${rect.bottom+box.height+16 < innerHeight ? rect.bottom+8 : Math.max(8,rect.top-box.height-8)}px`;
  }
  const laterHide = () => {clearTimeout(hideTimer);hideTimer = setTimeout(hideTip,120);};
  document.addEventListener('pointerover',event => {
    const target = event.target.closest('[data-tooltip]');
    if (target && !target.contains(event.relatedTarget)) showTip(target);
  });
  document.addEventListener('pointerout',event => {
    if (trigger && trigger.contains(event.target) && !trigger.contains(event.relatedTarget)) laterHide();
  });
  document.addEventListener('focusin',event => {const target = event.target.closest('[data-tooltip]'); if (target) showTip(target);});
  document.addEventListener('focusout',event => {if (trigger?.contains(event.target)) laterHide();});
  document.addEventListener('click',event => {
    const target = event.target.closest('.help-trigger');
    if (target) showTip(target);
    else if (!event.target.closest('[data-tooltip]')) hideTip();
  });
  document.addEventListener('keydown',event => {if (event.key === 'Escape') hideTip();});
  tip.addEventListener('pointerenter',() => clearTimeout(hideTimer)); tip.addEventListener('pointerleave',laterHide);
  window.addEventListener('scroll',hideTip,{passive:true}); window.addEventListener('resize',hideTip);
  window.MusicUI = {toast,enter,loading,hideTip};
})();

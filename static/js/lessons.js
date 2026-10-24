/* static/js/lessons.js
   Usage: openLesson(careerCode, skillName)   e.g. openLesson('15-2051.00', 'Python')
   Shows AI lesson + real links, then a 5-question quiz. */
(function () {
  const css = `
  .lz-back{position:fixed;inset:0;z-index:200;background:rgba(3,12,22,.7);backdrop-filter:blur(4px);display:none;align-items:flex-start;justify-content:center;overflow:auto;padding:5vh 4vw}
  .lz-back.on{display:flex}
  .lz{width:min(780px,100%);background:#0a2d52;border:1px solid var(--line);border-radius:22px;padding:24px;position:relative}
  .lz h2{font:800 1.4rem var(--display);margin-bottom:4px}
  .lz h3{font:800 1rem var(--display);margin:18px 0 8px}
  .lz .x{position:absolute;right:14px;top:12px;background:none;border:0;color:#fff;font-size:1.6rem;cursor:pointer}
  .lz .tabs2{display:flex;gap:6px;border-bottom:1px solid var(--line);margin:14px 0}
  .lz .tabs2 button{background:none;border:0;color:var(--muted);padding:9px 16px;font-weight:700;cursor:pointer;border-bottom:3px solid transparent}
  .lz .tabs2 button.on{color:#fff;border-color:var(--teal)}
  .lz ol{padding-left:20px;display:grid;gap:6px}
  .lz .res{display:flex;gap:12px;align-items:center;padding:12px 14px;border:1px solid var(--line);border-radius:14px;background:rgba(0,0,0,.18);margin-bottom:8px;transition:transform .2s,border-color .2s}
  .lz .res:hover{transform:translateX(4px);border-color:var(--teal)}
  .lz .res b{display:block}
  .lz .res small{color:var(--muted)}
  .lz .pv{flex:none;font-size:.7rem;font-weight:700;text-transform:uppercase;padding:4px 9px;border-radius:99px;background:rgba(0,168,150,.25);color:#7fe9dc}
  .lz .q{padding:14px;border:1px solid var(--line);border-radius:14px;margin-bottom:12px;background:rgba(0,0,0,.16)}
  .lz .q label{display:block;padding:8px 10px;border-radius:10px;cursor:pointer;margin-top:4px}
  .lz .q label:hover{background:rgba(255,255,255,.06)}
  .lz .q.right label.ok{background:rgba(18,181,123,.25)}
  .lz .q label.bad{background:rgba(230,70,70,.25)}
  .lz .exp{margin-top:8px;font-size:.88rem;color:var(--muted)}
  .lz .msg{padding:14px;border:1.5px dashed var(--line);border-radius:14px;color:var(--muted);text-align:center}
  .lz .score{font:800 1.3rem var(--display);margin:6px 0 14px}`;
  const st = document.createElement('style'); st.textContent = css; document.head.appendChild(st);

  const back = document.createElement('div');
  back.className = 'lz-back';
  back.innerHTML = '<div class="lz" role="dialog" aria-modal="true"><button class="x" aria-label="Close">×</button><div id="lzBody"></div></div>';
  document.body.appendChild(back);
  const body = back.querySelector('#lzBody');
  const close = () => back.classList.remove('on');
  back.querySelector('.x').onclick = close;
  back.addEventListener('click', e => { if (e.target === back) close(); });
  addEventListener('keydown', e => { if (e.key === 'Escape') close(); });

  const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const icon = { video: '▶', article: '📄', course: '🎓', docs: '📚' };

  async function api(url, opts) {
    const r = await fetch(url, opts);
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || 'Something went wrong.');
    return j;
  }

  window.openLesson = async function (code, skill) {
    back.classList.add('on');
    body.innerHTML = `<h2>${esc(skill)}</h2><p class="msg">Your AI tutor is preparing this lesson…</p>`;
    let lesson;
    try {
      lesson = await api(`/api/reskilling/lesson?code=${encodeURIComponent(code)}&skill=${encodeURIComponent(skill)}`);
    } catch (e) {
      body.innerHTML = `<h2>${esc(skill)}</h2><p class="msg">${esc(e.message)}</p>`;
      return;
    }
    body.innerHTML = `
      <h2>${esc(lesson.skill)}</h2>
      <small style="color:var(--muted)">For your path to: ${esc(lesson.career)}</small>
      <div class="tabs2"><button class="on" data-t="l">Lesson</button><button data-t="q">Quiz</button></div>
      <div id="lzL">
        <p>${esc(lesson.summary)}</p>
        <h3>How to learn it</h3>
        <ol>${lesson.steps.map(s => `<li>${esc(s)}</li>`).join('')}</ol>
        ${lesson.practice ? `<h3>Practice task</h3><p>${esc(lesson.practice)}</p>` : ''}
        <h3>Tutorials &amp; videos</h3>
        ${lesson.resources.map(r => `
          <a class="res" href="${esc(r.url)}" target="_blank" rel="noopener noreferrer">
            <span>${icon[r.type] || '🔗'}</span>
            <span style="flex:1"><b>${esc(r.title)}</b><small>${esc(r.why)}</small></span>
            <span class="pv">${esc(r.provider)}</span>
          </a>`).join('')}
        <p style="margin-top:14px"><button class="btn" id="lzToQuiz" type="button">I've finished — take the quiz</button></p>
      </div>
      <div id="lzQ" style="display:none"></div>`;

    const tabs = body.querySelectorAll('.tabs2 button');
    const showTab = t => {
      tabs.forEach(b => b.classList.toggle('on', b.dataset.t === t));
      body.querySelector('#lzL').style.display = t === 'l' ? '' : 'none';
      body.querySelector('#lzQ').style.display = t === 'q' ? '' : 'none';
      if (t === 'q' && !body.querySelector('#lzQ').dataset.loaded) loadQuiz(code, skill);
    };
    tabs.forEach(b => b.onclick = () => showTab(b.dataset.t));
    body.querySelector('#lzToQuiz').onclick = () => showTab('q');
  };

  async function loadQuiz(code, skill, refresh) {
    const box = body.querySelector('#lzQ');
    box.dataset.loaded = '1';
    box.innerHTML = '<p class="msg">Writing your quiz…</p>';
    let qs;
    try {
      qs = (await api('/api/reskilling/quiz', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, skill, refresh: !!refresh })
      })).questions;
    } catch (e) {
      box.dataset.loaded = '';
      box.innerHTML = `<p class="msg">${esc(e.message)}</p>`;
      return;
    }
    box.innerHTML = qs.map((q, i) => `
      <div class="q" data-i="${i}">
        <b>${i + 1}. ${esc(q.question)}</b>
        ${q.options.map((o, j) => `<label><input type="radio" name="q${i}" value="${j}"> ${esc(o)}</label>`).join('')}
        <div class="exp" hidden></div>
      </div>`).join('') +
      '<button class="btn" id="lzSubmit" type="button">Submit answers</button>';

    box.querySelector('#lzSubmit').onclick = async () => {
      let score = 0;
      const picks = qs.map((_, i) => box.querySelector(`input[name=q${i}]:checked`));
      if (picks.some(p => !p)) { alert('Please answer every question first.'); return; }
      qs.forEach((q, i) => {
        const card = box.querySelector(`.q[data-i="${i}"]`);
        const labels = card.querySelectorAll('label');
        const chosen = +picks[i].value;
        labels[q.answer].classList.add('ok');
        if (chosen === q.answer) { score++; card.classList.add('right'); }
        else labels[chosen].classList.add('bad');
        card.querySelectorAll('input').forEach(x => x.disabled = true);
        const ex = card.querySelector('.exp'); ex.hidden = false; ex.textContent = q.explanation;
      });
      const pass = score / qs.length >= 0.6;
      const head = document.createElement('div');
      head.innerHTML = `<div class="score">${score} / ${qs.length} — ${pass ? '🎉 Lesson complete!' : 'Almost there, review the lesson and retry.'}</div>`;
      box.prepend(head);
      box.querySelector('#lzSubmit').outerHTML =
        '<button class="btn" id="lzRetry" type="button">Try a new quiz</button>';
      box.querySelector('#lzRetry').onclick = () => loadQuiz(code, skill, true);
      box.scrollIntoView({ behavior: 'smooth' });
      try {
        await api('/api/reskilling/quiz/result', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ code, skill, score, total: qs.length })
        });
      } catch (_) { /* score still shown even if saving fails */ }
      document.dispatchEvent(new CustomEvent('lesson-done', { detail: { code, skill, pass } }));
    };   // closes the onclick handler
  }      // closes loadQuiz
})();    // closes the IIFE
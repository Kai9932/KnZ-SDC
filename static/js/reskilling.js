(() => {
  const q = s => document.querySelector(s);
  const qa = s => [...document.querySelectorAll(s)];
  const esc = s => String(s ?? '').replace(/[&<>"']/g,
    c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

  const st = document.createElement('style');
  st.textContent = `
    .kzic{display:grid;place-items:center;font-style:normal;font-size:1.3rem;border-style:solid!important;background:rgba(255,255,255,.05)}
    .m{color:var(--muted);font-size:.85rem}
    .fill{height:100%;border-radius:6px;background:linear-gradient(90deg,var(--teal),var(--green))}
    .gaprow{display:flex;justify-content:space-between;gap:10px;font-size:.9rem}
    .gaprow small{color:var(--muted);white-space:nowrap}
    .note.err{color:#ffb4b4;border-color:#ff8a8a}
    .note.ok{border-style:solid}
    .phase-n{font-weight:700;font-size:.92rem}
    .modrow{cursor:pointer;border-radius:10px;padding:8px 6px}
    .modrow:hover,.modrow.cur{background:rgba(255,255,255,.07)}
    .chips .chip{padding:4px 12px;border-radius:99px;border:1px solid var(--line);background:rgba(255,255,255,.06);font-size:.8rem}
    .bigicon{font-size:2.2rem}
    .chk.done i{background:var(--teal);border:1.5px solid var(--teal);display:grid;place-items:center;font-style:normal;font-size:.8rem;color:#fff}
    .chk.done i::after{content:"✓"}
    .chk.done span{color:var(--muted)}
    .btn[disabled]{opacity:.6;cursor:default;transform:none}
    .rs{display:flex;gap:12px;align-items:center;padding:12px 14px;border:1px solid var(--line);border-radius:14px;background:rgba(0,0,0,.18);margin-bottom:8px;transition:transform .2s,border-color .2s}
    .rs:hover{transform:translateX(4px);border-color:var(--teal)}
    .rs b{display:block}
    .rs small{color:var(--muted)}
    .rs .pv{flex:none;font-size:.7rem;font-weight:700;text-transform:uppercase;padding:4px 9px;border-radius:99px;background:rgba(0,168,150,.25);color:#7fe9dc}
    .ph-title{font:800 .95rem var(--display);margin:14px 0 4px;color:var(--sky)}
    .ci b{display:block;font-size:.95rem}
  `;
  document.head.appendChild(st);

  let plan = null, code = '', modIdx = 0, saved = [];
  let done = new Set(), pending = null, resToken = 0;
  const lessonCache = {};

  const flat = () => plan ? plan.phases.flatMap(p => p.modules) : [];
  const key = m => m.skill + '|' + m.title;
  const isDone = m => done.has(key(m));

  const loadDone = () => {
    try { done = new Set(JSON.parse(localStorage.getItem('kz-done:' + code) || '[]')); }
    catch (e) { done = new Set(); }
  };
  const saveDone = () => {
    try { localStorage.setItem('kz-done:' + code, JSON.stringify([...done])); } catch (e) {}
  };

  async function api(url, opts) {
    const r = await fetch(url, opts);
    let j = {};
    try { j = await r.json(); } catch (e) {}
    if (!r.ok) {
      const msg = typeof j.error === 'string' ? j.error
                : typeof j.detail === 'string' ? j.detail
                : r.status === 422 ? 'Please check your preferences.' : `Request failed (${r.status})`;
      throw new Error(msg);
    }
    return j;
  }

  const notes = () => [q('#v0 .note'), q('#v2 .note')];
  function status(text, cls = '') {
    notes().forEach(n => { n.textContent = text; n.className = 'note ' + cls; });
  }

  function readPrefs() {
    return {
      code,
      level: q('#pLevel').value,
      hours: +q('#pHours').value,
      style: q('#pStyle').value,
      free_only: q('#pFree').checked,
      target_date: q('#pDate').value || null,
    };
  }
  function writePrefs(p) {
    if (!p) return;
    q('#pLevel').value = p.level;
    q('#pHours').value = p.hours;
    q('#hrsOut').textContent = p.hours + ' hours/week';
    if ([...q('#pStyle').options].some(o => o.value === p.style || o.text === p.style)) q('#pStyle').value = p.style;
    q('#pFree').checked = !!p.free_only;
    q('#pDate').value = p.target_date || '';
  }

  function updateProgress() {
    const mods = flat();
    if (!plan || !mods.length) return;
    const n = mods.filter(isDone).length;
    const pct = Math.round(n / mods.length * 100);

    [q('#v0 .g4').children[0], q('#v1 .g4').children[0]].forEach(s => {
      const ring = s.querySelector('.ring');
      ring.textContent = pct + '%';
      ring.style.borderColor = pct ? 'var(--teal)' : '';
      ring.style.color = pct ? '#fff' : '';
      s.querySelector('.lines').innerHTML =
        `<b>Overall progress</b><span class="m">${n} of ${mods.length} modules</span>`;
    });

    const small = q('#vc h2 small');
    if (small) small.textContent = `${n} of ${mods.length} completed`;

    // skills gained
    const skills = [...new Set(mods.filter(isDone).map(m => m.skill))];
    q('#gained').innerHTML = skills.map(s => `<span class="chip">${esc(s)}</span>`).join('');
    const gnote = q('#v1 .g2 .note');
    if (gnote) gnote.style.display = skills.length ? 'none' : '';

    // upcoming = first module not done
    const up = q('#v0 > .panel');
    const next = mods.find(m => !isDone(m));
    up.querySelector('.img').innerHTML = '<span class="bigicon">📘</span>';
    up.querySelector('.lines').innerHTML = next ? `
      <b style="font-size:1.1rem">${esc(next.title)}</b>
      <span class="m">${esc(next.skill)} · ${next.hours} h · ${esc(next.level)}</span>
      <span class="m">${esc(next.objectives[0])}</span>`
      : `<b style="font-size:1.1rem">🎉 All modules complete!</b>`;
      upNext = next ? mods.indexOf(next) : 0;
  }

  function render() {
    const p = plan, mods = flat();
    loadDone();

    const stats = q('#v0 .g4').children;
    const card = (el, ic, big, small) => {
      el.innerHTML = `<i class="kzic">${ic}</i><div class="lines"><b>${big}</b><span class="m">${small}</span></div>`;
    };
    card(stats[1], '⏱', `${p.total_weeks} weeks`, `${p.total_hours} hours at ${p.hours_per_week} h/week`);
    card(stats[2], '📚', `${mods.length} modules`, `${p.phases.length} phases`);
    card(stats[3], '🎯', `${p.gaps.length} skill gaps`, `to reach ${esc(p.career)}`);

    const ph = q('#phases');
    ph.style.gridTemplateColumns = `repeat(${p.phases.length},1fr)`;
    ph.innerHTML = p.phases.map(x => `
      <div style="text-align:center"><span class="phase-n">${esc(x.name)}</span>
      <span class="m">Week ${x.start_week}${x.weeks > 1 ? '–' + (x.start_week + x.weeks - 1) : ''}</span></div>`).join('');
    const nDots = Math.min(p.total_weeks, 24);
    q('#dots').innerHTML = Array.from({length: nDots}, (_, i) => `<i title="Week ${i + 1}"></i>`).join('');

    q('#gaps').innerHTML = p.gaps.slice(0, 6).map(g => `
      <div><div class="gaprow"><span>${esc(g.topic)}</span><small>${esc(g.skill)}</small></div>
      <span class="track"><div class="fill" style="width:${g.priority}%"></div></span></div>`).join('');

    q('#prev').innerHTML = p.phases.map((x, i) => `
      <div class="pi"><b>${i + 1}</b>
        <div class="lines"><span style="font-weight:700">${esc(x.name)}</span>
        <span class="m">${x.modules.length} module${x.modules.length > 1 ? 's' : ''} · ${x.hours} h</span></div>
        <small style="color:var(--sky);font-weight:700;white-space:nowrap">${x.weeks} wk${x.weeks > 1 ? 's' : ''}</small>
      </div>`).join('');

    let msg = (p.summary || '') + (p.ai_generated ? '  (AI-generated, please double-check.)' : '');
    let cls = 'ok';
    if (p.note) msg = p.note + ' ' + msg;
    if (p.fits_deadline === false) {
      msg = `At ${p.hours_per_week} h/week this takes ${p.total_weeks} weeks, but your target date allows ${p.weeks_available}. Raise weekly hours or move the date.`;
      cls = 'err';
    }
    status(msg.trim(), cls);

    const firstOpen = mods.findIndex(m => !isDone(m));
    modIdx = firstOpen < 0 ? 0 : firstOpen;
    updateProgress();
    renderCourse();
  }

  function renderCourse() {
    const mods = flat();
    if (!plan || !mods.length) return;
    const m = mods[modIdx];
    const d = isDone(m);

    q('#vc > .panel.flex .img').innerHTML = '<span class="bigicon">📘</span>';
    q('#vc > .panel.flex .lines').innerHTML = `
      <b style="font-size:1.25rem">${esc(m.title)}</b>
      <span class="m">${esc(m.topic)}</span>
      <span class="chips">
        <span class="chip">${esc(m.level)}</span>
        <span class="chip">${m.hours} hours</span>
        <span class="chip">${m.kind === 'soft' ? 'Reflection exercise' : 'Quiz'}</span>
      </span>`;

    // Mark complete button + bar
    const mark = q('#vc > .panel.flex .btn');
    mark.disabled = d;
    mark.textContent = d ? '✓ Completed' : 'Take quiz to complete';
    const barFill = q('#vc .bar > div');
    barFill.style.cssText = `height:100%;border-radius:4px;background:linear-gradient(90deg,var(--teal),var(--green));width:${d ? 100 : 0}%`;
    q('#vc > .panel.flex small').textContent = `${d ? 100 : 0}% complete`;

    // overview
    q('#outc').innerHTML = m.objectives.map(o =>
      `<div class="chk ${d ? 'done' : ''}"><i></i><span>${esc(o)}</span></div>`).join('');
    const ci = qa('#vc .g4 .lines');
    const info = [['Level', m.level], ['Duration', m.hours + ' hours'], ['Skill', m.skill],
                  ['Assessment', m.kind === 'soft' ? 'Reflection + quiz' : 'Quiz (60% to pass)']];
    ci.forEach((el, i) => {
      if (info[i]) el.innerHTML = `<div class="ci"><span class="m">${info[i][0]}</span><b>${esc(info[i][1])}</b></div>`;
    });

    // checklist
    q('#mods').innerHTML = mods.map((x, i) => `
      <div class="chk modrow ${i === modIdx ? 'cur' : ''} ${isDone(x) ? 'done' : ''}" data-i="${i}">
        <i></i><span style="flex:1">${esc(x.title)}</span><small class="m">${x.hours}h</small>
      </div>`).join('');
    q('#mods').querySelectorAll('.modrow').forEach(r =>
      r.onclick = () => { modIdx = +r.dataset.i; renderCourse(); });

    renderModulesPane();
    if (qa('#itabs button')[2].classList.contains('on')) renderResources();
    else resetResources();

    const small = q('#vc h2 small');
    if (small) small.textContent = `${mods.filter(isDone).length} of ${mods.length} completed`;
  }

  function renderModulesPane() {
    const pane = qa('#vc .ipane')[1];
    let idx = -1;
    pane.innerHTML = '<div class="panel">' + plan.phases.map(ph => `
      <div class="ph-title">${esc(ph.name)} · Week ${ph.start_week}</div>
      ${ph.modules.map(x => { idx++; return `
        <div class="chk modrow ${idx === modIdx ? 'cur' : ''} ${isDone(x) ? 'done' : ''}" data-i="${idx}">
          <i></i>
          <span style="flex:1"><b>${esc(x.title)}</b><br><span class="m">${esc(x.skill)} · ${esc(x.level)}</span></span>
          <small class="m">${x.hours}h</small>
        </div>`; }).join('')}`).join('') + '</div>';
    pane.querySelectorAll('.modrow').forEach(r =>
      r.onclick = () => { modIdx = +r.dataset.i; renderCourse(); });
  }

  const icon = { video: '▶', article: '📄', course: '🎓', docs: '📚' };
  const resPane = () => qa('#vc .ipane')[2];

  function resetResources() {
    resPane().innerHTML = '<div class="panel"><div class="dash">Loading resources…</div></div>';
  }

  async function renderResources() {
    const m = flat()[modIdx];
    if (!m) return;
    const pane = resPane();
    const token = ++resToken;
    const ck = code + '|' + m.skill;

    pane.innerHTML = '<div class="panel"><div class="dash">Your AI tutor is finding the best resources…</div></div>';
    let lesson = lessonCache[ck];
    try {
      if (!lesson) lesson = lessonCache[ck] = await api(
        `/api/reskilling/lesson?code=${encodeURIComponent(code)}&skill=${encodeURIComponent(m.skill)}`);
    } catch (e) {
      if (token === resToken)
        pane.innerHTML = `<div class="panel"><div class="dash">${esc(e.message)}</div></div>`;
      return;
    }
    if (token !== resToken) return; // user switched module meanwhile

    pane.innerHTML = `
      <div class="panel">
        <h2>${esc(lesson.skill)}</h2>
        <p>${esc(lesson.summary)}</p>
        <h2 style="margin-top:18px">How to learn it</h2>
        <ol style="padding-left:20px;display:grid;gap:6px">${lesson.steps.map(s => `<li>${esc(s)}</li>`).join('')}</ol>
        ${lesson.practice ? `<h2 style="margin-top:18px">Practice task</h2><p>${esc(lesson.practice)}</p>` : ''}
        <h2 style="margin-top:18px">Tutorials &amp; videos</h2>
        ${lesson.resources.map(r => `
          <a class="rs" href="${esc(r.url)}" target="_blank" rel="noopener noreferrer">
            <span>${icon[r.type] || '🔗'}</span>
            <span style="flex:1"><b>${esc(r.title)}</b><small>${esc(r.why)}</small></span>
            <span class="pv">${esc(r.provider)}</span>
          </a>`).join('')}
        <p style="margin-top:16px"><button class="btn" type="button" id="resQuiz">
          ${isDone(m) ? 'Retake quiz' : "I've finished — take the quiz"}</button></p>
      </div>`;
    q('#resQuiz').onclick = () => startQuiz(m);
  }

  function startQuiz(m) {
    pending = key(m);
    window.openLesson(code, m.skill);   // lessons.js modal: lesson tab -> quiz tab
  }

  document.addEventListener('lesson-done', e => {
    const mods = flat();
    const m = mods.find(x => key(x) === pending);
    if (!m || !e.detail.pass) return;   // failed quiz = NOT complete
    done.add(key(m));
    saveDone();
    pending = null;
    const next = mods.findIndex(x => !isDone(x));
    updateProgress();
    renderCourse();
    status(`✅ "${m.title}" completed!` + (next >= 0 ? ' Next module is ready.' : ' You finished your whole plan!'), 'ok');
  });

  async function generate(force) {
    if (!code) return;
    const btn = q('#regen');
    btn.disabled = true;
    status('Building your plan…');
    try {
      const j = await api('/api/reskilling/plan', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({...readPrefs(), force}),
      });
      plan = j.plan;
      render();
    } catch (e) {
      status(e.message, 'err');
    } finally {
      btn.disabled = false;
    }
  }

  async function selectCareer(c) {
    code = c;
    if (!c) { plan = null; status('Choose a target career to see your plan.'); return; }
    status('Loading your plan…');
    try {
      const j = await api('/api/reskilling/plan?code=' + encodeURIComponent(c));
      if (j.plan) { writePrefs(j.prefs); plan = j.plan; render(); }
      else await generate(false);
    } catch (e) {
      status(e.message, 'err');
    }
  }

  qa('.career').forEach(el => el.addEventListener('change', () => selectCareer(el.value)));
  q('#regen').addEventListener('click', () => generate(true));
    q('[data-open-course]').addEventListener('click', () => {
    modIdx = upNext;
    renderCourse();      // the page's own handler already calls show('vc')
  });
  q('#vc > .panel.flex .btn').addEventListener('click', () => {
    const m = flat()[modIdx];
    if (m && !isDone(m)) startQuiz(m);
  });
  // load resources only when the Resources tab is opened
  qa('#itabs button')[2].addEventListener('click', () => { if (plan) renderResources(); });

  (async function init() {
    try {
      const j = await api('/api/plan');
      saved = (j.plan || []).slice().sort((a, b) => (b.saved_at || '').localeCompare(a.saved_at || ''));
    } catch (e) { saved = []; }

    if (!saved.length) {
      status('No saved careers yet. Open Career Matches and save a career to your plan first.');
      return;
    }
    fillCareers(saved.map(p => ({value: p.code, label: p.title})));
    const latest = saved[0].code;
    qa('.career').forEach(el => el.value = latest);
    selectCareer(latest);
  })();
})();



let plan = null, code = '', modIdx = 0, saved = [];
let done = new Set(), pending = null, resToken = 0;
let upNext = 0;   // index of the "upcoming" module
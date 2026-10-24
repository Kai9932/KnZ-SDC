/* static/js/reskilling.js
   Fills the existing skeleton in reskilling.html with a generated plan.
   Load it AFTER the page's inline <script>. */
(() => {
  const q = s => document.querySelector(s);
  const esc = s => String(s ?? '').replace(/[&<>"']/g,
    c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

  // small extra styles
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
  `;
  document.head.appendChild(st);

  let plan = null, code = '', modIdx = 0, saved = [];
  const flat = () => plan ? plan.phases.flatMap(p => p.modules) : [];

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

  /* ---------- preferences ---------- */
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

  /* ---------- rendering ---------- */
  function clear() {
    plan = null;
  }

  function render() {
    const p = plan, mods = flat();

    // stat cards
    const stats = q('#v0 .g4').children;
    stats[0].querySelector('.ring').textContent = '0%';
    stats[0].querySelector('.lines').innerHTML =
      `<b>Overall progress</b><span class="m">0 of ${mods.length} modules</span>`;
    const card = (el, ic, big, small) => {
      el.innerHTML = `<i class="kzic">${ic}</i><div class="lines"><b>${big}</b><span class="m">${small}</span></div>`;
    };
    card(stats[1], '⏱', `${p.total_weeks} weeks`, `${p.total_hours} hours at ${p.hours_per_week} h/week`);
    card(stats[2], '📚', `${mods.length} modules`, `${p.phases.length} phases`);
    card(stats[3], '🎯', `${p.gaps.length} skill gaps`, `to reach ${esc(p.career)}`);

    // timeline
    const ph = q('#phases');
    ph.style.gridTemplateColumns = `repeat(${p.phases.length},1fr)`;
    ph.innerHTML = p.phases.map(x => `
      <div style="text-align:center"><span class="phase-n">${esc(x.name)}</span>
      <span class="m">Week ${x.start_week}${x.weeks > 1 ? '–' + (x.start_week + x.weeks - 1) : ''}</span></div>`).join('');
    const nDots = Math.min(p.total_weeks, 24);
    q('#dots').innerHTML = Array.from({length: nDots}, (_, i) => `<i title="Week ${i + 1}"></i>`).join('');

    // skill gaps
    q('#gaps').innerHTML = p.gaps.slice(0, 6).map(g => `
      <div><div class="gaprow"><span>${esc(g.topic)}</span><small>${esc(g.skill)}</small></div>
      <span class="track"><div class="fill" style="width:${g.priority}%"></div></span></div>`).join('');

    // upcoming module
    const m = mods[0];
    const up = q('#v0 > .panel');
    up.querySelector('.img').innerHTML = '<span class="bigicon">📘</span>';
    up.querySelector('.lines').innerHTML = `
      <b style="font-size:1.1rem">${esc(m.title)}</b>
      <span class="m">${esc(m.skill)} · ${m.hours} h · ${esc(m.level)}</span>
      <span class="m">${esc(m.objectives[0])}</span>`;

    // customise preview
    q('#prev').innerHTML = p.phases.map((x, i) => `
      <div class="pi"><b>${i + 1}</b>
        <div class="lines"><span style="font-weight:700">${esc(x.name)}</span>
        <span class="m">${x.modules.length} module${x.modules.length > 1 ? 's' : ''} · ${x.hours} h</span></div>
        <small style="color:var(--sky);font-weight:700;white-space:nowrap">${x.weeks} wk${x.weeks > 1 ? 's' : ''}</small>
      </div>`).join('');

    // status line
    let msg = (p.summary || '') + (p.ai_generated ? '  (AI-generated, please double-check.)' : '');
    let cls = 'ok';
    if (p.note) msg = p.note + ' ' + msg;
    if (p.fits_deadline === false) {
      msg = `At ${p.hours_per_week} h/week this takes ${p.total_weeks} weeks, but your target date allows ${p.weeks_available}. Raise weekly hours or move the date.`;
      cls = 'err';
    }
    status(msg.trim(), cls);

    modIdx = 0;
    renderCourse();
  }

  function renderCourse() {
    const mods = flat();
    if (!plan || !mods.length) return;
    const m = mods[modIdx];

    q('#vc > .panel.flex .img').innerHTML = '<span class="bigicon">📘</span>';
    q('#vc > .panel.flex .lines').innerHTML = `
      <b style="font-size:1.25rem">${esc(m.title)}</b>
      <span class="m">${esc(m.topic)}</span>
      <span class="chips">
        <span class="chip">${esc(m.level)}</span>
        <span class="chip">${m.hours} hours</span>
        <span class="chip">${m.kind === 'soft' ? 'Reflection exercise' : 'Quiz'}</span>
      </span>`;

    q('#outc').innerHTML = m.objectives.map(o =>
      `<div class="chk"><i></i><span>${esc(o)}</span></div>`).join('');

    q('#mods').innerHTML = mods.map((x, i) => `
      <div class="chk modrow ${i === modIdx ? 'cur' : ''}" data-i="${i}">
        <i></i><span style="flex:1">${esc(x.title)}</span><small class="m">${x.hours}h</small>
      </div>`).join('');
    q('#mods').querySelectorAll('.modrow').forEach(r =>
      r.onclick = () => { modIdx = +r.dataset.i; renderCourse(); });

    const small = q('#vc h2 small');
    if (small) small.textContent = `0 of ${mods.length} completed`;
  }

  /* ---------- loading / generating ---------- */
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
    if (!c) { clear(); status('Choose a target career to see your plan.'); return; }
    status('Loading your plan…');
    try {
      const j = await api('/api/reskilling/plan?code=' + encodeURIComponent(c));
      if (j.plan) { writePrefs(j.prefs); plan = j.plan; render(); }
      else await generate(false);
    } catch (e) {
      status(e.message, 'err');
    }
  }

  /* ---------- init ---------- */
  document.querySelectorAll('.career').forEach(el =>
    el.addEventListener('change', () => selectCareer(el.value)));
  q('#regen').addEventListener('click', () => generate(true));

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
    document.querySelectorAll('.career').forEach(el => el.value = latest);
    selectCareer(latest);
  })();
})();
/* KnZ AI Career Counsellor widget. No dependencies. */
(function () {
  "use strict";
  var root = document.getElementById("knz-chat-root");
  if (!root) return;

  var API = "/api/chatbot";
  var MAX_LEN = 1000;
  var SUGGESTIONS = [
    "I don't know which career suits me. Where do I start?",
    "What skills should I learn to become a Data Analyst?",
    "I'm feeling anxious about my career future. Can we talk?",
  ];

  var state = { open: false, view: "chat", convId: null, busy: false, loaded: false, signedIn: true };

  /* ---------- helpers ---------- */
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function api(path, opts) {
    opts = opts || {};
    opts.credentials = "same-origin";
    if (opts.body) opts.headers = { "Content-Type": "application/json" };
    return fetch(API + path, opts).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (data) {
        return { ok: r.ok, status: r.status, data: data };
      });
    });
  }

  // Safe renderer: everything is escaped first, then **bold**, lists and paragraphs are added.
  function esc(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function inline(s) {
    return esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  }
  function render(text) {
    var out = [], list = null, para = [];
    function flushPara() { if (para.length) { out.push("<p>" + para.join("<br>") + "</p>"); para = []; } }
    function flushList() { if (list) { out.push("</" + list + ">"); list = null; } }
    text.split("\n").forEach(function (line) {
      var ul = line.match(/^\s*[-*•]\s+(.*)/), ol = line.match(/^\s*\d+[.)]\s+(.*)/);
      if (ul || ol) {
        flushPara();
        var want = ul ? "ul" : "ol";
        if (list !== want) { flushList(); out.push("<" + want + ">"); list = want; }
        out.push("<li>" + inline((ul || ol)[1]) + "</li>");
      } else if (!line.trim()) {
        flushPara(); flushList();
      } else {
        flushList();
        para.push(inline(line.replace(/^#+\s*/, "")));
      }
    });
    flushPara(); flushList();
    return out.join("");
  }

  function when(iso) {
    var d = new Date(iso), now = new Date();
    return d.toDateString() === now.toDateString()
      ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
      : d.toLocaleDateString([], { day: "numeric", month: "short" });
  }

  /* ---------- build DOM ---------- */
  var fab = el("button", "kc-fab");
  fab.type = "button";
  fab.setAttribute("aria-label", "Open AI Career Counsellor");
  fab.setAttribute("aria-expanded", "false");
  fab.innerHTML =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/></svg>';

  var panel = el("section", "kc-panel");
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-label", "AI Career Counsellor");

  var head = el("div", "kc-head");
  var title = el("h2", null, "AI Career Counsellor");
  var btnHist = el("button", "kc-icon-btn", "History");
  var btnNew = el("button", "kc-icon-btn", "New chat");
  var btnClose = el("button", "kc-icon-btn", "✕");
  btnHist.type = btnNew.type = btnClose.type = "button";
  btnClose.setAttribute("aria-label", "Close chat");
  head.append(title, btnHist, btnNew, btnClose);

  var body = el("div", "kc-body");
  body.setAttribute("aria-live", "polite");
  var hist = el("div", "kc-hist");
  hist.hidden = true;
  var histFoot = el("div", "kc-hist-foot");
  histFoot.hidden = true;

  var form = el("form", "kc-form");
  var input = el("textarea", "kc-input");
  input.rows = 1;
  input.maxLength = MAX_LEN;
  input.placeholder = "Ask about careers, skills or study paths";
  input.setAttribute("aria-label", "Your message");
  var send = el("button", "kc-send", "Send");
  send.type = "submit";
  form.append(input, send);
  var note = el("div", "kc-note", "AI guidance, not a guarantee. Check important details with official sources.");

  panel.append(head, body, hist, histFoot, form, note);
  root.append(fab, panel);

  /* ---------- view switching ---------- */
  function show(view) {
    state.view = view;
    var chat = view === "chat";
    body.hidden = !chat; form.hidden = !chat || !state.signedIn; note.hidden = !chat;
    hist.hidden = chat; histFoot.hidden = chat;
    btnHist.textContent = chat ? "History" : "Back";
    title.textContent = chat ? "AI Career Counsellor" : "Your chats";
  }

  function scrollDown() { body.scrollTop = body.scrollHeight; }

  function addMsg(role, text) {
    var m = el("div", "kc-msg " + (role === "user" ? "kc-user" : "kc-bot"));
    if (role === "user") m.textContent = text; else m.innerHTML = render(text);
    body.appendChild(m);
    scrollDown();
    return m;
  }
  function addError(text) {
    var m = el("div", "kc-msg kc-error", text);
    m.setAttribute("role", "alert");
    body.appendChild(m);
    scrollDown();
  }

  function welcome() {
    body.textContent = "";
    var w = el("div", "kc-welcome");
    w.append(el("strong", null, "Hi, I'm your career counsellor"),
             el("span", null, "Ask me about careers, skills or what to study next. I keep your past chats so you can pick up where you left off."));
    var chips = el("div", "kc-chips");
    SUGGESTIONS.forEach(function (s) {
      var c = el("button", "kc-chip", s);
      c.type = "button";
      c.addEventListener("click", function () { submit(s); });
      chips.appendChild(c);
    });
    w.appendChild(chips);
    body.appendChild(w);
  }

  function signInPrompt() {
    state.signedIn = false;
    body.textContent = "";
    var d = el("div", "kc-signin");
    d.append(el("strong", null, "Sign in to chat"),
             el("div", null, "Your conversations are saved to your account."));
    var a = el("a", null, "Sign in with Google");
    a.href = "/login";
    d.appendChild(a);
    body.appendChild(d);
    form.hidden = true;
  }

  /* ---------- data ---------- */
  function openConversation(id) {
    return api("/conversations/" + encodeURIComponent(id)).then(function (r) {
      if (!r.ok) { state.convId = null; welcome(); return; }
      state.convId = r.data.id;
      body.textContent = "";
      r.data.messages.forEach(function (m) { addMsg(m.role, m.content); });
      if (!r.data.messages.length) welcome();
      scrollDown();
    });
  }

  function loadLatest() {
    return api("/conversations").then(function (r) {
      if (r.status === 401) { signInPrompt(); return; }
      state.loaded = true;
      var list = (r.data.conversations || []);
      if (list.length) return openConversation(list[0].id);
      welcome();
    });
  }

  function newChat() {
    if (state.busy) return;
    state.convId = null;
    show("chat");
    welcome();
    input.focus();
  }

  function showHistory() {
    show("history");
    hist.textContent = "";
    histFoot.textContent = "";
    api("/conversations").then(function (r) {
      if (r.status === 401) { show("chat"); signInPrompt(); return; }
      var list = r.data.conversations || [];
      if (!list.length) hist.appendChild(el("div", "kc-empty", "No saved chats yet."));
      list.forEach(function (c) {
        var row = el("div", "kc-hist-row");
        var open = el("button", "kc-hist-open");
        open.type = "button";
        open.append(el("b", null, c.title), el("small", null, when(c.updated_at)));
        open.addEventListener("click", function () {
          if (state.busy) return;
          show("chat");
          openConversation(c.id);
        });
        var del = el("button", "kc-del", "Delete");
        del.type = "button";
        del.setAttribute("aria-label", "Delete chat: " + c.title);
        del.addEventListener("click", function () {
          if (!confirm("Delete this chat? This can't be undone.")) return;
          api("/conversations/" + encodeURIComponent(c.id), { method: "DELETE" }).then(function () {
            if (state.convId === c.id) { state.convId = null; welcome(); }
            showHistory();
          });
        });
        row.append(open, del);
        hist.appendChild(row);
      });

      var nb = el("button", "kc-primary", "New chat");
      nb.type = "button";
      nb.addEventListener("click", newChat);
      histFoot.appendChild(nb);
      if (list.length) {
        var clr = el("button", "kc-danger", "Clear all");
        clr.type = "button";
        clr.addEventListener("click", function () {
          if (!confirm("Delete all your chats? This can't be undone.")) return;
          api("/conversations", { method: "DELETE" }).then(function () {
            state.convId = null; welcome(); showHistory();
          });
        });
        histFoot.appendChild(clr);
      }
    });
  }

  function submit(text) {
    text = (text || "").trim();
    if (!text || state.busy || !state.signedIn) return;
    if (!body.querySelector(".kc-msg")) body.textContent = "";   // remove welcome screen
    addMsg("user", text);
    input.value = "";
    autosize();

    state.busy = true;
    send.disabled = true;
    var typing = el("div", "kc-msg kc-bot");
    typing.innerHTML = '<span class="kc-typing" aria-label="Thinking"><i></i><i></i><i></i></span>';
    body.appendChild(typing);
    scrollDown();

    api("/message", { method: "POST", body: JSON.stringify({ conversation_id: state.convId, message: text }) })
      .then(function (r) {
        typing.remove();
        if (r.data && r.data.conversation_id) state.convId = r.data.conversation_id;
        if (r.status === 401) { signInPrompt(); return; }
        if (!r.ok) { addError(r.data.error || "Something went wrong. Please try again."); return; }
        addMsg("assistant", r.data.reply);
      })
      .catch(function () {
        typing.remove();
        addError("Can't reach the server. Check your connection and try again.");
      })
      .then(function () {
        state.busy = false;
        send.disabled = false;
        input.focus();
      });
  }

  /* ---------- events ---------- */
  function autosize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 110) + "px";
  }

  function setOpen(open) {
    state.open = open;
    panel.classList.toggle("open", open);
    fab.setAttribute("aria-expanded", String(open));
    fab.setAttribute("aria-label", open ? "Close AI Career Counsellor" : "Open AI Career Counsellor");
    if (open) {
      if (!state.loaded && state.signedIn) { body.textContent = ""; loadLatest(); }
      if (state.signedIn) input.focus();
    }
  }

  fab.addEventListener("click", function () { setOpen(!state.open); });
  btnClose.addEventListener("click", function () { setOpen(false); fab.focus(); });
  btnNew.addEventListener("click", newChat);
  btnHist.addEventListener("click", function () {
    if (state.view === "chat") showHistory(); else show("chat");
  });
  form.addEventListener("submit", function (e) { e.preventDefault(); submit(input.value); });
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); submit(input.value); }
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && state.open) { setOpen(false); fab.focus(); }
  });

  show("chat");
})();
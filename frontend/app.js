"use strict";

/* Юридический ассистент — клиент веб-чата.
 *
 * Контракт API (docs/backend-contract.md):
 *   GET  /health/ready          -> {status} | 503
 *   POST /api/chat              -> {kind, answer, sources[]}
 *       {question, history:[{role:"user"|"assistant", content}]}
 *   GET  /api/sources/{chunk_id}-> source + text фрагмента
 * Метки [1]..[n] в тексте ответа соответствуют порядку массива sources.
 */

const API_BASE = new URLSearchParams(location.search).get("api") || "";

const LIMITS = {
  questionChars: 4000,
  historyMessages: 8,
  historyChars: 12000,
};

const KIND = {
  answer: { label: "Ответ", cls: "kind-answer" },
  partial: { label: "Частичный ответ", cls: "kind-partial" },
  clarification: { label: "Уточнение", cls: "kind-clarify" },
  insufficient: { label: "Недостаточно данных", cls: "kind-insufficient" },
  refusal: { label: "Отказ", cls: "kind-refusal" },
};

const $ = (id) => document.getElementById(id);
const ui = {
  messages: $("messages"),
  empty: $("emptyState"),
  examples: $("examples"),
  form: $("composer"),
  input: $("questionInput"),
  send: $("sendButton"),
  counter: $("counter"),
  panel: $("sourcePanel"),
  panelBody: $("sourcePanelBody"),
  panelClose: $("closeSourcePanel"),
  statusDot: $("statusDot"),
  statusText: $("statusText"),
};

const state = { history: [], busy: false, lastQuestion: "" };

/* ---------------------------------------------------------------- utils */

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatReference(source) {
  const parts = [];
  if (source.article) parts.push("ст. " + source.article);
  if (source.part) parts.push("ч. " + source.part);
  if (source.clause) parts.push("п. " + source.clause);
  if (source.subclause) parts.push("пп. " + source.subclause);
  if (source.section) parts.push("разд. " + source.section);
  if (source.appendix) parts.push("прил. " + source.appendix);
  return parts.join(", ");
}

async function api(path, options) {
  let response;
  try {
    response = await fetch(API_BASE + path, options);
  } catch (error) {
    throw { code: "network", message: "Нет связи с сервисом. Проверьте, что он запущен." };
  }
  let body = null;
  try {
    body = await response.json();
  } catch (error) {
    throw { code: "invalid_response", message: "Сервис вернул некорректный ответ." };
  }
  if (!response.ok) {
    const detail = (body && body.detail) || {};
    throw {
      code: detail.code || "internal",
      message: detail.message || "Не удалось обработать запрос.",
    };
  }
  return body;
}

/* ------------------------------------------------------------- статус */

function setStatus(kind, text) {
  ui.statusDot.className = "dot dot-" + kind;
  ui.statusText.textContent = text;
}

async function checkHealth() {
  try {
    await api("/health/ready");
    setStatus("ok", "сервис готов");
  } catch (error) {
    setStatus("down", "сервис недоступен");
  }
}

/* ------------------------------------------------------------- рендер */

function scrollToBottom() {
  ui.messages.scrollTop = ui.messages.scrollHeight;
}

function hideEmpty() {
  if (ui.empty) ui.empty.hidden = true;
}

function addUserMessage(text) {
  hideEmpty();
  const node = document.createElement("div");
  node.className = "msg msg-user";
  node.innerHTML = '<div class="bubble">' + escapeHtml(text) + "</div>";
  ui.messages.appendChild(node);
  scrollToBottom();
  return node;
}

function addTyping() {
  const node = document.createElement("div");
  node.className = "msg msg-assistant";
  node.innerHTML =
    '<div class="bubble bubble-typing"><span></span><span></span><span></span></div>';
  ui.messages.appendChild(node);
  scrollToBottom();
  return node;
}

function addErrorMessage(message, question) {
  hideEmpty();
  const node = document.createElement("div");
  node.className = "msg msg-error";
  node.innerHTML =
    '<div class="bubble">' +
    '<strong>Не получилось обработать запрос</strong>' +
    "<p>" + escapeHtml(message) + "</p>" +
    '<button type="button" class="retry">Повторить</button>' +
    "</div>";
  node.querySelector(".retry").addEventListener("click", () => {
    node.remove();
    ask(question, true);
  });
  ui.messages.appendChild(node);
  scrollToBottom();
}

function renderAnswerText(text, sources) {
  const escaped = escapeHtml(text);
  return escaped.replace(/\[(\d{1,2})\]/g, (match, number) => {
    const index = Number(number) - 1;
    if (!sources[index]) return match;
    return (
      '<button type="button" class="cite" data-source-index="' +
      index +
      '">[' + number + "]</button>"
    );
  });
}

function addAssistantMessage(result) {
  hideEmpty();
  const sources = Array.isArray(result.sources) ? result.sources : [];
  const kind = KIND[result.kind] || KIND.answer;

  const node = document.createElement("div");
  node.className = "msg msg-assistant";

  const badge = result.kind && result.kind !== "answer"
    ? '<div class="badge ' + kind.cls + '">' + kind.label + "</div>"
    : "";

  let sourceList = "";
  if (sources.length) {
    sourceList =
      '<div class="source-list">' +
      sources
        .map((source, index) => {
          const title = escapeHtml(source.doc_title || source.source_reference || "Документ");
          const ref = escapeHtml(formatReference(source));
          return (
            '<button type="button" class="source-chip" data-source-index="' + index + '">' +
            '<span class="chip-num">[' + (index + 1) + "]</span>" +
            "<span>" + title + (ref ? " · " + ref : "") + "</span>" +
            "</button>"
          );
        })
        .join("") +
      "</div>";
  }

  node.innerHTML =
    '<div class="bubble">' + badge +
    '<div class="answer">' + renderAnswerText(result.answer, sources) + "</div>" +
    sourceList + "</div>";

  node.querySelectorAll("[data-source-index]").forEach((button) => {
    button.addEventListener("click", () => {
      const source = sources[Number(button.dataset.sourceIndex)];
      if (source) openSourcePanel(source);
    });
  });

  ui.messages.appendChild(node);
  scrollToBottom();
}

/* ------------------------------------------------------- панель источника */

function metaRow(label, value) {
  if (!value) return "";
  return (
    '<div class="meta-row"><span class="meta-label">' + escapeHtml(label) +
    '</span><span class="meta-value">' + escapeHtml(value) + "</span></div>"
  );
}

async function openSourcePanel(source) {
  ui.panel.hidden = false;
  const title = source.doc_title || source.source_reference || "Документ";
  const ref = formatReference(source) || "—";

  ui.panelBody.innerHTML =
    '<h3>' + escapeHtml(title) + "</h3>" +
    '<div class="meta">' +
    metaRow("Положение", ref) +
    metaRow("Ссылка", source.source_reference) +
    metaRow("Номер документа", source.doc_number) +
    metaRow("Дата документа", source.doc_date) +
    metaRow("Редакция от", source.edition_date) +
    metaRow("Действует с", source.effective_date) +
    metaRow("Статус редакции", source.version_status) +
    metaRow("Временной статус", source.temporal_status) +
    "</div>" +
    '<div class="fragment"><div class="loader">Загрузка фрагмента…</div></div>';

  const fragmentBox = ui.panelBody.querySelector(".fragment");
  try {
    const fragment = await api(
      "/api/sources/" + encodeURIComponent(source.chunk_id)
    );
    fragmentBox.innerHTML =
      '<h4>Подтверждающий фрагмент</h4>' +
      '<blockquote>' + escapeHtml(fragment.text || "") + "</blockquote>";
  } catch (error) {
    fragmentBox.innerHTML =
      '<div class="fragment-error">' + escapeHtml(error.message) + "</div>";
  }
}

/* --------------------------------------------------------------- диалог */

function trimmedHistory() {
  const recent = state.history.slice(-LIMITS.historyMessages);
  while (
    recent.length > 1 &&
    recent.reduce((sum, turn) => sum + turn.content.length, 0) > LIMITS.historyChars
  ) {
    recent.shift();
  }
  return recent;
}

function setBusy(busy) {
  state.busy = busy;
  ui.send.disabled = busy;
  ui.send.textContent = busy ? "Думаю…" : "Отправить";
}

async function ask(question, isRetry) {
  if (state.busy) return;
  const text = question.trim();
  if (!text) return;

  state.lastQuestion = text;
  if (!isRetry) addUserMessage(text);
  setBusy(true);
  const typing = addTyping();

  try {
    const result = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: text, history: trimmedHistory() }),
    });
    typing.remove();
    addAssistantMessage(result);
    state.history.push({ role: "user", content: text });
    state.history.push({ role: "assistant", content: result.answer });
  } catch (error) {
    typing.remove();
    addErrorMessage(error.message || "Не удалось обработать запрос.", text);
  } finally {
    setBusy(false);
  }
}

/* ------------------------------------------------------- инициализация */

ui.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = ui.input.value;
  ui.input.value = "";
  resizeInput();
  updateCounter();
  ask(text, false);
});

ui.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    ui.form.requestSubmit();
  }
});

function resizeInput() {
  ui.input.style.height = "auto";
  ui.input.style.height = Math.min(ui.input.scrollHeight, 160) + "px";
}

function updateCounter() {
  ui.counter.textContent = ui.input.value.length + " / " + LIMITS.questionChars;
}

ui.input.addEventListener("input", () => {
  resizeInput();
  updateCounter();
});

ui.examples.addEventListener("click", (event) => {
  const button = event.target.closest(".example");
  if (!button) return;
  ui.input.value = button.textContent;
  ui.input.focus();
  resizeInput();
  updateCounter();
});

ui.panelClose.addEventListener("click", () => {
  ui.panel.hidden = true;
});

checkHealth();
updateCounter();

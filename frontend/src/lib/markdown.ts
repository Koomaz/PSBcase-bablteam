/**
 * Лёгкий рендер markdown для ответов ассистента — без сторонних зависимостей.
 *
 * Поддерживаемое подмножество: абзацы, переносы строк, **жирный**, *курсив*
 * и _курсив_, `код`, #### заголовки, списки - / * / 1. Весь входной HTML
 * экранируется до разметки, поэтому вставки безопасны: наружу уходят только
 * теги, созданные здесь.
 *
 * Метки цитат [1]…[n] превращаются в кнопки с data-cite; цитата приклеивается
 * к слову и пунктуации («слово[1].»), чтобы не рвать текст.
 */

const CITATION = /\[(\d{1,2})\]/g;
const CITE_MARK = "\u0001CITE$1\u0001";

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function inline(text: string): string {
  return text
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/(^|[\s(])_([^_\n]+)_/g, "$1<em>$2</em>");
}

const LIST_ITEM = /^(?:[-*•]\s+|\d+[.)]\s+)/;

function block(lines: string[], start: number): { html: string; next: number } {
  const line = lines[start];

  const heading = /^(#{1,6})\s+(.*)$/.exec(line);
  if (heading) {
    const level = Math.min(4, heading[1].length + 1);
    return { html: `<h${level}>${inline(heading[2])}</h${level}>`, next: start + 1 };
  }

  if (LIST_ITEM.test(line)) {
    const ordered = /^\d+[.)]\s+/.test(line);
    const items: string[] = [];
    let i = start;
    while (i < lines.length && LIST_ITEM.test(lines[i])) {
      items.push(`<li>${inline(lines[i].replace(LIST_ITEM, ""))}</li>`);
      i += 1;
    }
    const tag = ordered ? "ol" : "ul";
    return { html: `<${tag}>${items.join("")}</${tag}>`, next: i };
  }

  const paragraph: string[] = [];
  let i = start;
  while (
    i < lines.length &&
    lines[i].trim() !== "" &&
    !LIST_ITEM.test(lines[i]) &&
    !/^#{1,6}\s+/.test(lines[i])
  ) {
    paragraph.push(lines[i]);
    i += 1;
  }
  return { html: `<p>${inline(paragraph.join("<br>"))}</p>`, next: Math.max(i, start + 1) };
}

export function renderRichText(raw: string): string {
  const glued = raw
    .replace(/\s+\[(\d{1,2})\]/g, "[$1]")
    .replace(/\[(\d{1,2})\]\s+([.,;:!?])/g, "[$1]$2");

  const marked = escapeHtml(glued).replace(CITATION, CITE_MARK);
  const lines = marked.split(/\r?\n/);

  const parts: string[] = [];
  let i = 0;
  while (i < lines.length) {
    if (lines[i].trim() === "") {
      i += 1;
      continue;
    }
    const result = block(lines, i);
    parts.push(result.html);
    i = result.next;
  }

  return parts
    .join("")
    .replace(
      /\u0001CITE(\d{1,2})\u0001/g,
      '<button type="button" class="cite" data-cite="$1">[$1]</button>'
    );
}

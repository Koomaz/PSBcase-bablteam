/**
 * Утолщение палочки полосы прокрутки при наведении именно на неё.
 *
 * Chrome меняет по :hover цвета полосы, но не её геометрию, поэтому палочку
 * утолщаем классом: при курсоре у правого края прокручиваемой области (там
 * находится вертикальная полоса) контейнер получает класс `scroll-thick`.
 * Ширина дорожки при этом не меняется — контент не сдвигается.
 */

const SELECTOR = ".messages, .dialog-list, .source-panel-body, .doc-modal-body";
const EDGE_ZONE = 14; // px от правого края — запас на точность попадания курсором

function scrollableAncestor(target: EventTarget | null): HTMLElement | null {
  const element = target instanceof HTMLElement ? target.closest(SELECTOR) : null;
  return element instanceof HTMLElement ? element : null;
}

export function installScrollbarHover(): () => void {
  const clear = () => {
    document
      .querySelectorAll(".scroll-thick")
      .forEach((node) => node.classList.remove("scroll-thick"));
  };

  const onMouseMove = (event: MouseEvent) => {
    const container = scrollableAncestor(event.target);
    if (!container) {
      clear();
      return;
    }
    const rect = container.getBoundingClientRect();
    const overScrollbar =
      container.scrollHeight > container.clientHeight &&
      event.clientX >= rect.right - EDGE_ZONE;
    container.classList.toggle("scroll-thick", overScrollbar);
  };

  document.addEventListener("mousemove", onMouseMove, { passive: true });
  document.addEventListener("mouseleave", clear);
  return () => {
    document.removeEventListener("mousemove", onMouseMove);
    document.removeEventListener("mouseleave", clear);
  };
}

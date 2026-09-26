// Общий экземпляр Lenis: его создаёт App, а секциям иногда нужно ненадолго
// остановить прокрутку (например, пока играет анимация)
let lenis = null;
let heldY = null;

export function setLenis(instance) {
  lenis = instance;
}

// Пока прокрутка заперта, колесо и касания гасит сам Lenis. Клавиатура и перетаскивание
// полосы прокрутки — нативный scroll, его возвращаем на место здесь. Полосу прокрутки
// не прячем (см. index.css): иначе ширина страницы меняется и всё прыгает
const hold = () => {
  if (heldY !== null && Math.abs(window.scrollY - heldY) > 0.5) window.scrollTo(0, heldY);
};

// Запереть прокрутку на позиции y. Lenis по инерции мог проскочить её — тогда страница
// коротко доезжает обратно, а не прыгает
export function lockScrollAt(y, onLocked) {
  if (!lenis) return onLocked?.();
  const done = () => {
    lenis.stop();
    heldY = y;
    window.addEventListener("scroll", hold);
    onLocked?.();
  };
  if (Math.abs(lenis.scroll - y) < 1) done();
  else lenis.scrollTo(y, { duration: 0.2, force: true, lock: true, onComplete: done });
}

export function unlockScroll() {
  heldY = null;
  window.removeEventListener("scroll", hold);
  lenis?.start();
}

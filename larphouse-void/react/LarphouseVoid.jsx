import { useEffect, useRef, useState } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useGSAP } from "@gsap/react";

import { lockScrollAt, unlockScroll } from "@/lib/scroll";

gsap.registerPlugin(ScrollTrigger);
gsap.registerPlugin(useGSAP);

import "./LarphouseVoid.scss";

// Два пре-рендера, оба 1920×1080 с прозрачным фоном, 30 fps.
// Лежат в public, а не в assets: сотни импортов раздули бы бандл и граф модулей,
// а по URL их можно грузить по мере надобности.
//  - Void: проход сквозь арку, 112 кадров (3.7 с). Идёт по скроллу. Обрезан на кадре, где дверь
//    уже ушла за края: дальше в исходнике только пустота, и скроллить её незачем.
//  - Larphouse: надпись и меч в светлой печати, 5 с (играется ускоренно), угловые лозы отдельно — см. CORNERS. Идёт сама, когда прокрутка Void закончилась:
//    на это время скролл блокируется. При прокрутке назад она так же сама проигрывается задом наперёд
const FRAME_W = 1920;
const FRAME_H = 1080;
const VOID = { count: 112, src: (i) => `/void/frames/f_${String(i).padStart(3, "0")}.avif` };
const LARPHOUSE = { count: 150, fps: 30, src: (i) => `/larphouse/frames/f_${String(i).padStart(3, "0")}.avif` };
const LARPHOUSE_DURATION = (LARPHOUSE.count - 1) / LARPHOUSE.fps;   // 5 с — длина исходной анимации
// Во сколько раз быстрее исходника играть Larphouse (1.7 → ~2.9 с), пока страница заперта
const LARPHOUSE_SPEED = 1.7;

// Угловые лозы Larphouse не впечатаны в кадры: они рисуются прямо по углам экрана при любом
// разрешении. Для каждого угла — спрайт из CORNER.steps стадий роста (сетка CORNER.cols в ширину),
// ячейка cell×cell. Размер и отступ — как на странице 1920×1080 (высота лозы 372, отступ 38 при
// высоте 1080), но от меньшей стороны экрана, чтобы на телефоне лоза не занимала полэкрана.
// Тайминги роста — T_CORNERS из larphouse.py: [начало, длительность, степень замедления]
const CORNER = { w: 288, h: 372, margin: 38, page: 1080, steps: 40, cols: 8 };
const CORNERS = [
  { name: "bottom-right", time: [0.25, 4.0, 1.3], place: (W, H, w, h, m) => [W - m - w, H - m - h] },
  { name: "bottom-left", time: [0.6, 3.5, 1.7], place: (W, H, w, h, m) => [m, H - m - h] },
  { name: "top-right", time: [0.0, 3.7, 1.5], place: (W, H, w, h, m) => [W - m - w, m] },
  { name: "top-left", time: [0.45, 4.3, 1.9], place: (W, H, w, h, m) => [m, m] },
];
const cornerSrc = (name) => `/larphouse/corners/corner-${name}.avif`;
const cornerProgress = (t, [start, duration, power]) =>
  1 - (1 - Math.min(Math.max((t - start) / duration, 0), 1)) ** power;

// Сколько запросов держать одновременно: больше — забиваем канал и первые
// кадры приходят позже, меньше — весь набор тянется заметно дольше
const PARALLEL = 6;

// Кадры одного набора. Держать все декодированными нельзя: кадр 1920×1080 в памяти — 8 МБ,
// 262 кадра двух наборов — 2 ГБ, и браузер переставал создавать новые (оставались одни лозы).
// Поэтому файлы скачиваются заранее в сжатом виде (десятки МБ), а декодированными держится
// только окно вокруг текущего кадра: DECODED штук, с опережением AHEAD по ходу движения.
const DECODED = 24;
const AHEAD = 10;
const WARM_LIMIT = 3000;   // мс, сколько самое большее ждать кадры перед проигрыванием

function useFrames(set, onFirst) {
  const store = useRef(null);
  if (!store.current) {
    let resolve;
    const s = {
      blobs: new Array(set.count),
      cache: new Map(),                               // кадр → ImageBitmap, в порядке использования
      pending: new Set(),
      failed: new Set(),                              // не декодировались: больше не пробуем и не ждём
      version: 0,                                     // растёт с каждым декодированным кадром: пора перерисовать
      all: new Promise((r) => (resolve = r)),         // все файлы набора скачаны
      alive: true,
      focus: 0,
    };
    s.resolveAll = resolve;

    const decode = (i) => {
      if (i < 0 || i >= set.count || !s.blobs[i] || s.cache.has(i) || s.pending.has(i) || s.failed.has(i)) return;
      s.pending.add(i);
      createImageBitmap(s.blobs[i])
        .then((bitmap) => {
          s.pending.delete(i);
          if (!s.alive) return bitmap.close();
          s.cache.set(i, bitmap);
          s.version++;
          // лишнее выбрасываем, начиная с давно не нужных, но не трогаем окно вокруг s.focus
          for (const [k, b] of s.cache) {
            if (s.cache.size <= DECODED) break;
            if (Math.abs(k - s.focus) <= AHEAD) continue;
            b.close();
            s.cache.delete(k);
          }
        })
        .catch((err) => {
          s.pending.delete(i);
          s.failed.add(i);
          console.warn("LarphouseVoid: кадр", set.src(i), err);
        });
    };

    // Нужен кадр i, движемся в сторону dir (1 / −1): декодируем его и следующие по ходу
    s.need = (i, dir = 1) => {
      s.focus = i;
      const b = s.cache.get(i);
      if (b) { s.cache.delete(i); s.cache.set(i, b); } // свежий в порядке использования
      decode(i);
      for (let d = 1; d <= AHEAD; d++) decode(i + d * dir);
      decode(i - dir);
    };
    // Ближайший уже декодированный кадр — пока нужный ещё в пути
    s.nearest = (i) => {
      let best = -1;
      for (const k of s.cache.keys()) if (best < 0 || Math.abs(k - i) < Math.abs(best - i)) best = k;
      return best;
    };
    // Декодировать n кадров от from в сторону dir и дождаться их (перед проигрыванием).
    // Битые или не скачавшиеся кадры не ждём, и в любом случае не дольше WARM_LIMIT:
    // страница в это время заперта, и держать её вечно из-за одного файла нельзя
    s.warm = (from, dir, n) => {
      s.need(from, dir);
      const t0 = performance.now();
      return new Promise((done) => {
        const check = () => {
          let ok = true;
          for (let d = 0; d < n; d++) {
            const k = from + d * dir;
            if (k >= 0 && k < set.count && s.blobs[k] && !s.cache.has(k) && !s.failed.has(k)) ok = false;
          }
          if (ok || performance.now() - t0 > WARM_LIMIT) done();
          else setTimeout(check, 30);
        };
        check();
      });
    };
    store.current = s;
  }

  useEffect(() => {
    const s = store.current;
    s.alive = true;
    let cancelled = false;

    const fetchOne = async (i) => {
      if (s.blobs[i]) return;
      try {
        const res = await fetch(set.src(i));
        // dev-сервер на отсутствующий файл отдаёт index.html со статусом 200 — это не кадр
        const type = res.headers.get("content-type") || "";
        if (!res.ok || !type.startsWith("image/")) throw new Error(`${res.status} ${type}`);
        s.blobs[i] = await res.blob();
      } catch (err) {
        console.warn("LarphouseVoid: кадр", set.src(i), err);
      }
    };

    (async () => {
      await fetchOne(0);
      if (cancelled) return;
      s.need(s.focus);
      onFirst?.();
      const queue = Array.from({ length: set.count - 1 }, (_, i) => i + 1);
      await Promise.all(
        Array.from({ length: PARALLEL }, async () => {
          while (queue.length && !cancelled) await fetchOne(queue.shift());
        }),
      );
      if (cancelled) return;
      s.resolveAll();
      s.need(s.focus);                                // окно вокруг текущего кадра, когда всё пришло
    })();

    return () => {
      cancelled = true;
      s.alive = false;
      s.cache.forEach((bitmap) => bitmap.close());
      s.cache.clear();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return store;
}

function LarphouseVoid({ scrollLength = "+=190%" }) {
  const sectionRef = useRef();
  const voidCanvasRef = useRef();
  const larphouseCanvasRef = useRef();
  const [ready, setReady] = useState(false);

  const voidFrames = useFrames(VOID, () => setReady(true));
  const larphouseFrames = useFrames(LARPHOUSE);

  // Спрайты угловых лоз: четыре небольших файла, грузим сразу
  const cornerSheets = useRef([]);
  useEffect(() => {
    let cancelled = false;
    CORNERS.forEach(async (c, k) => {
      try {
        const bitmap = await createImageBitmap(await (await fetch(cornerSrc(c.name))).blob());
        if (cancelled) return bitmap.close();
        cornerSheets.current[k] = bitmap;
        larphouseFrames.current.version++;             // перерисовать слой Larphouse
      } catch (err) {
        console.warn("LarphouseVoid: угол", c.name, err);
      }
    });
    return () => {
      cancelled = true;
      cornerSheets.current.forEach((bitmap) => bitmap?.close());
      cornerSheets.current = [];
    };
  }, [larphouseFrames]);

  // Положение обеих анимаций живёт вне React: меняется 60 раз в секунду,
  // и каждый setState перерисовывал бы компонент впустую
  const play = useRef({ voidFrame: 0, larphouseTime: 0 });

  useEffect(() => {
    const layers = [
      // Void — cover: кадр заполняет экран, проём в центре (на вертикальных экранах обрезаются бока)
      { canvas: voidCanvasRef.current, frames: voidFrames, fit: Math.max, index: () => play.current.voidFrame },
      // Larphouse — contain: надпись вписана целиком и на вертикальных экранах
      {
        canvas: larphouseCanvasRef.current, frames: larphouseFrames, fit: Math.min,
        index: () => Math.round(play.current.larphouseTime * LARPHOUSE.fps),
        // лозы — по углам самого экрана, а не кадра
        extra: (ctx, W, H) => {
          const t = play.current.larphouseTime;
          const k = Math.min(W, H) / CORNER.page;
          const w = CORNER.w * k, h = CORNER.h * k, m = CORNER.margin * k;
          CORNERS.forEach((c, i) => {
            const sheet = cornerSheets.current[i];
            const p = cornerProgress(t, c.time);
            if (!sheet || p <= 0) return;
            const j = Math.round(p * (CORNER.steps - 1));
            const sx = (j % CORNER.cols) * CORNER.w, sy = Math.floor(j / CORNER.cols) * CORNER.h;
            const [x, y] = c.place(W, H, w, h, m);
            ctx.drawImage(sheet, sx, sy, CORNER.w, CORNER.h, x, y, w, h);
          });
        },
      },
    ].map((l) => ({ ...l, ctx: l.canvas.getContext("2d"), drawn: -1, version: -1, want: -1, dir: 1 }));

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      for (const l of layers) {
        l.canvas.width = Math.round(l.canvas.clientWidth * dpr);
        l.canvas.height = Math.round(l.canvas.clientHeight * dpr);
        l.drawn = -1;
      }
    };

    const draw = () => {
      for (const l of layers) {
        const want = l.index();
        const store = l.frames.current;
        if (want !== l.want) {
          l.dir = want < l.want ? -1 : 1;
          l.want = want;
          store.need(want, l.dir);
        }
        if (want === l.drawn && store.version === l.version) continue;
        l.version = store.version;
        const k = store.nearest(want);
        const { width: W, height: H } = l.canvas;
        l.ctx.clearRect(0, 0, W, H);
        if (k >= 0) {
          const s = l.fit(W / FRAME_W, H / FRAME_H);
          l.ctx.drawImage(store.cache.get(k), (W - FRAME_W * s) / 2, (H - FRAME_H * s) / 2, FRAME_W * s, FRAME_H * s);
        }
        l.extra?.(l.ctx, W, H);
        l.drawn = k === want ? want : -1;
      }
    };

    const observer = new ResizeObserver(resize);
    layers.forEach((l) => observer.observe(l.canvas));
    resize();

    // Рисуем в общем тикере gsap, а не в своём requestAnimationFrame: на нём же
    // крутится Lenis, и два разных цикла дали бы дрожание на кадр
    gsap.ticker.add(draw);

    return () => {
      gsap.ticker.remove(draw);
      observer.disconnect();
    };
  }, [voidFrames, larphouseFrames]);

  useGSAP(
    () => {
      const state = play.current;
      let busy = false;                                // пока Larphouse играет, скролл стоит и события игнорируются

      // Проиграть Larphouse до конца (to = длительность) или назад к началу (to = 0),
      // заперев страницу в конце секции
      // Кадры Larphouse могли ещё не догрузиться (дошли до конца Void быстрее, чем пришли
      // 14 МБ): ждём их при запертой прокрутке, а заодно декодируем первые кадры по ходу
      const runLarphouse = (st, to) => {
        busy = true;
        const frames = larphouseFrames.current;
        const from = Math.round(state.larphouseTime * LARPHOUSE.fps);
        const dir = to > state.larphouseTime ? 1 : -1;
        lockScrollAt(st.end, () => frames.all.then(() => frames.warm(from, dir, AHEAD)).then(() => {
          gsap.to(state, {
            larphouseTime: to,
            duration: Math.abs(to - state.larphouseTime) / LARPHOUSE_SPEED,
            ease: "none",
            overwrite: true,
            onComplete: () => {
              // Страница стоит ровно в конце секции: следующее обновление ScrollTrigger
              // не должно принять это за новый проход
              busy = false;
              unlockScroll();
            },
          });
        }));
      };

      const proxy = { frame: 0 };
      gsap.to(proxy, {
        frame: VOID.count - 1,
        ease: "none",
        onUpdate: () => {
          state.voidFrame = Math.round(proxy.frame);
        },
        scrollTrigger: {
          trigger: sectionRef.current,
          start: "top top",
          end: scrollLength,
          pin: true,
          scrub: 0.6,
          invalidateOnRefresh: true,

          onUpdate: (self) => {
            if (busy) return;
            const atEnd = self.progress >= 1;
            // Дошли вниз до конца Void, а Larphouse ещё не сыгран — играем вперёд
            if (atEnd && self.direction === 1 && state.larphouseTime < LARPHOUSE_DURATION) {
              runLarphouse(self, LARPHOUSE_DURATION);
            }
            // Пошли назад из конца, а Larphouse сыгран — сначала он проигрывается назад
            else if (!atEnd && self.direction === -1 && state.larphouseTime > 0) {
              runLarphouse(self, 0);
            }
          },

          // Страницу открыли уже ниже секции: Larphouse считается сыгранным,
          // чтобы при прокрутке вверх он проигрался назад
          onRefresh: (self) => {
            if (!busy && self.progress >= 1) state.larphouseTime = LARPHOUSE_DURATION;
          },
        },
      });

      return () => {
        if (busy) unlockScroll();
      };
    },
    { scope: sectionRef, dependencies: [scrollLength] },
  );

  return (
    <section ref={sectionRef} className={`larphouse-void${ready ? "" : " larphouse-void--loading"}`}>
      <canvas ref={voidCanvasRef} className="larphouse-void__canvas" aria-hidden="true" />
      <canvas ref={larphouseCanvasRef} className="larphouse-void__canvas" aria-label="Larphouse" role="img" />
    </section>
  );
}

export default LarphouseVoid;

# Larphouse × Void: проход сквозь арку к надписи

7 секунд, 1920×1080, 60 fps, со звуком:

- `larphouse-void.webm`: VP9 с альфа-каналом — **прозрачный фон** вокруг арки (видны только точки
  камня), бумага страницы проявляется из прозрачного, пока проходим арку, и в конце непрозрачная;
- `larphouse-void.mp4`: та же анимация на чёрном фоне, без прозрачности;
- `contact-sheet.png`: раскадровка (на чёрном).

Камера от первого лица идёт к романской арке, набранной пунктиром (Void). Проём тёмный, в нём
начинает проступать надпись, а под ней, пока мы проходим арку, плавно появляется бумага. Камера проходит под аркой, страница
заполняет экран, и дальше разыгрывается анимация Larphouse: буквы, лозы по углам, меч с розами, кот.

| Время | Что происходит |
|---|---|
| 0.0–2.4 с | подход к арке; проём тёмный (прозрачный) |
| 2.3–3.5 с | проход под аркой; в проёме проступают первые буквы «Larphouse» |
| 2.6–4.8 с | бумага страницы плавно проявляется из прозрачного; арка уходит за края, страница заполняет кадр |
| 1.8–6.1 с | анимация Larphouse целиком (её время сдвинуто на 1.8 с): буквы 2.3–4.7 с, кот 2.8–6.1 с |
| до 7.0 с | финальная композиция |

Звук — от Void: ветер и шаги снаружи, гулкие шаги под аркой, низкий тон; после прохода он
затихает к 5 с.

Страница за дверью — кадр `../larphouse/larphouse.py` на бумаге; пока мы идём, она масштабируется
от 0.82 до 1 и светлеет, как плоскость за проёмом. Кадры Void рендерятся заранее из
[riso-windowseat](https://github.com/sevenevesai/riso-windowseat) `films/void-wide/index.html` в два
набора: `?alpha=1` (непрозрачная земля, прозрачный проём) → `out/void/` вместе с `void.wav`, и
`?alpha=1&clear=1` (только точки камня) → `out/void_clear/`. Первый даёт маску проёма, второй — точки.

Для сайта (компонент `LarphouseVoid` в larphouse) из прозрачных кадров берётся каждый второй
(30 fps, 210 кадров) и сжимается в AVIF: `avifenc -q 70 --qalpha 90 -s 6`.

```
pip install pillow numpy opencv-python-headless scipy imageio-ffmpeg
python3 combine.py render larphouse-void          # → larphouse-void.mp4 (на чёрном)
python3 combine.py frames out/rgba                # → прозрачные RGBA PNG
ffmpeg -framerate 60 -i out/rgba/f_%04d.png -i out/void/void.wav \
  -filter_complex "[1:a]afade=t=out:st=4.0:d=1.0,apad=whole_dur=7[a]" -map 0:v -map "[a]" \
  -c:v libvpx-vp9 -pix_fmt yuva420p -b:v 9M -maxrate 12M -bufsize 18M -auto-alt-ref 0 \
  -c:a libopus -t 7 larphouse-void.webm
python3 combine.py still 3.2 out/still.png
python3 combine.py sheet 0.2:6.9:0.56 contact-sheet.png
```

Тайминги — константы `T_TITLE`, `T_THROUGH`, `T_PAPER`, `PAGE_SCALE` в начале `combine.py`.

## На сайте

`react/LarphouseVoid.jsx` — секция для larphouse (React, GSAP ScrollTrigger, Lenis). Void идёт по
прокрутке (кадры `?alpha=1&clear=1`, только точки камня, обрезаны на кадре, где дверь ушла за края);
в конце прокрутка запирается и сама проигрывается Larphouse в светлой печати (`larphouse.py frames
--light --no-corners`), угловые лозы рисуются по углам экрана из спрайтов `larphouse.py corners`.
При прокрутке назад Larphouse так же сам проигрывается задом наперёд. `react/scroll.js` — общий
экземпляр Lenis и запирание прокрутки без скачка ширины (плюс правило для `html.lenis-stopped` в
index.css, см. комментарий в `scroll.js`).

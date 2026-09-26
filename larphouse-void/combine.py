"""Larphouse × Void — 7 s, 1920×1080, 60 fps.

The camera walks up to the Romanesque arch (Void) and through it. In the opening glows a paper
page; once through, the page fills the screen and the Larphouse title prints itself on it
(letters bleed in, vines grow, the sword with roses, the cat).

Void's frames come pre-rendered (opaque ink ground, transparent opening) from
riso-windowseat/films/void-wide (`?alpha=1`) into out/void/; Larphouse is rendered here from
../larphouse/larphouse.py. Every frame is a pure function of t.

  python3 combine.py still 3.2 out/still.png
  python3 combine.py sheet 0.2:6.9:0.56 contact-sheet.png
  python3 combine.py render larphouse-void [--fps 60]    # → .mp4 with sound, on black
  python3 combine.py frames out/rgba [--fps 60]           # transparent RGBA PNGs
  python3 combine.py sheet 0.2:6.9:0.56 out/a.png --alpha
"""
import os, sys, subprocess
import numpy as np
from PIL import Image, ImageDraw
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'larphouse'))
import larphouse as LH                                  # noqa: E402

W, H = 1920, 1080
DURATION = 7.0
VOID_FPS, VOID_LEN = 60, 5.0
VOID_DIR = os.path.join(HERE, 'out', 'void')            # opaque ink ground, transparent opening
CLEAR_DIR = os.path.join(HERE, 'out', 'void_clear')     # the stone's dots only, everything else transparent

# timeline (seconds)
T_TITLE = 1.8            # Larphouse's own clock starts here: letters bleed in 2.3–4.7 s (first through the door), the cat runs 2.8–6.1 s
T_THROUGH = 4.7          # the lens is past the arch; from here the page fills the screen
T_PAPER = (2.6, 4.8)     # the page's paper fades in from transparent while we pass under the arch
PAGE_SCALE = (0.82, 1.0)  # the page, seen as a plane beyond the door, comes closer as we walk
PAPER = np.array(LH.PREVIEW_BG, np.float32) / 255


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


class Combined:
    def __init__(self):
        self.lh = LH.Film()
        self._void = {}

    def void(self, t, d=None):
        """Void frame at t (RGBA float); past its end the frame is empty — we are in the void."""
        d = d or VOID_DIR
        i = int(round(t * VOID_FPS))
        if i >= round(VOID_LEN * VOID_FPS):
            return None
        if (d, i) not in self._void:
            if len(self._void) > 4:
                self._void.clear()
            self._void[(d, i)] = np.asarray(Image.open(os.path.join(d, f'f_{i:04d}.png')).convert('RGBA')).astype(np.float32) / 255
        return self._void[(d, i)]

    def page(self, t):
        """The Larphouse title as seen through the door: straight-alpha RGB and alpha, scaled about
        the centre. The paper under the ink starts transparent (the opening is dark) and fades in
        over T_PAPER, as we pass under the arch; the ink itself is always there."""
        f = self.lh.frame(max(0.0, t - T_TITLE)).astype(np.float32) / 255
        ink_rgb, ink_a = f[..., :3], f[..., 3:4]
        s = PAGE_SCALE[0] + (PAGE_SCALE[1] - PAGE_SCALE[0]) * smooth(t / T_THROUGH)
        paper_a = np.ones((H, W, 1), np.float32)
        if s < 0.999:                                            # warp premultiplied, so edges stay clean
            M = np.array([[s, 0, W / 2 * (1 - s)], [0, s, H / 2 * (1 - s)]], np.float32)
            warp = lambda x: cv2.warpAffine(x, M, (W, H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            pre = warp(ink_rgb * ink_a)
            ink_a = warp(ink_a)[..., None]
            ink_rgb = pre / np.maximum(ink_a, 1e-6)
            paper_a = warp(paper_a[..., 0])[..., None]
        paper_a = paper_a * smooth((t - T_PAPER[0]) / (T_PAPER[1] - T_PAPER[0]))
        a = ink_a + paper_a * (1 - ink_a)
        rgb = (ink_rgb * ink_a + PAPER * paper_a * (1 - ink_a)) / np.maximum(a, 1e-6)
        # the page is lit from beyond the door: it brightens as it comes into view
        glow = 0.55 + 0.45 * smooth((t - 0.4) / (T_THROUGH - 0.4))
        return rgb * glow, a

    def frame_rgba(self, t):
        """The black ground around the arch is transparent: only the stone's dots print. The page
        shows only through the opening (Void's own mask), under the dots; once through the arch the
        page is the whole frame (opaque as soon as its paper has faded in)."""
        page, pa = self.page(t)
        v = self.void(t)
        if v is None:
            return (np.dstack([np.clip(page, 0, 1), pa]) * 255 + 0.5).astype(np.uint8)
        hole = (1 - v[..., 3:4]) * pa                            # the opening, as far as the page covers it
        dots = self.void(t, CLEAR_DIR)
        da = dots[..., 3:4]
        a = da + hole * (1 - da)
        rgb = (dots[..., :3] * da + page * hole * (1 - da)) / np.maximum(a, 1e-6)
        return (np.dstack([np.clip(rgb, 0, 1), a]) * 255 + 0.5).astype(np.uint8)

    def frame(self, t):
        """The same frame on black (for the MP4 and sheets)."""
        f = self.frame_rgba(t).astype(np.float32) / 255
        return (np.clip(f[..., :3] * f[..., 3:4], 0, 1) * 255 + 0.5).astype(np.uint8)

def sheet(film, times, path, cols=4, alpha=False):
    tiles = []
    for t in times:
        if alpha:                                               # transparency shown over a checker
            f = film.frame_rgba(t).astype(np.float32) / 255
            yy, xx = np.mgrid[0:H, 0:W]
            chk = np.where(((xx // 40 + yy // 40) % 2)[..., None] == 0, 0.35, 0.5).astype(np.float32)
            img = (f[..., :3] * f[..., 3:4] + chk * (1 - f[..., 3:4])) * 255
            im = Image.fromarray(cv2.resize(img.astype(np.uint8), (480, 270), interpolation=cv2.INTER_AREA))
        else:
            im = Image.fromarray(cv2.resize(film.frame(t), (480, 270), interpolation=cv2.INTER_AREA))
        ImageDraw.Draw(im).text((8, 6), f't={t:.2f}', fill=(200, 0, 0))
        tiles.append(np.asarray(im))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)]
    Image.fromarray(np.concatenate(rows, 0)).save(path)


if __name__ == '__main__':
    cmd = sys.argv[1]
    film = Combined()
    if cmd == 'still':
        Image.fromarray(film.frame(float(sys.argv[2]))).save(sys.argv[3])
    elif cmd == 'sheet':
        a, b, s = map(float, sys.argv[2].split(':'))
        sheet(film, list(np.arange(a, b + 1e-6, s)), sys.argv[3], alpha='--alpha' in sys.argv)
    elif cmd == 'frames':
        # RGBA PNG per frame, for the site (→ AVIF) and the alpha video
        out = sys.argv[2]
        fps = int(sys.argv[sys.argv.index('--fps') + 1]) if '--fps' in sys.argv else 60
        os.makedirs(out, exist_ok=True)
        n = int(round(DURATION * fps))
        for i in range(n):
            Image.fromarray(film.frame_rgba(i / fps)).save(os.path.join(out, f'f_{i:04d}.png'), compress_level=1)
            if i % 60 == 0:
                print('frame', i, '/', n, flush=True)
    elif cmd == 'render':
        stem = sys.argv[2]
        fps = int(sys.argv[sys.argv.index('--fps') + 1]) if '--fps' in sys.argv else 60
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        # Void's sound runs 5 s; the rest is its tone's tail into silence (apad to the full length)
        p = subprocess.Popen([ff, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                              '-r', str(fps), '-i', '-', '-i', os.path.join(VOID_DIR, 'void.wav'),
                              '-filter_complex', f'[1:a]afade=t=out:st=4.0:d=1.0,apad=whole_dur={DURATION}[a]',
                              '-map', '0:v', '-map', '[a]', '-c:v', 'libx264', '-preset', 'slow', '-crf', '22',
                              '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart',
                              '-t', str(DURATION), stem + '.mp4'], stdin=subprocess.PIPE)
        n = int(round(DURATION * fps))
        for i in range(n):
            p.stdin.write(film.frame(i / fps).tobytes())
            if i % 60 == 0:
                print('frame', i, '/', n, flush=True)
        p.stdin.close(); p.wait()
        print('wrote', stem + '.mp4', n, 'frames')

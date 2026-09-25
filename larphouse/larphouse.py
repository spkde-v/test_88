"""Larphouse — 5 s gothic title animation.

Every frame is a pure function of time: frame(t) never depends on an earlier frame,
and all noise is seeded once at build time, so any frame can be rendered alone.

  python3 larphouse.py still 4.9 out/final.png
  python3 larphouse.py sheet 0.5:3.0:0.25 out/sheet.png
  python3 larphouse.py render out/larphouse.mp4 [--fps 60]
"""
import sys, os, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'refs')
W, H = 1920, 1080
DURATION = 5.0

PAPER = np.array([238, 229, 211], np.float32) / 255
INK = np.array([33, 27, 30], np.float32) / 255      # warm overprint dark, never pure black
RED = np.array([150, 28, 38], np.float32) / 255

# ---------------------------------------------------------------- timeline (seconds)
T_LETTERS = (0.5, 2.9)       # glyphs bleed in, left to right
T_L_EXTRA = 0.35             # the capital runs longer than the others
STAGGER = 0.13
T_SWORDS = (1.5, 3.5)
T_CORNERS = (1.4, 3.8)
T_ROSE = (2.2, 3.8)
T_GLEAM = (3.7, 4.55)


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def ease_io(x):
    x = min(max(x, 0.0), 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def span(t, a, b):
    return min(max((t - a) / (b - a), 0.0), 1.0)


# ---------------------------------------------------------------- assets
def ink_from(path, lo=0.12, hi=0.86):
    """Luminance -> ink coverage 0..1 (paper is 0)."""
    a = np.asarray(Image.open(path).convert('L')).astype(np.float32) / 255
    return np.clip((hi - a) / (hi - lo), 0, 1)


def fit(ink, height=None, width=None):
    h, w = ink.shape
    s = height / h if height else width / w
    return cv2.resize(ink, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def crop(ink, pad=6):
    ys, xs = np.where(ink > 0.05)
    return ink[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad):xs.max() + pad]


def rank_noise(shape, sigma, seed):
    """Smooth noise remapped to a uniform 0..1 distribution so reveals progress evenly."""
    rng = np.random.default_rng(seed)
    n = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), sigma)
    r = n.ravel().argsort().argsort().astype(np.float32) / (n.size - 1)
    return r.reshape(shape)


def geodesic(mask, seeds, step=3):
    """Distance travelled along the ink from the seed points (so vines grow along their stems)."""
    h, w = mask.shape
    sm = cv2.resize(mask.astype(np.uint8), (w // step + 1, h // step + 1), interpolation=cv2.INTER_NEAREST) > 0
    sm = cv2.dilate(sm.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    idx = -np.ones(sm.shape, np.int64)
    ys, xs = np.nonzero(sm)
    idx[ys, xs] = np.arange(len(ys))
    rows, cols, wts = [], [], []
    for dy, dx, c in ((0, 1, 1), (1, 0, 1), (1, 1, 1.414), (1, -1, 1.414)):
        y2, x2 = ys + dy, xs + dx
        ok = (y2 < sm.shape[0]) & (x2 >= 0) & (x2 < sm.shape[1])
        ok[ok] = sm[y2[ok], x2[ok]]
        rows.append(idx[ys[ok], xs[ok]]); cols.append(idx[y2[ok], x2[ok]]); wts.append(np.full(ok.sum(), c))
    g = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(len(ys),) * 2)
    src = []
    for sy, sx in seeds:
        d2 = (ys - sy / step) ** 2 + (xs - sx / step) ** 2
        src.append(int(d2.argmin()))
    d = dijkstra(g, directed=False, indices=src, min_only=True)
    far = np.isfinite(d)
    d[~far] = d[far].max() if far.any() else 0
    field = np.zeros(sm.shape, np.float32)
    field[ys, xs] = d / max(d.max(), 1e-6)
    # spread to neighbours so the full-res mask never samples an empty cell
    field = np.where(sm, field, 1.0)
    field = cv2.erode(field, np.ones((3, 3), np.uint8))
    return cv2.resize(field, (w, h), interpolation=cv2.INTER_LINEAR)


class Layer:
    """One piece of artwork placed on the page with its own reveal field."""

    def __init__(self, ink, x, y, field, color='ink', soft=0.16, stain=0.55, blur=7):
        self.ink = ink.astype(np.float32)
        self.x, self.y = int(round(x)), int(round(y))
        self.field = field.astype(np.float32)     # 0 = appears first, 1 = last
        self.soft = soft
        self.stain = cv2.GaussianBlur(self.ink, (0, 0), blur) * stain
        self.color = color

    def coverage(self, p):
        """p: 0..1 progress. Early pixels are a pale blurred stain; they sharpen as they darken."""
        if p <= 0:
            return None
        q = np.clip((p * (1 + self.soft) - self.field) / self.soft, 0, 1)
        q = q * q * (3 - 2 * q)
        dens = np.clip(q * 1.35, 0, 1)
        sharp = np.clip((q - 0.25) / 0.75, 0, 1)
        return (self.stain * (1 - sharp) + self.ink * sharp) * dens


def engrave(mask, seed):
    """Give flat font glyphs the engraved, lit-metal look of the reference capital:
    solid ink body, a stippled bevel on the lit (upper-left) side of each stroke."""
    m = cv2.GaussianBlur(mask, (0, 0), 0.8)
    solid = (mask > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(solid, cv2.DIST_L2, 5)
    body = cv2.GaussianBlur(dist, (0, 0), 2.5)
    gy, gx = np.gradient(body)
    lit = np.clip(-(gx * 0.55 + gy * 0.85), 0, None)
    lit = np.clip(lit / (np.percentile(lit[solid > 0], 97) + 1e-6), 0, 1)
    bevel = np.exp(-((dist - 5.0) / 2.6) ** 2) * lit                 # a lit ridge just inside the edge
    rng = np.random.default_rng(seed)
    dots = cv2.GaussianBlur(rng.random(mask.shape).astype(np.float32), (0, 0), 0.9)
    dots = (dots - dots.mean()) / dots.std()
    holes = smooth((bevel * 2.2 - 0.35 + dots * 0.35 * np.clip(bevel * 3, 0, 1)) / 0.5)
    # a faint secondary stipple over the lit faces so the body is not flat
    face = smooth((lit * 0.5 - 0.3 + dots * 0.2 * lit) / 0.5) * 0.18
    ink = m * (1 - np.maximum(holes * 0.85, face) * (dist > 1.6))
    return np.clip(ink, 0, 1)


def build():
    A = {}
    # --- the capital from the reference
    L = crop(ink_from(os.path.join(REF, 'letter-L.webp')))
    L = fit(L, height=610)
    lh, lw = L.shape

    # --- lowercase from UnifrakturCook, measured so its baseline meets the L's foot bar
    font = ImageFont.truetype(os.path.join(REF, 'UnifrakturCook-Bold.ttf'), 330)
    word = 'arphouse'
    glyphs = []
    x = 0
    for ch in word:
        bb = font.getbbox(ch)
        adv = font.getlength(ch)
        glyphs.append((ch, x, bb))
        x += adv
    total_w = x
    asc = max(-min(b[1] for _, _, b in glyphs), 0)
    canvas = Image.new('L', (int(total_w) + 40, 420), 0)
    d = ImageDraw.Draw(canvas)
    base_in = 300                                                      # baseline row inside canvas
    per = []
    for ch, gx, bb in glyphs:
        g = Image.new('L', canvas.size, 0)
        ImageDraw.Draw(g).text((20 + gx, base_in), ch, font=font, fill=255, anchor='ls')
        per.append(np.asarray(g).astype(np.float32) / 255)
    # x-height from 'o'
    o = per[word.index('o')]
    ys = np.where(o.max(1) > 0.5)[0]
    xheight = ys.max() - ys.min()

    # page layout
    L_foot = 0.845                          # fraction of L height where its foot bar sits (baseline)
    word_scale = 1.0
    gap = -34                               # lowercase tucks under the L's foot sweep
    comp_w = lw + gap + (total_w + 40) * word_scale
    left = (W - comp_w) / 2
    top_L = 175
    baseline = top_L + lh * L_foot
    A['L'] = (L, left, top_L)
    A['glyphs'] = []
    for i, gm in enumerate(per):
        ys_, xs_ = np.where(gm > 0.02)
        y0, y1, x0, x1 = ys_.min() - 4, ys_.max() + 4, xs_.min() - 4, xs_.max() + 4
        g = gm[y0:y1, x0:x1]
        gx = left + lw + gap + x0
        gy = baseline - (base_in - y0)
        A['glyphs'].append((engrave(g, 100 + i), gx, gy))
    A['baseline'] = baseline
    A['word_box'] = (left, top_L, left + comp_w, baseline + 130)
    return A


def make_layers():
    A = build()
    layers = {}
    L, lx, ly = A['L']
    h, w = L.shape
    # the capital unfurls from the middle of its stem out to the scroll tips
    geo = geodesic(L > 0.3, [(h * 0.55, w * 0.33)], step=2)
    fL = 0.62 * geo + 0.38 * rank_noise(L.shape, 14, 1)
    layers['L'] = Layer(L, lx, ly, fL, soft=0.45, blur=9)

    layers['glyphs'] = []
    for i, (g, gx, gy) in enumerate(A['glyphs']):
        gh, gw = g.shape
        yy = np.linspace(0, 1, gh, dtype=np.float32)[:, None] * np.ones((1, gw), np.float32)
        f = 0.55 * rank_noise(g.shape, 16, 10 + i) + 0.3 * yy + 0.15 * rank_noise(g.shape, 2.5, 30 + i)
        f = (f - f.min()) / (f.max() - f.min())
        layers['glyphs'].append(Layer(g, gx, gy, f, soft=0.5, blur=8))

    # corner vines, mirrored into all four corners; they grow from the corner along the stems
    cv = crop(ink_from(os.path.join(REF, 'corner-vine.png'), lo=0.2, hi=0.75))
    cv = fit(cv, height=372)
    ch, cw = cv.shape
    ys, xs = np.nonzero(cv > 0.4)
    corner_pt = np.argmax(xs * 0.6 + ys)                              # bottom-right-most ink
    geo = geodesic(cv > 0.35, [(ys[corner_pt], xs[corner_pt])], step=2)
    fc = 0.85 * geo + 0.15 * rank_noise(cv.shape, 6, 3)
    m = 38
    layers['corners'] = [
        Layer(cv, W - m - cw, H - m - ch, fc, soft=0.25),                                     # bottom right
        Layer(cv[:, ::-1], m, H - m - ch, fc[:, ::-1], soft=0.25),                            # bottom left
        Layer(cv[::-1, :], W - m - cw, m, fc[::-1, :], soft=0.25),                            # top right
        Layer(cv[::-1, ::-1], m, m, fc[::-1, ::-1], soft=0.25),                              # top left
    ]

    # crossed rapiers behind the word, pale; they condense from paper tone outward from the cross
    sw = crop(ink_from(os.path.join(REF, 'crossed-swords.png'), lo=0.15, hi=0.8))
    sw = fit(sw, height=600)
    sh, sww = sw.shape
    yy, xx = np.mgrid[0:sh, 0:sww].astype(np.float32)
    r = np.hypot((xx - sww * 0.5) / sww, (yy - sh * 0.47) / sh)
    fs = 0.45 * (r / r.max()) + 0.55 * rank_noise(sw.shape, 22, 4)
    fs = (fs - fs.min()) / (fs.max() - fs.min())
    layers['swords'] = Layer(sw * 0.27, (W - sww) / 2 + 40, 455 - sh / 2, fs, soft=0.3, blur=10)

    # sword wrapped in a thorned rose, laid flat under the word as a divider
    rs = ink_from(os.path.join(REF, 'rose-sword.png'), lo=0.1, hi=0.78)
    # flower heads (reference pixels, x, y, r) take the red plate inside their outlines only
    blooms = [(480, 380, 78), (178, 725, 80), (65, 145, 45), (520, 880, 30)]
    # close the outline gaps, then flood the outside: what remains unfilled is inside a shape
    closed = cv2.morphologyEx((rs > 0.35).astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    flood = closed.copy()
    cv2.floodFill(flood, np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8), (0, 0), 1)
    enclosed = ((flood == 0) | (closed > 0)) & (rs < 0.35)
    yy, xx = np.mgrid[0:rs.shape[0], 0:rs.shape[1]]
    disc = np.zeros(rs.shape, bool)
    for bx, by, br in blooms:
        disc |= np.hypot(xx - bx, yy - by) < br
    petal = cv2.GaussianBlur((enclosed & disc).astype(np.float32), (0, 0), 1.2) * (1 - rs)
    # lay the sword flat: pommel (185,15) to point (475,1065) becomes horizontal, point to the right
    ang = np.degrees(np.arctan2(1050, 290))
    hh, ww = rs.shape
    M = cv2.getRotationMatrix2D((ww / 2, hh / 2), ang, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(hh * sin + ww * cos), int(hh * cos + ww * sin)
    M[0, 2] += nw / 2 - ww / 2; M[1, 2] += nh / 2 - hh / 2
    rs = cv2.warpAffine(rs, M, (nw, nh), flags=cv2.INTER_CUBIC)
    petal = cv2.warpAffine(petal, M, (nw, nh), flags=cv2.INTER_CUBIC)
    ys_, xs_ = np.where(rs > 0.05)
    y0, y1, x0, x1 = ys_.min() - 6, ys_.max() + 6, xs_.min() - 6, xs_.max() + 6
    rs, petal = np.clip(rs[y0:y1, x0:x1], 0, 1), np.clip(petal[y0:y1, x0:x1], 0, 1)
    rs = fit(rs, width=620); petal = fit(petal, width=620)
    rh, rw = rs.shape
    xx = np.linspace(0, 1, rw, dtype=np.float32)[None, :] * np.ones((rh, 1), np.float32)
    fr = 0.55 * xx + 0.45 * rank_noise(rs.shape, 8, 5)          # hilt (left) first, then to the point
    fr = (fr - fr.min()) / (fr.max() - fr.min())
    rx, ry = (W - rw) / 2 + 40, 900 - rh / 2
    layers['rose'] = Layer(rs, rx, ry, fr, soft=0.35)
    fp = np.clip(fr + 0.25, 0, 1)                                       # blooms open last
    layers['rose_red'] = Layer(petal * 0.9, rx + 2, ry + 1, fp, color='red', soft=0.25, blur=5)
    layers['A'] = A
    return layers


def make_paper():
    rng = np.random.default_rng(7)
    fine = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 0.7)
    fib = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), sigmaX=6, sigmaY=0.8)
    blot = cv2.GaussianBlur(rng.standard_normal((H // 8, W // 8)).astype(np.float32), (0, 0), 9)
    blot = cv2.resize(blot / blot.std(), (W, H), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vig = ((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2
    lum = 1 + 0.018 * fine / fine.std() + 0.006 * fib / fib.std() + 0.02 * blot - 0.13 * vig ** 1.4
    paper = PAPER[None, None, :] * lum[..., None]
    # tint the blotches toward foxing brown
    paper[..., 2] -= 0.02 * np.clip(blot, 0, None)
    # ink wear: the print is old, so density varies and paper fibre shows through
    wear = 0.9 + 0.08 * np.clip(blot * 0.5 + 0.5, 0, 1) - 0.06 * np.clip(fine / fine.std(), 0, None) * 0.5
    grain = [cv2.GaussianBlur(np.random.default_rng(20 + k).standard_normal((H, W)).astype(np.float32), (0, 0), 1.1)
             for k in range(3)]
    grain = [g / g.std() for g in grain]
    return paper.astype(np.float32), np.clip(wear, 0.7, 1).astype(np.float32), grain


class Film:
    def __init__(self):
        self.lay = make_layers()
        self.paper, self.wear, self.grain = make_paper()
        A = self.lay['A']
        self.baseline = A['baseline']
        rng = np.random.default_rng(99)
        n = 70
        self.specks = np.stack([rng.uniform(0, W, n), rng.uniform(0, H, n), rng.uniform(0.6, 2.2, n),
                                rng.uniform(-9, 9, n), rng.uniform(-14, -3, n), rng.uniform(0, 1, n)], 1)

    def progress(self):
        pass

    def frame(self, t):
        ink = np.zeros((H, W), np.float32)
        red = np.zeros((H, W), np.float32)
        lay = self.lay

        def put(layer, p, target=None):
            c = layer.coverage(p)
            if c is None:
                return
            tgt = red if layer.color == 'red' else ink
            h, w = c.shape
            x, y = layer.x, layer.y
            sl = tgt[y:y + h, x:x + w]
            np.maximum(sl, c[:sl.shape[0], :sl.shape[1]], out=sl)

        a, b = T_LETTERS
        glyph_len = (b - a) - STAGGER * len(lay['glyphs'])
        put(lay['swords'], ease_io(span(t, *T_SWORDS)))
        # knock the pale swords out behind the word so the letters own their value
        put(lay['L'], span(t, a, a + glyph_len + T_L_EXTRA))
        for i, g in enumerate(lay['glyphs']):
            s = a + STAGGER * (i + 1) + 0.05
            put(g, span(t, s, s + glyph_len))
        cp = [T_CORNERS[0] + k * 0.12 for k in range(4)]
        for k, c in enumerate(lay['corners']):
            put(c, ease_io(span(t, cp[k], cp[k] + (T_CORNERS[1] - T_CORNERS[0]) - 0.36)))
        put(lay['rose'], ease_io(span(t, *T_ROSE)))
        put(lay['rose_red'], ease_io(span(t, T_ROSE[0] + 0.3, T_ROSE[1] + 0.35)))

        # gleam: a soft diagonal band of light passes over the capital and the word
        g = span(t, *T_GLEAM)
        if 0 < g < 1:
            x0, y0, x1, y1 = lay['A']['word_box']
            c = x0 - 300 + (x1 - x0 + 600) * ease_io(g)
            yy, xx = np.mgrid[int(y0):int(y1), int(x0):int(x1)].astype(np.float32)
            band = np.exp(-(((xx + (yy - y0) * 0.45) - c) / 90) ** 2) * np.sin(np.pi * g) ** 0.7
            sl = ink[int(y0):int(y1), int(x0):int(x1)]
            sl *= 1 - 0.3 * band * np.clip(0.8 + 0.2 * self.grain[0][int(y0):int(y1), int(x0):int(x1)], 0.4, 1)

        # ink specks drifting on the bare paper, settling as the title prints
        k = 1 - 0.8 * smooth((t - 0.4) / 2.0)
        if k > 0:
            for sx, sy, r, vx, vy, ph in self.specks:
                px = sx + vx * t
                py = (sy + vy * t) % H
                al = 0.22 * k * (0.5 + 0.5 * np.sin(2 * np.pi * (ph + t * 0.35)))
                cv2.circle(ink, (int(px * 4), int(py * 4)), int(r * 4), float(al), -1, cv2.LINE_AA, shift=2)

        # breathing paper grain: three fixed textures cross-faded slowly (no per-frame noise crawl)
        ph = t * 0.6
        wts = [0.5 + 0.5 * np.cos(2 * np.pi * (ph - j / 3)) for j in range(3)]
        gr = sum(w_ * g_ for w_, g_ in zip(wts, self.grain)) / sum(wts)
        dens = self.wear * (1 + 0.05 * gr)

        a_ink = np.clip(ink * dens, 0, 1)[..., None]
        a_red = np.clip(red * dens * 0.9, 0, 1)[..., None]
        out = self.paper * (1 - a_red * (1 - RED / PAPER))
        out = out * (1 - a_ink * (1 - INK / PAPER))

        return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


def sheet(film, times, path, cols=4):
    tiles = []
    for t in times:
        f = cv2.resize(film.frame(t), (480, 270), interpolation=cv2.INTER_AREA)
        im = Image.fromarray(f)
        ImageDraw.Draw(im).text((8, 6), f't={t:.2f}', fill=(200, 0, 0))
        tiles.append(np.asarray(im))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]) + 255)
    rows = [np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)]
    Image.fromarray(np.concatenate(rows, 0)).save(path)


if __name__ == '__main__':
    cmd = sys.argv[1]
    film = Film()
    if cmd == 'still':
        Image.fromarray(film.frame(float(sys.argv[2]))).save(sys.argv[3])
    elif cmd == 'sheet':
        a, b, s = map(float, sys.argv[2].split(':'))
        sheet(film, list(np.arange(a, b + 1e-6, s)), sys.argv[3])
    elif cmd == 'render':
        out = sys.argv[2]
        fps = int(sys.argv[sys.argv.index('--fps') + 1]) if '--fps' in sys.argv else 60
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        p = subprocess.Popen([ff, '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(fps),
                              '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '14',
                              '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out], stdin=subprocess.PIPE)
        n = int(round(DURATION * fps))
        for i in range(n):
            p.stdin.write(film.frame(i / fps).tobytes())
        p.stdin.close(); p.wait()
        print('wrote', out, n, 'frames')

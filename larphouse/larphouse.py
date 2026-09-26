"""Larphouse — 5 s gothic title animation.

Every frame is a pure function of time: frame(t) never depends on an earlier frame,
and all noise is seeded once at build time, so any frame can be rendered alone.

  python3 larphouse.py still 4.9 out/final.png
  python3 larphouse.py sheet 0.5:3.0:0.25 out/sheet.png
  python3 larphouse.py render out/larphouse.mp4 [--fps 60]
  python3 larphouse.py frames out/frames [--fps 60] [--light] [--no-corners]   # RGBA PNGs; --light: pale print for dark backgrounds
  python3 larphouse.py corners out/corners [--light]   # growth sprite sheet per corner, for laying out at a page's edges
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
CAT_EYE = np.array([240, 234, 222], np.float32) / 255
# --light: the print (letters, corner vines, the sword) in pale paper colour, for dark backgrounds.
# The red plate (the roses) and the cat keep their colours.
LIGHT = '--light' in sys.argv
PRINT = np.array([238, 229, 211], np.float32) / 255 if LIGHT else INK
# --no-corners: frames without the corner vines (a page lays them out at its own edges from
# the `corners` sprite sheets instead)
NO_CORNERS = '--no-corners' in sys.argv
CORNER_STEPS = 40            # growth stages per corner in a sprite sheet
CORNER_COLS = 8

# ---------------------------------------------------------------- timeline (seconds)
T_LETTERS = (0.5, 2.9)       # glyphs bleed in, left to right
T_L_EXTRA = 0.35             # the capital runs longer than the others
STAGGER = 0.13
# corner vines: (start, duration, ease-out power) per corner — deliberately unequal so they
# don't grow in lockstep. Order: bottom right, bottom left, top right, top left.
T_CORNERS = [(0.25, 4.0, 1.3), (0.6, 3.5, 1.7), (0.0, 3.7, 1.5), (0.45, 4.3, 1.9)]
T_ROSE = (2.2, 3.8)
# the cat: trots in from the right, sees the blooming rose, bristles, jumps, bolts back out
T_CAT_IN = (1.0, 2.85)       # trot from off-screen right to its stopping mark
T_CAT_BRISTLE = (3.0, 3.2)   # fur stands up
T_CAT_JUMP = (3.2, 3.62)     # startle jump (the front-facing reference pose)
T_CAT_RUN = (3.72, 4.3)      # dash off to the right


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
    m = 38
    places = [(lambda a: a, W - m - cw, H - m - ch),                   # bottom right
              (lambda a: a[:, ::-1], m, H - m - ch),                     # bottom left
              (lambda a: a[::-1, :], W - m - cw, m),                     # top right
              (lambda a: a[::-1, ::-1], m, m)]                           # top left
    layers['corners'] = []
    for k, (flip, x, y) in enumerate(places):
        # each corner gets its own noise and its own balance of growth vs. blotchy bleed
        mix = (0.88, 0.72, 0.8, 0.65)[k]
        fc = mix * geo + (1 - mix) * rank_noise(cv.shape, (5, 9, 7, 12)[k], 40 + k)
        layers['corners'].append(Layer(flip(cv), x, y, flip(fc), soft=(0.3, 0.42, 0.34, 0.5)[k]))

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


# ---------------------------------------------------------------- the cat
CAT_SCALE = 0.43
CAT_GROUND = 1046            # feet line on the page
CAT_STOP_X = 1480            # where it freezes (centre of the side-view sprite)


def signed_distance(mask):
    m = (mask > 0.5).astype(np.uint8)
    return cv2.distanceTransform(m, cv2.DIST_L2, 5) - cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)


def fill_holes(mask):
    m = (mask > 0.5).astype(np.uint8)
    flood = m.copy()
    cv2.floodFill(flood, np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8), (0, 0), 1)
    return ((flood == 0) | (m > 0)).astype(np.float32)


class Cat:
    """Cut-out puppet built from the two cat references.

    Side view (cat-scared.png): the silhouette is split into a body and four legs that swing
    around their hips. A calm body is made by opening away the bristled fur; the scare morphs
    the calm signed distance field into the original spiky one, so the fur visibly stands up.
    Jump (cat-jump.png): the front-facing startled pose with its ground shadow.
    """

    def __init__(self):
        s = CAT_SCALE
        side = ink_from(os.path.join(REF, 'cat-scared.png'), lo=0.2, hi=0.7)
        side = side[20:470, 60:580]                      # crop in reference pixels
        oy, ox = 20, 60
        sil = fill_holes(side)
        eyes = np.clip(sil - (side > 0.5), 0, 1)
        hip_y = 358 - oy                                 # belly line: the four legs separate below it
        # legs: ink below the belly line, one component each
        below = (side > 0.5).astype(np.uint8)
        below[:hip_y] = 0
        n, lab, st, _ = cv2.connectedComponentsWithStats(below, 8)
        legs = []
        for k in range(1, n):
            x, y, w, h, area = st[k]
            if area < 300:
                continue
            piece = np.zeros_like(sil)
            x0, x1 = x - 3, x + w + 3
            piece[hip_y - 12:hip_y, x0:x1] = sil[hip_y - 12:hip_y, x0:x1]
            own = cv2.dilate((lab == k).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            piece[hip_y:] = sil[hip_y:] * own[hip_y:]      # only this leg's own ink below the belly
            legs.append((piece, (x + w / 2, hip_y - 6)))
        legs.sort(key=lambda p: p[1][0])
        body_spiky = sil.copy()
        body_spiky[hip_y:] = 0
        # calm fur: open the back and tail, keep the head (ears, whiskers) as drawn
        opened = cv2.morphologyEx((body_spiky > 0.5).astype(np.uint8), cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))).astype(np.float32)
        opened = (cv2.GaussianBlur(opened, (0, 0), 6) > 0.5).astype(np.float32)
        head = np.zeros_like(sil)
        head[:, :170] = 1                                # reference x < 230
        body_calm = np.maximum(opened, body_spiky * head)
        # the tail's top tuft is thin; keep a smoothed stub of it so the calm cat still has a tail
        tail = np.zeros_like(sil); tail[:250, 390:] = 1
        tail_soft = cv2.morphologyEx((body_spiky * tail > 0.5).astype(np.uint8), cv2.MORPH_OPEN,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))).astype(np.float32)
        tail_soft = (cv2.GaussianBlur(tail_soft, (0, 0), 5) > 0.5).astype(np.float32)
        body_calm = np.maximum(body_calm, tail_soft)

        def sc(a):
            return cv2.resize(a, (round(a.shape[1] * s), round(a.shape[0] * s)), interpolation=cv2.INTER_AREA)

        self.sd_calm = signed_distance(sc(body_calm))
        self.sd_spiky = signed_distance(sc(body_spiky))
        self.eyes = sc(eyes)
        self.legs = [(sc(p), (hx * s, hy * s)) for p, (hx, hy) in legs]
        self.side_h, self.side_w = self.sd_calm.shape
        ys = np.where(sc(sil).max(1) > 0.5)[0]
        self.side_feet = ys.max()                        # sprite row that touches the ground

        jump = ink_from(os.path.join(REF, 'cat-jump.png'), lo=0.2, hi=0.7)
        body = jump[110:660, 70:580]
        shadow = jump[720:810, 170:530]
        jsil = fill_holes(body)
        js = 0.36 / s                                    # the jump drawing is larger than the side view

        def scj(a):
            return sc(cv2.resize(a, (round(a.shape[1] * js), round(a.shape[0] * js)), interpolation=cv2.INTER_AREA))
        self.jump = scj(jsil)
        self.jump_eyes = scj(np.clip(jsil - (body > 0.5), 0, 1))
        self.shadow = scj(shadow)
        ys = np.where(self.jump.max(1) > 0.5)[0]
        self.jump_feet = ys.max()

    # -- pose from time ------------------------------------------------------------------
    def pose(self, t):
        """Returns None when off screen, else a dict describing the pose at time t."""
        a, b = T_CAT_IN
        if t < a or t > T_CAT_RUN[1] + 0.05:
            return None
        p = dict(bristle=0.0, flip=False, jump=False, lift=0.0, sx=1.0, sy=1.0, lean=0.0)
        trot_f = 3.4                                     # strides per second
        if t < b:
            u = span(t, a, b)
            # constant trot, braking hard over the last 12%
            k = 0.88
            d = u / k * (1 - 0.06) if u < k else (1 - 0.06) + 0.06 * (1 - (1 - (u - k) / (1 - k)) ** 2)
            p['x'] = 2080 + (CAT_STOP_X - 2080) * d
            amp = 26 * (1 - smooth((u - 0.85) / 0.15))
            p['phase'] = (t - a) * trot_f
            p['amp'] = amp
        elif t < T_CAT_JUMP[0]:
            p['x'] = CAT_STOP_X
            p['phase'] = (b - a) * trot_f
            p['amp'] = 0.0
            br = smooth(span(t, *T_CAT_BRISTLE))
            p['bristle'] = br
            p['sy'] = 1 + 0.07 * br                      # arches up
            p['sx'] = 1 - 0.03 * br
            p['x'] = CAT_STOP_X + 10 * br                # a small flinch back
        elif t < T_CAT_JUMP[1]:
            u = span(t, *T_CAT_JUMP)
            p.update(jump=True, x=CAT_STOP_X + 10, bristle=1.0, phase=0, amp=0)
            p['lift'] = 150 * 4 * u * (1 - u)            # parabola
            p['sy'] = 1 + 0.06 * np.sin(np.pi * u)       # stretched while airborne
        else:
            p.update(bristle=1.0, flip=True)
            r0, r1 = T_CAT_RUN
            if t < r0:                                   # landing squash, turning to flee
                u = span(t, T_CAT_JUMP[1], r0)
                p.update(x=CAT_STOP_X + 10, phase=0, amp=0, sy=1 - 0.1 * np.sin(np.pi * u), sx=1 + 0.08 * np.sin(np.pi * u))
            else:
                u = span(t, r0, r1)
                p['x'] = CAT_STOP_X + 10 + 740 * u * u * (1.4 - 0.4 * u)   # accelerating dash
                p['phase'] = (t - r0) * 7.0
                p['amp'] = 34 * smooth(u / 0.2)
                p['sx'] = 1 + 0.14 * smooth(u / 0.3)
                p['sy'] = 1 - 0.05 * smooth(u / 0.3)
                p['lean'] = 5 * smooth(u / 0.3)
        p.setdefault('phase', 0); p.setdefault('amp', 0)
        return p

    # -- sprite ---------------------------------------------------------------------------
    def sprite(self, p):
        """Alpha, eye-white alpha and the feet row for this pose, in sprite space."""
        if p['jump']:
            return self.jump, self.jump_eyes, self.jump_feet
        sd = self.sd_calm * (1 - p['bristle']) + self.sd_spiky * p['bristle']
        alpha = np.clip(0.5 + sd, 0, 1)
        bob = abs(np.sin(np.pi * 2 * p['phase'])) * 0.18 * p['amp']
        pad = 30
        h, w = alpha.shape
        canvas = np.zeros((h + pad, w), np.float32)
        eyes = np.zeros_like(canvas)
        canvas[:h] = np.roll(alpha, -int(round(bob)), axis=0) if bob >= 0.5 else alpha
        eyes[:h] = np.roll(self.eyes, -int(round(bob)), axis=0) if bob >= 0.5 else self.eyes
        body = canvas.copy()
        for i, (leg, (hx, hy)) in enumerate(self.legs):
            # diagonal pairs move together: front-left with back-right
            ph = 2 * np.pi * p['phase'] + (0 if i in (0, 3) else np.pi)
            ang = p['amp'] * np.sin(ph)
            up = max(0.0, np.cos(ph)) * p['amp'] * 0.22     # foot lifts on the forward swing
            M = cv2.getRotationMatrix2D((hx, hy), ang, 1.0)
            M[1, 2] -= up + bob
            lp = np.zeros_like(canvas); lp[:h] = leg
            canvas = np.maximum(canvas, cv2.warpAffine(lp, M, (w, h + pad), flags=cv2.INTER_LINEAR))
        canvas = np.maximum(canvas, body)
        return canvas, eyes, self.side_feet

    def draw(self, t, rgb, A):
        p = self.pose(t)
        if p is None:
            return
        alpha, eyes, feet = self.sprite(p)
        h, w = alpha.shape
        # sprite -> page: scale about the feet, optional mirror, lean, subpixel translate
        sx = p['sx'] * (-1 if p['flip'] else 1)
        cx = w / 2
        M = np.array([[sx, 0, 0], [0, p['sy'], 0]], np.float64)
        M[0, 2] = p['x'] - sx * cx
        M[1, 2] = CAT_GROUND - p['lift'] - p['sy'] * feet
        if p['lean']:
            R = cv2.getRotationMatrix2D((p['x'], CAT_GROUND), -p['lean'] * (1 if p['flip'] else -1), 1.0)
            M = R @ np.vstack([M, [0, 0, 1]])
        a = cv2.warpAffine(alpha, M, (W, H), flags=cv2.INTER_LINEAR)
        e = cv2.warpAffine(eyes, M, (W, H), flags=cv2.INTER_LINEAR)
        # ground shadow (from the jump reference), smaller and fainter the higher the cat is
        k = 1 - p['lift'] / 220
        sh = self.shadow
        shw = (0.8 if not p['jump'] else 1.0) * k * p['sx']
        S = np.array([[abs(shw) * 0.85, 0, 0], [0, 0.55 * k, 0]], np.float64)
        S[0, 2] = p['x'] - abs(shw) * 0.85 * sh.shape[1] / 2
        S[1, 2] = CAT_GROUND + 2 - 0.55 * k * sh.shape[0] / 2
        s_a = cv2.warpAffine(sh, S, (W, H), flags=cv2.INTER_LINEAR) * (0.35 + 0.55 * k) * (1 if p['jump'] or p['lift'] > 0 else 0.6)
        a = np.maximum(a, 0)
        # clear a thin gap in the print around the cat so it reads over the ornaments
        halo = cv2.dilate(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        A *= 1 - halo
        # shadow first, then the cat
        for alpha_, col in ((s_a, INK), (a, INK), (e * a, CAT_EYE)):
            al = np.clip(alpha_, 0, 1)
            if col is CAT_EYE:
                al = np.clip(e, 0, 1) * (a > 0.01)
            newA = al + A * (1 - al)
            rgb[:] = (col * al[..., None] + rgb * (A * (1 - al))[..., None]) / np.maximum(newA, 1e-6)[..., None]
            A[:] = newA


def corner_progress(t, st, du, pw):
    return 1 - (1 - span(t, st, st + du)) ** pw              # starts growing at once, settles slowly


class Film:
    def __init__(self):
        self.lay = make_layers()
        _, self.wear, self.grain = make_paper()
        self.cat = Cat()
        A = self.lay['A']
        self.baseline = A['baseline']

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
        put(lay['L'], span(t, a, a + glyph_len + T_L_EXTRA))
        for i, g in enumerate(lay['glyphs']):
            s = a + STAGGER * (i + 1) + 0.05
            put(g, span(t, s, s + glyph_len))
        if not NO_CORNERS:
            for c, (st, du, pw) in zip(lay['corners'], T_CORNERS):
                put(c, corner_progress(t, st, du, pw))
        put(lay['rose'], ease_io(span(t, *T_ROSE)))
        put(lay['rose_red'], ease_io(span(t, T_ROSE[0] + 0.3, T_ROSE[1] + 0.35)))

        # breathing grain: three fixed textures cross-faded slowly (no per-frame noise crawl)
        ph = t * 0.6
        wts = [0.5 + 0.5 * np.cos(2 * np.pi * (ph - j / 3)) for j in range(3)]
        gr = sum(w_ * g_ for w_, g_ in zip(wts, self.grain)) / sum(wts)
        dens = self.wear * (1 + 0.05 * gr)
        a_ink = np.clip(ink * dens, 0, 1)
        a_red = np.clip(red * dens * 0.9, 0, 1)

        ink_rgb = np.broadcast_to(PRINT, (H, W, 3))

        # straight-alpha RGBA: ink printed over the red plate
        A = 1 - (1 - a_ink) * (1 - a_red)
        rgb = (RED * (a_red * (1 - a_ink))[..., None] + ink_rgb * a_ink[..., None]) / np.maximum(A, 1e-6)[..., None]
        # the cat is a solid cut-out on top of the print
        self.cat.draw(t, rgb, A)
        out = np.dstack([np.clip(rgb, 0, 1), A])
        return (out * 255 + 0.5).astype(np.uint8)


def over(rgba, bg):
    """Composite an RGBA frame onto a solid colour (for previews and sheets)."""
    a = rgba[..., 3:4].astype(np.float32) / 255
    bg = np.array(bg, np.float32)
    return (rgba[..., :3] * a + bg * (1 - a) + 0.5).astype(np.uint8)


PREVIEW_BG = (236, 228, 212)


def sheet(film, times, path, cols=4):
    tiles = []
    for t in times:
        f = cv2.resize(over(film.frame(t), PREVIEW_BG), (480, 270), interpolation=cv2.INTER_AREA)
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
    elif cmd == 'frames':
        # RGBA PNG per frame (for the site: → AVIF)
        out = sys.argv[2]
        fps = int(sys.argv[sys.argv.index('--fps') + 1]) if '--fps' in sys.argv else 60
        os.makedirs(out, exist_ok=True)
        for i in range(int(round(DURATION * fps))):
            Image.fromarray(film.frame(i / fps)).save(os.path.join(out, f'f_{i:04d}.png'), compress_level=1)
    elif cmd == 'corners':
        # One sprite sheet per corner: CORNER_STEPS growth stages (progress 0..1, even steps) in a
        # CORNER_COLS-wide grid, each cell the corner's own box, RGBA in the print colour. Order and
        # timing follow T_CORNERS: bottom right, bottom left, top right, top left. With the page's ink
        # wear taken where the corner sits on the 1920×1080 page.
        out = sys.argv[2]
        os.makedirs(out, exist_ok=True)
        names = ['bottom-right', 'bottom-left', 'top-right', 'top-left']
        for k, c in enumerate(film.lay['corners']):
            ch, cw = c.ink.shape
            wear = film.wear[c.y:c.y + ch, c.x:c.x + cw]
            rows = (CORNER_STEPS + CORNER_COLS - 1) // CORNER_COLS
            sheet = np.zeros((rows * ch, CORNER_COLS * cw, 4), np.float32)
            for j in range(CORNER_STEPS):
                cov = c.coverage(j / (CORNER_STEPS - 1))
                a = np.zeros((ch, cw), np.float32) if cov is None else np.clip(cov * wear, 0, 1)
                r, q = divmod(j, CORNER_COLS)
                cell = sheet[r * ch:(r + 1) * ch, q * cw:(q + 1) * cw]
                cell[..., :3] = PRINT; cell[..., 3] = a
            Image.fromarray((sheet * 255 + 0.5).astype(np.uint8)).save(os.path.join(out, f'corner-{names[k]}.png'))
        print('corner cell', cw, 'x', ch, 'page margin', film.lay['corners'][3].x, 'steps', CORNER_STEPS, 'cols', CORNER_COLS)
    elif cmd == 'render':
        # one pass feeds three encoders: ProRes 4444 and VP9 keep the alpha, the MP4 is a preview on paper
        stem = sys.argv[2]
        fps = int(sys.argv[sys.argv.index('--fps') + 1]) if '--fps' in sys.argv else 60
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        src = [ff, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-s', f'{W}x{H}', '-r', str(fps)]
        jobs = [
            (src + ['-pix_fmt', 'rgba', '-i', '-', '-c:v', 'prores_ks', '-profile:v', '4444',
                    '-pix_fmt', 'yuva444p10le', '-alpha_bits', '16', '-vendor', 'apl0', stem + '.mov'], True),
            (src + ['-pix_fmt', 'rgba', '-i', '-', '-c:v', 'libvpx-vp9', '-pix_fmt', 'yuva420p',
                    '-b:v', '0', '-crf', '18', '-row-mt', '1', '-auto-alt-ref', '0', stem + '.webm'], True),
            (src + ['-pix_fmt', 'rgb24', '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '14',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', stem + '-preview.mp4'], False),
        ]
        procs = [(subprocess.Popen(cmd_, stdin=subprocess.PIPE), alpha) for cmd_, alpha in jobs]
        n = int(round(DURATION * fps))
        for i in range(n):
            f = film.frame(i / fps)
            prev = over(f, PREVIEW_BG)
            for p, alpha in procs:
                p.stdin.write((f if alpha else prev).tobytes())
        for p, _ in procs:
            p.stdin.close(); p.wait()
        print('wrote', stem + '.{mov,webm,-preview.mp4}', n, 'frames')

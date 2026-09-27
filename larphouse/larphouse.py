"""Larphouse — 6.5 s gothic title animation: the word prints itself, a rapier slashes through it,
the cut bleeds and roses grow out of the wound.

Every frame is a pure function of time: frame(t) never depends on an earlier frame,
and all noise is seeded once at build time, so any frame can be rendered alone.

  python3 larphouse.py still 4.9 out/final.png
  python3 larphouse.py sheet 0.5:3.0:0.25 out/sheet.png
  python3 larphouse.py render larphouse [--fps 60]    # larphouse.mov, .webm, -preview.mp4
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
DURATION = 6.5

PAPER = np.array([238, 229, 211], np.float32) / 255
INK = np.array([33, 27, 30], np.float32) / 255      # warm overprint dark, never pure black
RED = np.array([150, 28, 38], np.float32) / 255
CAT_EYE = np.array([240, 234, 222], np.float32) / 255
BLOOD = np.array([118, 12, 22], np.float32) / 255
SHOW_CAT = False             # the cat sits this version out
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
T_LETTERS = (0.45, 2.9)      # the whole word, from the first drop to the last letter full
# The capital: the ink spreads from the middle of its stem along the stem and out into the scrolls
# over T_L_FLOW; the lily on top fills last and catches a glint.
T_L_FLOW = (0.45, 2.3)
# All letters start together at T_LETTERS[0]; each lowercase letter fills at its own pace
# (flow duration in seconds), all done by T_LETTERS[1].
T_GLYPHS = [(0.45, 1.9), (0.45, 2.3), (0.45, 1.75), (0.45, 2.45), (0.45, 2.05), (0.45, 2.4),
            (0.45, 1.85), (0.45, 2.2)]
# corner vines: (start, duration, ease-out power) per corner — deliberately unequal so they
# don't grow in lockstep. Order: bottom right, bottom left, top right, top left.
T_CORNERS = [(0.25, 4.0, 1.3), (0.6, 3.5, 1.7), (0.0, 3.7, 1.5), (0.45, 4.3, 1.9)]
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


class FlowLayer(Layer):
    """Ink poured into a letter: it spreads from where the drop lands along the strokes (geodesic
    distance through the letter), filling the middle of each stroke before its edges and the
    thick strokes before the thin ones. A wet, slightly spread rim runs just ahead of the front."""

    def __init__(self, ink, x, y, field, soft=0.07, wet=0.10):
        super().__init__(ink, x, y, field, soft=soft, stain=0.6, blur=2.2)
        self.wet = wet

    def coverage(self, p):
        if p <= 0:
            return None
        f = p * (1 + self.soft + self.wet)
        q = np.clip((f - self.field) / self.soft, 0, 1)
        q = q * q * (3 - 2 * q)
        body = self.ink * q
        # the wet front: a band just behind and ahead of it, a touch spread beyond the stroke edge;
        # once the whole letter is full it dries back into the clean print
        d = f - self.field                               # > 0: already reached
        band = np.where(d > 0, np.exp(-(d / self.wet) ** 2), np.exp(-(d / (self.wet * 0.25)) ** 2))
        band = band * (1 - smooth((p - 0.9) / 0.1))
        return np.maximum(body, self.stain * band * 1.25)


def flow_field(mask, seeds, late=None, step=2):
    """0 where the drop lands ... 1 last: mostly distance travelled along the letter, a little of
    'edges after middles' (so the ink runs down the stroke channels) and an optional extra delay."""
    geo = geodesic(mask > 0.3, seeds, step=step)
    dt = cv2.distanceTransform((mask > 0.3).astype(np.uint8), cv2.DIST_L2, 5)
    dt = cv2.GaussianBlur(dt, (0, 0), 1.5)
    edge = 1 - dt / (dt.max() + 1e-6)
    f = 0.8 * geo + 0.2 * edge
    if late is not None:
        f = f + late
    # even the pace out: rank the letter's own pixels so the front covers equal ink per unit time
    m = mask > 0.05
    r = np.zeros_like(f)
    v = f[m]
    r[m] = v.argsort().argsort().astype(np.float32) / max(1, v.size - 1)
    r[~m] = 1.0
    return r.astype(np.float32)


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
    # the capital: the drop lands in the middle of the stem; the ink runs up and down the stem and
    # out along every scroll to its tip; the lily on top is held back so it fills last
    drop = (h * 0.52, w * 0.36)
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
    lily = (1 - smooth(yy / 0.17)) * 0.55
    fL = flow_field(L, [drop], late=lily)
    layers['L'] = FlowLayer(L, lx, ly, fL, soft=0.06, wet=0.09)
    layers['L_drop'] = (lx + drop[1], ly + drop[0])
    ys_, xs_ = np.nonzero(L > 0.5)
    top = ys_.min()
    layers['L_lily'] = (lx + float(xs_[ys_ < top + 6].mean()), ly + top + 34.0)   # the lily's heart

    rng = np.random.default_rng(55)
    layers['glyphs'] = []
    for i, (g, gx, gy) in enumerate(A['glyphs']):
        gh, gw = g.shape
        # each letter's drop lands on one of its thickest points, a different one each time
        dt = cv2.distanceTransform((g > 0.3).astype(np.uint8), cv2.DIST_L2, 5)
        cand = np.argwhere(dt > dt.max() * 0.7)
        pick = cand[int(rng.integers(len(cand)))]
        f = flow_field(g, [(float(pick[0]), float(pick[1]))])
        f = 0.9 * f + 0.1 * rank_noise(g.shape, 3, 30 + i)
        f = (f - f.min()) / (f.max() - f.min())
        layers['glyphs'].append(FlowLayer(g, gx, gy, f, soft=float(rng.uniform(0.05, 0.1)),
                                          wet=float(rng.uniform(0.07, 0.13))))

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


# ---------------------------------------------------------------- the slash
# A rapier (one of the crossed swords) appears above the end of the word, winds up, dives onto the
# cut line and runs through the whole word tip-first, right to left. The word splits along a
# slightly jagged line: the upper half slides down-left along the cut and lifts a little. Both cut
# edges bleed (a red rim, beads, drips down the strokes), and from the wound thorned stems grow
# along the cut and the beads open into roses, right to left, following the blow.
CUT_C = (945.0, 612.0)           # a point on the cut (page px)
CUT_ANG = np.radians(5.0)        # the cut rises to the right
CUT_U = np.array([np.cos(CUT_ANG), -np.sin(CUT_ANG)])      # along the cut, to the right
CUT_N = np.array([-np.sin(CUT_ANG), -np.cos(CUT_ANG)])     # normal, pointing up
SLIDE = -17.0 * CUT_U + 3.2 * CUT_N                         # where the upper half ends up
GAP = 2.2                        # px of clean gap at the cut

T_SWORD_IN = (2.45, 2.95)        # the rapier bleeds in above the end of the word
T_WINDUP = (2.95, 3.10)
T_SWOOP = (3.10, 3.17)           # dives onto the cut line
T_CUT = (3.17, 3.405)            # tip runs along the cut from X_ENTER to X_EXIT (hilt fully off screen)
X_ENTER, X_EXIT = 1760.0, -760.0
T_SHAKE = 0.34                   # how long the frame shakes after the blade enters
SWORD_LEN = 680
ROSE_T0, ROSE_STEP, ROSE_OPEN = 4.15, 0.3, 0.7


def cut_y(x):
    return CUT_C[1] - (x - CUT_C[0]) * np.tan(CUT_ANG)


def t_pass(x):
    """When the tip passes page x on the cut."""
    return T_CUT[0] + (X_ENTER - x) / (X_ENTER - X_EXIT) * (T_CUT[1] - T_CUT[0])


def ease_out_back(x, k=1.4):
    x = np.clip(x, 0, 1)
    return 1 + (k + 1) * (x - 1) ** 3 + k * (x - 1) ** 2


def bezier(P, u):
    P = np.asarray(P, np.float64)
    return (1 - u) ** 3 * P[0] + 3 * (1 - u) ** 2 * u * P[1] + 3 * (1 - u) * u * u * P[2] + u ** 3 * P[3]


def rapier_sprite():
    """The rapier whose point is top right in crossed-swords.png, laid horizontal, point to +x.
    Returns ink (h, w) and the pixel column of the point."""
    sw = ink_from(os.path.join(REF, 'crossed-swords.png'), lo=0.15, hi=0.8)
    h, w = sw.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    tip, pom = np.array([845., 18.]), np.array([55., 875.])
    d = pom - tip
    Ln = np.linalg.norm(d)
    u = d / Ln
    n = np.array([-u[1], u[0]])
    s = ((xx - tip[0]) * u[0] + (yy - tip[1]) * u[1]) / Ln
    q = (xx - tip[0]) * n[0] + (yy - tip[1]) * n[1]
    halfw = np.where(s < 0.6, 10 + 8 * s, 175)                      # narrow on the blade: no crossing stub
    one = sw * ((np.abs(q) < halfw) & (s > -0.02) & (s < 1.05))
    # rotate so the point is to the right, on a canvas big enough for any angle
    ang = np.degrees(np.arctan2(-(tip - pom)[1], (tip - pom)[0]))   # point direction, screen up = +
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -ang, 1.0)
    big = int(np.hypot(h, w))
    M[0, 2] += big / 2 - w / 2; M[1, 2] += big / 2 - h / 2
    r = cv2.warpAffine(one.astype(np.float32), M, (big, big), flags=cv2.INTER_CUBIC)
    r = crop(np.clip(r, 0, 1), pad=4)
    r = fit(r, width=SWORD_LEN)
    return r


def rose_sprite():
    """The front-facing rose from rose-sword.png: ink outline, red petal fill, silhouette."""
    rs = ink_from(os.path.join(REF, 'rose-sword.png'), lo=0.1, hi=0.78)
    closed = cv2.morphologyEx((rs > 0.35).astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    flood = closed.copy()
    cv2.floodFill(flood, np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8), (0, 0), 1)
    enclosed = ((flood == 0) | (closed > 0)) & (rs < 0.35)
    bx, by, br = 178, 725, 80
    yy, xx = np.mgrid[0:rs.shape[0], 0:rs.shape[1]]
    inside = enclosed & (np.hypot(xx - bx, yy - by) < br)
    sil = cv2.dilate(inside.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)
    sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    r = int(br * 1.2)
    sl = np.s_[by - r:by + r, bx - r:bx + r]
    ink = (rs * sil)[sl]
    red = (cv2.GaussianBlur(inside.astype(np.float32), (0, 0), 1.0) * (1 - rs))[sl]
    sil = cv2.GaussianBlur(sil[sl], (0, 0), 1.0)
    return ink.astype(np.float32), red.astype(np.float32), sil.astype(np.float32)


class Slash:
    def __init__(self, film):
        self.film = film
        A = film.lay['A']
        x0, y0, x1, y1 = A['word_box']
        self.box = (int(x0) - 60, int(y0) - 40, int(x1) + 60, int(y1) + 60)
        bx0, by0, bx1, by1 = self.box
        yy, xx = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
        self.xx, self.yy = xx, yy
        # the cut: signed distance to a slightly jagged line (positive = above)
        s = (xx - CUT_C[0]) * CUT_U[0] + (yy - CUT_C[1]) * CUT_U[1]
        d = (xx - CUT_C[0]) * CUT_N[0] + (yy - CUT_C[1]) * CUT_N[1]
        rng = np.random.default_rng(71)
        jag1 = np.interp(s, np.arange(-1200, 1200, 7), rng.normal(0, 0.9, len(range(-1200, 1200, 7))))
        jag2 = np.interp(s, np.arange(-1200, 1200, 37), rng.normal(0, 1.4, len(range(-1200, 1200, 37))))
        self.d = d + jag1 + jag2
        self.s = s
        self.sword = rapier_sprite()
        self.sword_field = rank_noise(self.sword.shape, 5, 90)
        self.sword_stain = cv2.GaussianBlur(self.sword, (0, 0), 6) * 0.5
        self.rose_ink, self.rose_red, self.rose_sil = rose_sprite()
        self.title_final = None                                         # filled on first use
        self.rng = np.random.default_rng(72)

    # -- sites along the cut (from the finished letters) ---------------------------------
    def sites(self, title):
        """Where the cut crosses ink: bead sites for the blood, rose sites among them."""
        bx0, by0, bx1, by1 = self.box
        xs = np.arange(bx0 + 10, bx1 - 10, 1.0)
        ys = cut_y(xs)
        on = title[np.clip(ys.astype(int), 0, H - 1), xs.astype(int)] > 0.5
        runs = []
        start = None
        for x, o in zip(xs, on):
            if o and start is None:
                start = x
            if not o and start is not None:
                if x - start > 6:
                    runs.append((start, x))
                start = None
        rng = np.random.default_rng(73)
        beads = []
        for a, b in runs:
            n = 1 + int((b - a) // 60)
            for k in range(n):
                x = a + (b - a) * (k + 0.5) / n + rng.uniform(-0.2, 0.2) * (b - a) / n
                beads.append(dict(x=float(x), r=float(rng.uniform(3.5, 7.0)),
                                  top=bool(rng.random() < 0.45), drip=float(rng.uniform(35, 120)) if rng.random() < 0.7 else 0.0,
                                  delay=float(rng.uniform(0.15, 0.5))))
        # roses: spread along the word, at bead sites
        xs_b = np.array([b['x'] for b in beads])
        targets = np.array([1575.0, 1205.0, 830.0, 430.0])
        roses = []
        used = set()
        for k, tx in enumerate(targets):
            i = int(np.argmin(np.abs(xs_b - tx) + [1e9 if j in used else 0 for j in range(len(xs_b))]))
            used.add(i)
            roses.append(dict(x=float(xs_b[i]), size=float((0.7, 0.6, 0.76, 0.66)[k]),
                              rot=float(rng.uniform(-40, 40)), flip=bool(k % 2),
                              dx=float((34, -30, 40, -36)[k]), off=float((52, -50, 58, -56)[k]),
                              t0=ROSE_T0 + k * ROSE_STEP))
        self.beads, self.roses = beads, roses

    # -- the rapier ----------------------------------------------------------------------------
    def tip_pose(self, t):
        """Point position and pointing direction (unit) of the rapier, or None when not shown."""
        hover_tip = np.array([1700.0, 372.0])
        hover_dir = np.array([-0.46, 0.89])
        wind_tip = np.array([1765.0, 296.0])
        wind_dir = np.array([-0.28, 0.96])
        if t < T_SWORD_IN[0] or t > T_CUT[1] + 0.02:
            return None
        if t < T_WINDUP[0]:
            bob = 4 * np.sin(2 * np.pi * (t - T_SWORD_IN[0]) / 0.9)
            return hover_tip + np.array([0, bob]), hover_dir
        if t < T_SWOOP[0]:
            u = ease_io(span(t, *T_WINDUP))
            dvec = hover_dir + (wind_dir - hover_dir) * u
            return hover_tip + (wind_tip - hover_tip) * u, dvec / np.linalg.norm(dvec)
        if t < T_CUT[0]:
            u = span(t, *T_SWOOP) ** 1.6
            P = [wind_tip, (1795, 470), (1880, cut_y(1880)), (X_ENTER, cut_y(X_ENTER))]
            p = bezier(P, u)
            q = bezier(P, min(1.0, u + 0.02)) - bezier(P, max(0.0, u - 0.02))
            return p, q / (np.linalg.norm(q) + 1e-9)
        u = span(t, *T_CUT)
        x = X_ENTER + (X_EXIT - X_ENTER) * u
        return np.array([x, cut_y(x)]), -CUT_U

    def draw_sword(self, t, fps, ink):
        """Adds the rapier's coverage to `ink` (full page). Motion blur: many sub-frames across
        the frame's shutter, each warped only into the box the blade occupies."""
        fast = T_SWOOP[0] <= t <= T_CUT[1] + 0.02
        n = 48 if fast else 1
        sh, sw_ = self.sword.shape
        if t < T_WINDUP[0]:
            p = span(t, *T_SWORD_IN)
            q = np.clip((p * 1.4 - self.sword_field) / 0.4, 0, 1)
            spr = (self.sword_stain * (1 - q) + self.sword * q) * np.clip(p * 2.5, 0, 1)
        else:
            spr = self.sword
        acc = np.zeros((H, W), np.float32)
        corners = np.array([[0, 0, 1], [sw_, 0, 1], [0, sh, 1], [sw_, sh, 1]], np.float64)
        drawn = 0
        for k in range(n):
            ts = t + ((k + 0.5) / n - 0.5) / fps if n > 1 else t
            pose = self.tip_pose(ts)
            if pose is None:
                continue
            tip, dvec = pose
            ang = np.degrees(np.arctan2(-dvec[1], dvec[0]))
            M = cv2.getRotationMatrix2D((sw_ - 1.0, sh / 2), ang, 1.0)
            M[0, 2] += tip[0] - (sw_ - 1.0)
            M[1, 2] += tip[1] - sh / 2
            pc = corners @ M.T
            x0, y0 = int(max(0, np.floor(pc[:, 0].min()) - 2)), int(max(0, np.floor(pc[:, 1].min()) - 2))
            x1, y1 = int(min(W, np.ceil(pc[:, 0].max()) + 2)), int(min(H, np.ceil(pc[:, 1].max()) + 2))
            drawn += 1
            if x1 <= x0 or y1 <= y0:
                continue
            M[0, 2] -= x0; M[1, 2] -= y0
            acc[y0:y1, x0:x1] += cv2.warpAffine(spr, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
        if drawn:
            np.maximum(ink, np.clip(acc / n, 0, 1), out=ink)

    # -- the cut, the blood ------------------------------------------------------------------
    def sep(self, t):
        """Separation 0..1(+overshoot) per box column, following the blade."""
        tp = t_pass(self.xx[0])
        return ease_out_back((t - tp - 0.02) / 0.38, 1.2)

    def split(self, title, t):
        """Returns the title with its upper half slid along the cut (full page array), plus the
        per-pixel displacement used, so the blood on the upper edge can ride along."""
        bx0, by0, bx1, by1 = self.box
        sub = title[by0:by1, bx0:bx1]
        passed = (t - t_pass(self.xx) > 0).astype(np.float32)
        sp = self.sep(t)[None, :] * passed
        dx, dy = SLIDE[0] * sp, SLIDE[1] * sp
        mx, my = self.xx - dx - bx0, self.yy - dy - by0
        moved = cv2.remap(sub, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR)
        d_src = cv2.remap(self.d, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
        # the gap opens where the blade has been: a clean slice, anti-aliased
        g = GAP * passed
        top = moved * np.clip(d_src - g / 2 + 0.5, 0, 1)
        bot = sub * np.clip(-self.d - g / 2 + 0.5, 0, 1)
        out = title.copy()
        out[by0:by1, bx0:bx1] = np.maximum(top, bot) if t > T_CUT[0] else sub
        return out, (mx, my)

    def blood(self, title_static, t, maps):
        """Blood coverage (full page). Built in each half's own coordinates, the upper half's then
        carried by the same slide as the letters."""
        bx0, by0, bx1, by1 = self.box
        sub = title_static[by0:by1, bx0:bx1]
        tp = t_pass(self.xx)
        grow = smooth((t - tp - 0.12) / 0.45)
        d = self.d
        # rim: along both edges, inside the letters
        w = 0.5 + 5.0 * grow
        rim_top = sub * (d > 0) * np.exp(-np.clip(d, 0, None) / np.maximum(w, 1e-3)) * (grow > 0)
        rim_bot = sub * (d < 0) * np.exp(-np.clip(-d, 0, None) / np.maximum(w, 1e-3)) * (grow > 0)
        top = rim_top * 0.95
        bot = rim_bot * 0.95
        for b in self.beads:
            tb = t_pass(b['x']) + b['delay']
            g = smooth((t - tb) / 0.35)
            if g <= 0:
                continue
            x = b['x']
            yl = cut_y(x)
            r = b['r'] * (0.3 + 0.7 * g)
            if b['top']:
                # a drop hanging from the upper edge, swelling
                cy = yl - 1.5 + r * 0.9
                blob = np.clip(r - np.hypot((self.xx - x) / 0.85, (self.yy - cy) / 1.3) + 0.5, 0, 1)
                top = np.maximum(top, blob * (np.abs(self.xx - x) < r + 3))
            else:
                cy = yl + 1.0 + r * 0.6
                blob = np.clip(r - np.hypot(self.xx - x, self.yy - cy) + 0.5, 0, 1)
                bot = np.maximum(bot, blob * (sub > 0.3))
                if b['drip'] > 0:
                    L_ = b['drip'] * smooth((t - tb - 0.25) / 1.3)
                    if L_ > 0.5:
                        wd = 1.6 + 1.3 * b['r'] / 5
                        col = np.clip(wd - np.abs(self.xx - x) + 0.5, 0, 1)
                        run = col * (self.yy > cy) * (self.yy < cy + L_)
                        end = np.clip(wd * 1.35 - np.hypot(self.xx - x, self.yy - (cy + L_)) + 0.5, 0, 1)
                        bot = np.maximum(bot, np.maximum(run, end) * (sub > 0.3))
        mx, my = maps
        top_moved = cv2.remap(top.astype(np.float32), mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR)
        out = np.zeros((H, W), np.float32)
        out[by0:by1, bx0:bx1] = np.maximum(top_moved, bot)
        return out

    # -- stems and roses -----------------------------------------------------------------------
    def stems(self, t, deco, halo):
        """For each rose: a thorned stem that pushes out of the gap and curls to where the rose
        opens, and two short tendrils running along the cut. Drawn into `deco` (uint8) with a
        clearing `halo` (uint8) so they read over the letters."""
        S = 16

        def draw_path(pts, grow, thick, seed, leaves=True):
            n = int(len(pts) * grow)
            if n < 2:
                return
            pts = pts[:n]
            for j in range(0, n - 1, 3):
                seg = pts[j:min(n, j + 4)]
                taper = min(1.0, (n - j) / 18)
                th = max(1, int(round(thick * taper)))
                P = np.round(seg * S).astype(np.int32)
                cv2.polylines(halo, [P], False, 255, th + 4, cv2.LINE_AA, 4)
                cv2.polylines(deco, [P], False, 255, th, cv2.LINE_AA, 4)
            for j in range(9, n - 4, 11):
                p0 = pts[j]
                tang = pts[j + 1] - pts[j - 1]
                tang = tang / (np.linalg.norm(tang) + 1e-9)
                side = 1 if (j // 11 + seed) % 2 else -1
                nrm = np.array([-tang[1], tang[0]]) * side
                sz = 5.5 * min(1.0, (n - j) / 20)
                if sz < 1:
                    continue
                tri = np.array([p0 - tang * sz * 0.45, p0 + tang * sz * 0.45, p0 + nrm * sz * 1.15 - tang * sz * 0.8])
                P = np.round(tri * S).astype(np.int32)
                cv2.fillPoly(halo, [P], 255, cv2.LINE_AA, 4)
                cv2.fillPoly(deco, [P], 255, cv2.LINE_AA, 4)
                if leaves and (j // 11 + seed) % 3 == 1:
                    ls = 13 * min(1.0, (n - j) / 26)
                    if ls > 2:
                        base = p0 + nrm * 2
                        tipp = base + (nrm * 0.85 - tang * 0.5) * ls * 1.7
                        mid = (base + tipp) / 2
                        sd = np.array([-(tipp - base)[1], (tipp - base)[0]])
                        sd = sd / (np.linalg.norm(sd) + 1e-9) * ls * 0.42
                        curve = np.array([bezier([base, mid + sd, mid + sd, tipp], u) for u in np.linspace(0, 1, 8)] +
                                         [bezier([tipp, mid - sd, mid - sd, base], u) for u in np.linspace(0, 1, 8)])
                        P = np.round(curve * S).astype(np.int32)
                        cv2.fillPoly(halo, [P], 255, cv2.LINE_AA, 4)
                        cv2.fillPoly(deco, [P], 255, cv2.LINE_AA, 4)
                        cv2.line(deco, tuple(np.round(base * S).astype(int)), tuple(np.round(tipp * S).astype(int)),
                                 0, 1, cv2.LINE_AA, 4)                    # the midrib, in paper

        for k, rz in enumerate(self.roses):
            x = rz['x']
            base = np.array([x, cut_y(x)])
            end = np.array([x + rz['dx'], cut_y(x) - rz['off']])
            up = -1 if rz['off'] > 0 else 1                  # screen y direction toward the rose
            c1 = base + np.array([-rz['dx'] * 0.6, up * 30])
            c2 = end + np.array([rz['dx'] * 0.5, -up * 26])
            stem = np.array([bezier([base, c1, c2, end], u) for u in np.linspace(0, 1, 60)])
            g = smooth((t - (rz['t0'] - 0.5)) / 0.55)
            if g > 0:
                draw_path(stem, g, 3, k)
            # tendrils along the cut, both ways, a little after the stem
            for dirn, L_, sd in ((1, 95 + 25 * (k % 2), 2), (-1, 120 - 20 * (k % 2), 3)):
                g2 = smooth((t - (rz['t0'] - 0.3)) / 0.8)
                if g2 <= 0:
                    continue
                xs = x + dirn * np.arange(0, L_, 2.0)
                off = 5 * np.sin((xs - x) / 17 + k + sd) * np.minimum(1, np.abs(xs - x) / 25)
                pts = np.stack([xs + CUT_N[0] * off, cut_y(xs) + CUT_N[1] * off], 1)
                draw_path(pts, g2, 2, k + sd, leaves=True)

    def draw_roses(self, t, deco, red, halo):
        for rz in self.roses:
            u = (t - rz['t0']) / ROSE_OPEN
            if u <= 0:
                continue
            x = rz['x'] + rz['dx']
            y = cut_y(rz['x']) - rz['off']            # off > 0: above the cut
            # a red bud swells first; then the rose opens out of it with a little twist
            bud = smooth(u / 0.3) * (1 - smooth((u - 0.45) / 0.3))
            if bud > 0:
                rb = 3 + 8 * smooth(u / 0.3)
                yy_, xx_ = np.ogrid[int(y - 20):int(y + 21), int(x - 20):int(x + 21)]
                disc = np.clip(rb - np.hypot(xx_ - x, (yy_ - y) / 1.15) + 0.5, 0, 1) * bud
                sl_ = np.s_[int(y - 20):int(y + 21), int(x - 20):int(x + 21)]
                np.maximum(red[sl_], disc, out=red[sl_])
                np.maximum(halo[sl_], np.clip(disc * 1.5, 0, 1), out=halo[sl_])
                ring = np.clip(1.2 - np.abs(np.hypot(xx_ - x, (yy_ - y) / 1.15) - rb), 0, 1) * bud
                np.maximum(deco[sl_], ring, out=deco[sl_])
            v = (u - 0.25) / 0.75
            if v <= 0:
                continue
            open_ = ease_out_back(v, 1.1)
            sc = rz['size'] * (0.3 + 0.7 * open_)
            rot = rz['rot'] + 55 * (1 - smooth(v))
            ink, rd, sil = self.rose_ink, self.rose_red, self.rose_sil
            if rz['flip']:
                ink, rd, sil = ink[:, ::-1], rd[:, ::-1], sil[:, ::-1]
            h, w = ink.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, sc)
            M[0, 2] += x - w / 2
            M[1, 2] += y - h / 2
            R = int(max(h, w) * sc) + 8
            x0, y0 = int(x) - R, int(y) - R
            M[0, 2] -= x0; M[1, 2] -= y0
            reveal = smooth(v * 4)
            ci = cv2.warpAffine(ink, M, (2 * R, 2 * R), flags=cv2.INTER_LINEAR) * reveal
            cr = cv2.warpAffine(rd, M, (2 * R, 2 * R), flags=cv2.INTER_LINEAR) * smooth(v * 3)
            cs = cv2.warpAffine(sil, M, (2 * R, 2 * R), flags=cv2.INTER_LINEAR) * reveal
            hal = cv2.dilate(cs, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            ys, xs_ = slice(max(0, y0), min(H, y0 + 2 * R)), slice(max(0, x0), min(W, x0 + 2 * R))
            sy, sx = slice(ys.start - y0, ys.stop - y0), slice(xs_.start - x0, xs_.stop - x0)
            np.maximum(deco[ys, xs_], ci[sy, sx], out=deco[ys, xs_])
            np.maximum(red[ys, xs_], cr[sy, sx], out=red[ys, xs_])
            np.maximum(halo[ys, xs_], hal[sy, sx], out=halo[ys, xs_])
            # the rose's own paper (between its lines) hides what is under it
            np.maximum(halo[ys, xs_], cs[sy, sx], out=halo[ys, xs_])

    def trail(self, t):
        """The blow's trace along the cut: a dark swoosh that thins to a sharp point at the blade
        and a bright slit at its core, both dying away within a quarter second. Returns
        (smear coverage, clearing) for the full page, or None."""
        if t < T_CUT[0] or t > T_CUT[1] + 0.35:
            return None
        if not hasattr(self, 'full_d'):
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            self.full_d = (xx - CUT_C[0]) * CUT_N[0] + (yy - CUT_C[1]) * CUT_N[1]
            self.full_x = xx
            self.band = np.abs(self.full_d) < 70
        age = t - t_pass(self.full_x)
        live = (age > 0) & self.band
        w = 34.0 * np.clip(age / 0.05, 0, 1) ** 0.7 * np.exp(-np.clip(age, 0, None) / 0.14)
        k = np.exp(-np.clip(age, 0, None) / 0.11)
        smear = np.clip(1 - np.abs(self.full_d + 0.35 * w) / np.maximum(w, 1e-3), 0, 1) ** 1.2 * k * 0.8 * live
        core = np.clip(1.0 + 0.16 * w - np.abs(self.full_d), 0, 1) * np.clip(k * 1.5, 0, 1) * live
        return smear.astype(np.float32), core.astype(np.float32)

    def shake(self, t):
        u = (t - T_CUT[0]) / T_SHAKE
        if u < 0 or u > 1:
            return 0.0, 0.0
        a = 7.0 * (1 - u) ** 2
        return a * np.sin(2 * np.pi * 23 * u), a * 0.6 * np.cos(2 * np.pi * 17 * u + 0.7)

    def chips(self, t, ink):
        """Flecks of ink knocked out of the cut, flying the way of the blow and falling (uint8)."""
        rng = np.random.default_rng(74)
        for k in range(26):
            x0 = rng.uniform(250, 1700)
            tp = t_pass(x0)
            u = t - tp
            if u < 0 or u > 0.9:
                continue
            vx = -rng.uniform(250, 700)
            vy = rng.uniform(-260, 60)
            px = x0 + vx * u
            py = cut_y(x0) + vy * u + 900 * u * u
            r = rng.uniform(1.2, 3.2) * (1 - u / 0.9)
            if r < 0.4:
                continue
            cv2.circle(ink, (int(px * 16), int(py * 16)), int(r * 16), 255, -1, cv2.LINE_AA, 4)


def corner_progress(t, st, du, pw):
    return 1 - (1 - span(t, st, st + du)) ** pw              # starts growing at once, settles slowly


class Film:
    def __init__(self):
        self.lay = make_layers()
        _, self.wear, self.grain = make_paper()
        self.cat = Cat() if SHOW_CAT else None
        A = self.lay['A']
        self.baseline = A['baseline']
        self.slash = Slash(self)
        self.slash.sites(self.title(DURATION))

    def glint(self, t):
        """A small four-point star catching the lily of the L as it fills (a clearing, so it reads
        bright on a light page and dark on a dark one). Returns (x0, y0, patch) or None."""
        tg = T_L_FLOW[1] - 0.12
        u = (t - tg) / 0.4
        if u <= 0 or u >= 1:
            return None
        k = np.sin(np.pi * u) ** 1.5
        x, y = self.lay['L_lily']
        R = 34
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
        ang = 0.6 * u
        c, s_ = np.cos(ang), np.sin(ang)
        a_, b_ = xx * c + yy * s_, -xx * s_ + yy * c
        star = np.exp(-np.abs(a_) / 1.3) * np.exp(-np.abs(b_) / (R * 0.45)) + \
               np.exp(-np.abs(b_) / 1.3) * np.exp(-np.abs(a_) / (R * 0.45))
        star = np.clip(star * 1.2 + np.exp(-(xx ** 2 + yy ** 2) / 18), 0, 1) * k
        return int(x) - R, int(y) - R, star

    def title(self, t):
        """Coverage of the printed word (the capital and the lowercase) at time t."""
        out = np.zeros((H, W), np.float32)
        lay = self.lay
        parts = [(lay['L'], ease_io(span(t, *T_L_FLOW)) ** 0.9)]
        for g, (t0, du) in zip(lay['glyphs'], T_GLYPHS):
            parts.append((g, ease_io(span(t, t0, t0 + du))))
        for layer, p in parts:
            c = layer.coverage(p)
            if c is None:
                continue
            h, w = c.shape
            sl = out[layer.y:layer.y + h, layer.x:layer.x + w]
            np.maximum(sl, c[:sl.shape[0], :sl.shape[1]], out=sl)
        return out

    fps = 60

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

        if not NO_CORNERS:
            for c, (st, du, pw) in zip(lay['corners'], T_CORNERS):
                put(c, corner_progress(t, st, du, pw))
        sl = self.slash
        title = self.title(t)
        blood = np.zeros((H, W), np.float32)
        if t > T_CUT[0]:
            split, maps = sl.split(title, t)
            blood = sl.blood(title, t, maps)
            title = split
        np.maximum(ink, title, out=ink)
        chips = np.zeros((H, W), np.uint8)
        sl.chips(t, chips)
        np.maximum(ink, chips.astype(np.float32) / 255, out=ink)
        deco8 = np.zeros((H, W), np.uint8)
        halo8 = np.zeros((H, W), np.uint8)
        sl.stems(t, deco8, halo8)
        deco = deco8.astype(np.float32) / 255
        halo = halo8.astype(np.float32) / 255
        rose_red = np.zeros((H, W), np.float32)
        sl.draw_roses(t, deco, rose_red, halo)
        sword = np.zeros((H, W), np.float32)
        sl.draw_sword(t, self.fps, sword)

        # breathing grain: three fixed textures cross-faded slowly (no per-frame noise crawl)
        ph = t * 0.6
        wts = [0.5 + 0.5 * np.cos(2 * np.pi * (ph - j / 3)) for j in range(3)]
        gr = sum(w_ * g_ for w_, g_ in zip(wts, self.grain)) / sum(wts)
        dens = self.wear * (1 + 0.05 * gr)
        # premultiplied layers, bottom to top: the print, the blood on its cut edges, a clearing
        # around the stems and roses, the red of the petals, the stems' and roses' ink, the rapier
        P = np.zeros((H, W, 3), np.float32)
        A = np.zeros((H, W), np.float32)

        def lay_over(cov, color):
            nonlocal P, A
            c = np.clip(cov, 0, 1)
            P = color * c[..., None] + P * (1 - c[..., None])
            A = c + A * (1 - c)

        lay_over(ink * dens, PRINT)
        lay_over(blood * (0.9 + 0.1 * dens), BLOOD)
        h_ = np.clip(halo, 0, 1)
        P *= (1 - h_[..., None]); A *= (1 - h_)
        lay_over(rose_red * dens * 0.95, RED)
        lay_over(deco * dens, PRINT)
        gl = self.glint(t)
        if gl is not None:
            gx0, gy0, star = gl
            hh, ww = star.shape
            P[gy0:gy0 + hh, gx0:gx0 + ww] *= (1 - star[..., None])
            A[gy0:gy0 + hh, gx0:gx0 + ww] *= (1 - star)
        tr = sl.trail(t)
        if tr is not None:
            smear, core = tr
            lay_over(smear, PRINT)
            P *= (1 - core[..., None]); A *= (1 - core)
        lay_over(sword, PRINT)
        rgb = P / np.maximum(A, 1e-6)[..., None]
        if self.cat is not None:
            self.cat.draw(t, rgb, A)
        dx, dy = sl.shake(t)
        if dx or dy:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            rgb = cv2.warpAffine(rgb * A[..., None], M, (W, H), flags=cv2.INTER_LINEAR)
            A = cv2.warpAffine(A, M, (W, H), flags=cv2.INTER_LINEAR)
            rgb = rgb / np.maximum(A, 1e-6)[..., None]
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
        film.fps = fps
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
        film.fps = fps
        for i in range(n):
            f = film.frame(i / fps)
            prev = over(f, PREVIEW_BG)
            for p, alpha in procs:
                p.stdin.write((f if alpha else prev).tobytes())
        for p, _ in procs:
            p.stdin.close(); p.wait()
        print('wrote', stem + '.{mov,webm,-preview.mp4}', n, 'frames')

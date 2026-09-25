"""Seraph — 7 s flight: rises from the bottom-right corner, hovers near the top, shoots off upward.

The seraph is a cut-out puppet built from refs/seraph.png: a static body (eye disc, neck, tail)
and six wings that rotate about their roots. Every frame is a pure function of time t.

  python3 seraph.py still 3.6 out/hover.png
  python3 seraph.py sheet 0.2:6.8:0.44 out/sheet.png
  python3 seraph.py cycle out/cycle.png          # one wing beat of the isolated puppet
  python3 seraph.py render seraph [--fps 60]     # seraph.mov, seraph.webm, seraph-preview.mp4
"""
import sys, os, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'refs', 'seraph.png')
W, H = 1920, 1080
DURATION = 7.0
S = 360 / 674                      # reference -> screen scale (seraph ~360 px tall)
AXIS = 270.5                       # mirror axis of the reference
BODY_C = (271.0, 418.0)            # centre of the eye disc (reference px); the puppet's origin

# ---------------------------------------------------------------- timeline (seconds)
T_RISE = (0.3, 3.0)
T_HOVER = (3.0, 5.2)               # stops and looks around: eyes roll, the body tilts after its gaze
T_GATHER = (5.2, 5.52)             # slow deep downbeat, body dips
T_EXIT = (5.52, 6.25)
SHUTTER = 0.5                      # fraction of a frame the shutter is open (180 degrees)
SUBFRAMES = 5

GLOW = np.array([255, 240, 208], np.float32) / 255
FEATHER = np.array([236, 224, 198], np.float32) / 255
PREVIEW_BG = (236, 232, 226)


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def span(t, a, b):
    return min(max((t - a) / (b - a), 0.0), 1.0)


def affine(M):
    return np.vstack([M, [0, 0, 1]])


# ---------------------------------------------------------------- cut-out
def matte(rgb):
    """Alpha from the white background; the thin grey rim around the silhouette is eroded away."""
    mn = rgb.min(2) * 255
    solid = (mn < 243).astype(np.uint8)
    solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    core = cv2.erode(solid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    a = cv2.GaussianBlur(core.astype(np.float32), (0, 0), 0.8)
    return np.clip(a * 1.15, 0, 1)


def poly(shape, pts):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.array(pts, np.int32)], 1)
    return m > 0


def labels(shape):
    """Part id per reference pixel for the left half, mirrored for the right.
    0 background, 1 body, 2 upper, 3 middle, 4 lower (left); +10 for the right side."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    lab = np.zeros(shape, np.int32)
    left = xx <= AXIS
    middle = poly(shape, [(0, 497), (40, 455), (90, 410), (140, 384), (185, 371), (228, 370), (252, 392),
                          (248, 440), (226, 463), (190, 471), (140, 479), (80, 489), (20, 500)])
    # the lower wings start at x~105; anything further out at that height is the middle wing's tip
    middle = middle | ((xx < 105) & (yy > 400))
    lower = (yy > 452) & ~middle
    upper = (yy <= 452) & ~middle
    lab[left & upper] = 2
    lab[left & middle] = 3
    lab[left & lower] = 4
    # mirror the left labels onto the right half
    mir = lab[:, ::-1]
    mir = np.roll(mir, 1, axis=1) if w % 2 == 0 else mir
    right = ~left
    lab[right] = np.where(mir[right] > 0, mir[right] + 10, 0)
    body = np.hypot(xx - BODY_C[0], yy - BODY_C[1]) < 56
    neck = (np.abs(xx - AXIS) < 24) & (yy > 322) & (yy < 380)
    tail = (np.abs(xx - AXIS) < 15) & (yy > 460) & (yy < 500)
    lab[body | neck | tail] = 1
    return lab


def assign_sides(lab, a):
    """A wing tip that crosses the mirror axis (or touches the other tip) belongs to the wing it
    grows from: give each upper/lower wing pixel to the side whose root is closer along the ink."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    roots = {2: (340, 60), 4: (480, 30)}                # (row, distance from the axis) of each root
    for k in (2, 4):
        m = np.isin(lab, [k, k + 10]) & (a > 0.3)
        ys, xs = np.nonzero(m)
        idx = -np.ones(m.shape, np.int64)
        idx[ys, xs] = np.arange(len(ys))
        rows, cols, wts = [], [], []
        for dy, dx, c in ((0, 1, 1), (1, 0, 1), (1, 1, 1.414), (1, -1, 1.414)):
            y2, x2 = ys + dy, xs + dx
            ok = (y2 < m.shape[0]) & (x2 >= 0) & (x2 < m.shape[1])
            ok[ok] = m[y2[ok], x2[ok]]
            rows.append(idx[ys[ok], xs[ok]]); cols.append(idx[y2[ok], x2[ok]]); wts.append(np.full(ok.sum(), c))
        g = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(len(ys),) * 2)
        ry, rd = roots[k]
        seeds = []
        for sx in (AXIS - rd, AXIS + rd):
            d2 = (ys - ry) ** 2 + (xs - sx) ** 2
            seeds.append(int(d2.argmin()))
        d = dijkstra(g, directed=False, indices=seeds)
        left = d[0] <= d[1]
        both_inf = ~np.isfinite(d[0]) & ~np.isfinite(d[1])
        left[both_inf] = xs[both_inf] < AXIS
        lab[ys, xs] = np.where(left, k, k + 10)
    return lab


# wing pivots (left side, reference px); the right side mirrors them
PIVOTS = {2: (258.0, 372.0), 3: (230.0, 424.0), 4: (256.0, 470.0)}
# beat amplitude (degrees) and phase lag (cycles) per wing pair
AMP = {2: 14.0, 3: 10.0, 4: 8.0}
LAG = {2: 0.0, 3: 1 / 6, 4: 1 / 3}
# back to front
ORDER = [2, 12, 4, 14, 3, 13]

# eyes (reference px): centre, half-axes, rotation (deg), owning part
EYE_MAIN = dict(c=(270.3, 417.5), ax=(20.5, 9.5), iris=(270.6, 417.0, 10.0))
EYES_WING_LEFT = [((171, 270), (6.5, 12), 25, 2), ((187, 327), (6.5, 10), 40, 2),
                  ((176, 395), (9, 5.5), 10, 3), ((240, 490), (6, 9), -20, 4)]
# blink times for the wing eyes (seconds); each eye gets its own moments
WING_BLINKS = [[1.35, 4.55], [2.2, 3.7], [0.95, 4.15], [2.75, 5.0],     # left eyes
               [1.7, 4.8], [0.7, 3.45], [2.45, 4.35], [1.15, 3.9]]      # right eyes
MAIN_BLINK = 5.05
# wing-eye roll during the hover: angular speed (rad/s, sign = direction) and start angle per eye
WING_ROLL = [(4.2, 0.3), (-5.1, 2.0), (3.4, 4.1), (-3.9, 1.1), (-4.6, 5.0), (5.4, 0.9), (-3.2, 3.3), (4.8, 2.6)]
BLINK_LEN = 0.2


def mirror_pt(p):
    return (2 * AXIS - p[0], p[1])


class Puppet:
    def __init__(self):
        rgb = np.asarray(Image.open(REF).convert('RGB')).astype(np.float32) / 255
        self.ref_shape = rgb.shape[:2]
        a = matte(rgb)
        lab = labels(self.ref_shape)
        lab[a < 0.02] = 0
        lab = assign_sides(lab, a)
        self.parts = {}
        for pid in [1] + ORDER:
            own = (lab == pid)
            layer_a = a * own
            col = rgb.copy()
            if pid != 1:
                # extend each wing under whatever sits in front of it, so rotating a front wing
                # never reveals a hole: in-paint the hidden band from the wing's own feathers
                # only the body and the same side's middle wing ever cover this wing
                side = 0 if pid < 10 else 10
                front = np.isin(lab, [1] if pid % 10 == 3 else [1, 3 + side])
                core = own & (a > 0.95)
                core = cv2.erode(core.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                dist = cv2.distanceTransform((~own).astype(np.uint8), cv2.DIST_L2, 5)
                ext = front & (a > 0.5) & (dist < 40)
                # nearest solid pixel of this wing for every hidden pixel, lightly softened
                _, near = cv2.distanceTransformWithLabels((~core).astype(np.uint8), cv2.DIST_L2, 5,
                                                          labelType=cv2.DIST_LABEL_PIXEL)
                ys, xs = np.nonzero(core)
                idx = np.clip(near - 1, 0, len(ys) - 1)
                fill = cv2.GaussianBlur(rgb[ys[idx], xs[idx]], (0, 0), 1.2)
                col = np.where(own[..., None], rgb, fill)
                # the hidden extension fades out with depth, so an exposed edge is soft, never a shard
                layer_a = np.maximum(layer_a, ext * a * smooth(1 - dist / 40))
            self.parts[pid] = self.scaled(col, layer_a)
        self.lid_color = {}
        self.pivots = {}
        for k, p in PIVOTS.items():
            self.pivots[k] = (p[0] * S, p[1] * S)
            self.pivots[k + 10] = (mirror_pt(p)[0] * S, p[1] * S)
        self.body_c = (BODY_C[0] * S, BODY_C[1] * S)
        # eyes on screen scale
        self.wing_eyes = []
        for (c, ax, rot, pid) in EYES_WING_LEFT:
            self.wing_eyes.append((c, ax, rot, pid))
        for (c, ax, rot, pid) in EYES_WING_LEFT:
            self.wing_eyes.append((mirror_pt(c), ax, -rot, pid + 10))
        self.eye_lid_rgb = []
        for (c, ax, rot, pid) in self.wing_eyes:
            ring = self.ring_color(rgb, c, ax)
            self.eye_lid_rgb.append(ring)
        self.eye_sclera = [self.sclera_color(rgb, c, ax) for (c, ax, rot, pid) in self.wing_eyes]
        self.main_lid = self.ring_color(rgb, EYE_MAIN['c'], (EYE_MAIN['ax'][0] + 4, EYE_MAIN['ax'][1] + 6))
        self.sclera = np.array([0.80, 0.78, 0.74], np.float32)

    @staticmethod
    def ring_color(rgb, c, ax):
        yy, xx = np.mgrid[0:rgb.shape[0], 0:rgb.shape[1]]
        r = np.hypot((xx - c[0]) / ax[0], (yy - c[1]) / ax[1])
        m = (r > 1.25) & (r < 1.9)
        return np.median(rgb[m], axis=0).astype(np.float32)

    @staticmethod
    def sclera_color(rgb, c, ax):
        yy, xx = np.mgrid[0:rgb.shape[0], 0:rgb.shape[1]]
        m = np.hypot((xx - c[0]) / ax[0], (yy - c[1]) / ax[1]) < 1
        px = rgb[m]
        lum = px.mean(1)
        return np.median(px[lum >= np.percentile(lum, 75)], axis=0).astype(np.float32)

    def shift_iris(self, img, c, ax, rot, sclera, dx, dy):
        """Roll a small wing eye: its dark iris slides inside the opening by (dx, dy) screen px."""
        if abs(dx) < 1e-3 and abs(dy) < 1e-3:
            return img
        img = img.copy()
        cx, cy = c[0] * S, c[1] * S
        ax_ = (ax[0] * S, ax[1] * S)
        r = int(max(ax_) + 5)
        x0, y0 = int(cx) - r, int(cy) - r
        patch = img[y0:y0 + 2 * r, x0:x0 + 2 * r].copy()
        yy, xx = np.mgrid[y0:y0 + 2 * r, x0:x0 + 2 * r].astype(np.float32)
        th = np.radians(rot)
        u = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
        v = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
        eye = np.clip((1 - np.hypot(u / ax_[0], v / ax_[1])) * 3, 0, 1)
        a = patch[..., 3:4]
        lum = (patch[..., :3] / np.maximum(a, 1e-6)).mean(2)
        thr = np.percentile(lum[eye > 0.5], 45)
        iris = np.clip((thr - lum) / 0.06 + 0.5, 0, 1) * (eye > 0.2)
        moved = cv2.warpAffine(patch * iris[..., None], np.float32([[1, 0, dx], [0, 1, dy]]), (2 * r, 2 * r),
                               flags=cv2.INTER_LINEAR)
        base = np.concatenate([sclera * a, a], 2)
        inner = moved + base * (1 - moved[..., 3:4] / np.maximum(a, 1e-6))
        e = eye[..., None]
        img[y0:y0 + 2 * r, x0:x0 + 2 * r] = patch * (1 - e) + inner * e
        return img

    def scaled(self, col, a):
        """Premultiplied RGBA at screen scale (one area-filtered resize from the reference)."""
        h, w = self.ref_shape
        size = (round(w * S), round(h * S))
        pm = np.dstack([col * a[..., None], a]).astype(np.float32)
        return cv2.resize(pm, size, interpolation=cv2.INTER_AREA)

    # -- eyes ----------------------------------------------------------------------------
    def lid(self, img, c, ax, rot, color, close):
        """Close an eye by `close` (0..1): a lid of the surrounding feather colour sweeps down."""
        if close <= 0:
            return img
        img = img.copy()
        cx, cy = c[0] * S, c[1] * S
        ax_ = (ax[0] * S * 1.25, ax[1] * S * 1.3)
        r = int(max(ax_) + 4)
        x0, y0 = int(cx) - r, int(cy) - r
        yy, xx = np.mgrid[y0:y0 + 2 * r, x0:x0 + 2 * r].astype(np.float32)
        th = np.radians(rot)
        u = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
        v = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
        inside = np.clip(1.5 - np.hypot(u / ax_[0], v / ax_[1]) * 1.5 + 0.5, 0, 1)
        edge = -ax_[1] + 2 * ax_[1] * close                           # lid edge in the eye's frame
        cover = np.clip(edge - v + 0.5, 0, 1) * inside
        lash = np.exp(-((v - edge) / 0.8) ** 2) * inside * (0.25 < close < 0.98)
        patch = img[y0:y0 + 2 * r, x0:x0 + 2 * r]
        a = patch[..., 3:4]
        rgb = color * a
        patch[..., :3] = patch[..., :3] * (1 - cover[..., None]) + rgb * cover[..., None]
        patch[..., :3] *= (1 - 0.35 * lash[..., None])
        return img

    def look(self, img, dx, dy):
        """Shift the central iris by (dx, dy) screen px inside the eye opening."""
        if abs(dx) < 1e-3 and abs(dy) < 1e-3:
            return img
        img = img.copy()
        cx, cy = EYE_MAIN['c'][0] * S, EYE_MAIN['c'][1] * S
        ax, ay = EYE_MAIN['ax'][0] * S, EYE_MAIN['ax'][1] * S
        r = int(ax + 6)
        x0, y0 = int(cx) - r, int(cy) - r
        patch = img[y0:y0 + 2 * r, x0:x0 + 2 * r].copy()
        yy, xx = np.mgrid[y0:y0 + 2 * r, x0:x0 + 2 * r].astype(np.float32)
        eye = np.clip((1 - np.hypot((xx - cx) / ax, (yy - cy) / ay)) * ax * 0.6, 0, 1)
        ix, iy, ir = [v * S for v in EYE_MAIN['iris']]
        iris = np.clip(ir - np.hypot(xx - ix, yy - iy) + 0.5, 0, 1)[..., None] * patch
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        moved = cv2.warpAffine(iris, M, (2 * r, 2 * r), flags=cv2.INTER_LINEAR)
        a = patch[..., 3:4]
        base = np.concatenate([self.sclera * a, a], 2)
        inner = moved + base * (1 - moved[..., 3:4] / np.maximum(a, 1e-6))
        e = eye[..., None]
        patch = patch * (1 - e) + inner * e
        img[y0:y0 + 2 * r, x0:x0 + 2 * r] = patch
        return img

    def body_at(self, t):
        img = self.parts[1]
        img = self.look(img, *gaze(t))
        close = blink(t, MAIN_BLINK)
        if close > 0:
            c, ax = EYE_MAIN['c'], EYE_MAIN['ax']
            img = self.lid(img, c, (ax[0] * 0.95, ax[1] * 1.1), 0, self.main_lid, close)
        return img

    def wing_at(self, pid, t):
        img = self.parts[pid]
        for i, (c, ax, rot, owner) in enumerate(self.wing_eyes):
            if owner != pid:
                continue
            img = self.shift_iris(img, c, ax, rot, self.eye_sclera[i], *wing_gaze(t, i))
            close = max([blink(t, b) for b in WING_BLINKS[i]] + [0])
            if close > 0:
                img = self.lid(img, c, ax, rot, self.eye_lid_rgb[i], close)
        return img


def blink(t, at):
    u = (t - at) / BLINK_LEN
    if u <= 0 or u >= 1:
        return 0.0
    return float(np.sin(np.pi * u) ** 0.7)


RX, RY = 3.4, 1.4                  # how far the central iris can travel (screen px)


def _roll(t, ta, tb, th0, th1):
    u = smooth(span(t, ta, tb))
    th = th0 + (th1 - th0) * u
    return (RX * np.cos(th), RY * np.sin(th))


def gaze(t):
    """Central iris offset (screen px). Looks ahead while rising; in the hover it glances left,
    rolls up and over to the right, holds, spins a full circle, then settles on the viewer."""
    ahead = (-0.8, -0.9)
    up = (0.0, -1.2)
    left = (-RX, 0.3)

    def mix(p, q, u):
        u = smooth(u)
        return (p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u)

    if t < 2.75:
        return ahead
    if t < 3.2:
        return mix(ahead, left, span(t, 2.75, 3.2))
    if t < 3.6:
        return (left[0], left[1] - 0.3 * np.sin(np.pi * span(t, 3.2, 3.6)))
    if t < 4.05:                                      # left -> up -> right
        return mix(left, _roll(t, 3.6, 4.05, np.pi, 2 * np.pi), min(1, span(t, 3.6, 3.7) * 4))
    if t < 4.4:
        return (RX, 0.0)
    if t < 4.95:                                      # a full spin: right -> up -> left -> down -> right
        return _roll(t, 4.4, 4.95, 0.0, -2 * np.pi)
    if t < 5.2:
        return mix((RX, 0.0), (0.0, 0.0), span(t, 4.95, 5.15))
    return mix((0.0, 0.0), up, span(t, 5.25, 5.5))


def hover_env(t):
    return smooth(span(t, T_HOVER[0] - 0.2, T_HOVER[0] + 0.3)) * (1 - smooth(span(t, T_HOVER[1] - 0.2, T_HOVER[1] + 0.1)))


def wing_gaze(t, i):
    """Each wing eye rolls its iris in its own direction and tempo while the seraph hovers."""
    w, th0 = WING_ROLL[i]
    env = hover_env(t)
    th = th0 + w * max(0.0, t - T_HOVER[0])
    roll = (1.3 * np.cos(th), 1.0 * np.sin(th))
    rest = (0.0, -0.6)
    return (rest[0] * (1 - env) + roll[0] * env, rest[1] * (1 - env) + roll[1] * env)


def tilt(t):
    """-1..1: the body leans after the gaze (a little behind it) during the hover."""
    return float(np.clip(gaze(t - 0.14)[0] / RX, -1, 1)) * hover_env(t)


# ---------------------------------------------------------------- motion
def beat_freq(t):
    """Wing beats per second."""
    rise = 2.2
    hover = 1.1
    if t < T_RISE[1] - 0.6:
        return rise
    if t < T_HOVER[0] + 0.3:
        return rise + (hover - rise) * smooth(span(t, T_RISE[1] - 0.6, T_HOVER[0] + 0.3))
    if t < T_GATHER[0] - 0.25:
        return hover
    if t < T_GATHER[1]:
        return hover + (0.9 - hover) * smooth(span(t, T_GATHER[0] - 0.25, T_GATHER[0]))
    return 0.9 + (3.0 - 0.9) * smooth(span(t, T_GATHER[1], T_GATHER[1] + 0.25))


_PH_DT = 1 / 2000
_PH_T = np.arange(0, DURATION + 1, _PH_DT)
_PH = np.concatenate([[0], np.cumsum([beat_freq(x) * _PH_DT for x in _PH_T[:-1]])])
# choose the offset so the deep downbeat lands on the gather
_PH0 = (0.25 - np.interp(T_GATHER[0] + 0.05, _PH_T, _PH)) % 1


def phase(t):
    return float(np.interp(t, _PH_T, _PH)) + _PH0


def stroke(p):
    """+1 = wings up, -1 = wings down. Downstroke takes 40% of the cycle (faster than the upstroke)."""
    s = p % 1
    if s < 0.4:
        return float(np.cos(np.pi * s / 0.4))
    return float(-np.cos(np.pi * (s - 0.4) / 0.6))


def downstroke_push(p):
    """0..1, how hard the wings are pushing (for the small lift on each downstroke)."""
    s = p % 1
    return float(np.sin(np.pi * s / 0.4) if s < 0.4 else 0.0)


def bezier(P, u):
    P = np.asarray(P, np.float64)
    return ((1 - u) ** 3 * P[0] + 3 * (1 - u) ** 2 * u * P[1] + 3 * (1 - u) * u * u * P[2] + u ** 3 * P[3])


RISE_PATH = [(1735, 1330), (1790, 880), (1470, 520), (1500, 300)]
HOVER_PT = np.array(RISE_PATH[-1], np.float64)


def base_position(t):
    """Centre of the eye disc on the page, before the wing-driven lift."""
    if t <= T_RISE[1]:
        u = span(t, *T_RISE)
        # quick start, long deceleration into the hover
        e = 1 - (1 - u) ** 1.7
        return bezier(RISE_PATH, e)
    p = HOVER_PT.copy()
    bob_amt = smooth(span(t, T_HOVER[0], T_HOVER[0] + 0.5))
    p[1] += 10 * np.sin(2 * np.pi * (t - T_HOVER[0]) / 1.3) * bob_amt
    if t > T_GATHER[0]:
        g = span(t, *T_GATHER)
        p[1] += 18 * np.sin(np.pi * 0.5 * g) ** 2 * (1 - smooth(span(t, T_GATHER[1], T_GATHER[1] + 0.12)))
    if t > T_EXIT[0]:
        u = span(t, *T_EXIT)
        p[1] -= 870 * u ** 2.4
        p[0] += 10 * u
    return p


def velocity(t, dt=1 / 240):
    return (base_position(t + dt) - base_position(t - dt)) / (2 * dt)


def pose(t):
    pos = base_position(t)
    ph = phase(t)
    # each downstroke lifts the body a few pixels
    pos = pos + np.array([0.0, -5.0 * downstroke_push(ph)])
    v = velocity(t)
    speed = np.hypot(*v)
    lean = np.clip(-np.degrees(np.arctan2(v[0], -v[1] + 1e-6)) * 0.5, -6, 6) * smooth(speed / 250)
    # looking around: lean toward the side it looks at (counter-clockwise = left) and drift that way
    tl = tilt(t)
    lean += -8.0 * tl
    pos = pos + np.array([7.0 * tl, 0.0])
    stretch = 1 + 0.04 * smooth((speed - 500) / 900) if t > T_EXIT[0] else 1.0
    # deep gather beat: bigger amplitude on the downbeat before the exit
    gain = 1 + 0.6 * np.sin(np.pi * span(t, T_GATHER[0] - 0.15, T_GATHER[1])) ** 2
    return pos, ph, lean, stretch, gain


# ---------------------------------------------------------------- drawing
class Film:
    def __init__(self):
        self.p = Puppet()
        rng = np.random.default_rng(11)
        # shed feathers: spawn time, offset from the body, fall speed, sway, spin, size
        spawns = list(rng.uniform(0.7, 2.8, 7)) + list(rng.uniform(5.4, 6.1, 4))
        self.feathers = []
        for i, ts in enumerate(sorted(spawns)):
            self.feathers.append(dict(t0=ts, off=(rng.uniform(-120, 120), rng.uniform(-60, 90)),
                                      vy=rng.uniform(35, 75), sway=rng.uniform(12, 28),
                                      sf=rng.uniform(0.4, 0.8), sp=rng.uniform(0, 1),
                                      rot0=rng.uniform(0, 180), spin=rng.uniform(-90, 90),
                                      L=rng.uniform(7, 12), life=rng.uniform(1.6, 2.4)))

    def seraph(self, t):
        """Premultiplied RGBA of the whole seraph on the page at an instant."""
        pos, ph, lean, stretch, gain = pose(t)
        out = np.zeros((H, W, 4), np.float32)
        cx, cy = self.p.body_c
        # global: puppet space -> page (lean about the body, stretch along travel)
        G = cv2.getRotationMatrix2D((cx, cy), lean, 1.0)
        G = affine(G) @ affine(np.float64([[1 / stretch ** 0.5, 0, cx * (1 - 1 / stretch ** 0.5)],
                                           [0, stretch, cy * (1 - stretch)]]))
        G = affine(np.float64([[1, 0, pos[0] - cx], [0, 1, pos[1] - cy]])) @ G
        # crop to the bounding box the seraph can occupy (for speed)
        R = 330
        x0, y0 = int(pos[0]) - R, int(pos[1]) - R
        if x0 > W or y0 > H or x0 + 2 * R < 0 or y0 + 2 * R < 0:
            return out
        T = affine(np.float64([[1, 0, -x0], [0, 1, -y0]]))
        local = np.zeros((2 * R, 2 * R, 4), np.float32)

        def over(dst, src):
            dst *= 1 - src[..., 3:4]
            dst += src

        for pid in ORDER:
            k = pid % 10
            sign = -1 if pid < 10 else 1          # left wings rotate clockwise on the upstroke
            st = stroke(ph - LAG[k])
            if k == 2:
                # the tall upper wings mostly open outward; on the upstroke they only just close
                st = 0.55 * st - 0.35
            ang = sign * AMP[k] * gain * st
            px, py = self.p.pivots[pid]
            Rm = affine(cv2.getRotationMatrix2D((px, py), ang, 1.0))
            # feathers flex a little at the ends of the stroke
            flex = 1 + 0.04 * abs(stroke(ph - LAG[k]))
            F = affine(np.float64([[flex, 0, px * (1 - flex)], [0, 1, 0]])) if k == 3 else \
                affine(np.float64([[1, 0, 0], [0, flex, py * (1 - flex)]]))
            M = (T @ G @ Rm @ F)[:2]
            img = self.p.wing_at(pid, t)
            over(local, cv2.warpAffine(img, M, (2 * R, 2 * R), flags=cv2.INTER_CUBIC))
        M = (T @ G)[:2]
        over(local, cv2.warpAffine(self.p.body_at(t), M, (2 * R, 2 * R), flags=cv2.INTER_CUBIC))
        local = np.clip(local, 0, 1)
        # paste
        ox0, oy0 = max(0, x0), max(0, y0)
        ox1, oy1 = min(W, x0 + 2 * R), min(H, y0 + 2 * R)
        out[oy0:oy1, ox0:ox1] = local[oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0]
        return out

    def glow(self, alpha, t):
        small = cv2.resize(alpha, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(small, (0, 0), 7)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        k = 0.30 + 0.14 * smooth(span(t, T_HOVER[0] - 0.3, T_HOVER[0] + 0.4)) * (1 - smooth(span(t, *T_GATHER)))
        a = np.clip(g * 1.6, 0, 1) * k
        return np.dstack([GLOW * a[..., None], a])

    def feathers_layer(self, t):
        a8 = np.zeros((H, W), np.uint8)
        rib = np.zeros((H, W), np.uint8)
        for f in self.feathers:
            u = t - f['t0']
            if u < 0 or u > f['life']:
                continue
            base = base_position(f['t0'])
            x = base[0] + f['off'][0] + f['sway'] * np.sin(2 * np.pi * (f['sf'] * u + f['sp']))
            y = base[1] + f['off'][1] + f['vy'] * u + 8 * u * u
            ang = f['rot0'] + f['spin'] * u + 25 * np.sin(2 * np.pi * f['sf'] * u)
            fade = smooth(u / 0.25) * (1 - smooth((u - f['life'] + 0.6) / 0.6))
            L = f['L']
            c = (int(x * 16), int(y * 16))
            cv2.ellipse(a8, c, (int(L * 16), int(L * 0.32 * 16)), ang, 0, 360, int(220 * fade), -1, cv2.LINE_AA, 4)
            dx, dy = np.cos(np.radians(ang)) * L * 16, np.sin(np.radians(ang)) * L * 16
            cv2.line(rib, (int(c[0] - dx), int(c[1] - dy)), (int(c[0] + dx * 0.9), int(c[1] + dy * 0.9)),
                     int(120 * fade), 1, cv2.LINE_AA, 4)
        a = a8.astype(np.float32) / 255
        r = rib.astype(np.float32) / 255
        col = FEATHER[None, None, :] * (1 - 0.35 * r[..., None])
        return np.dstack([col * a[..., None], a])

    def frame(self, t, fps=60):
        # temporal supersampling inside the shutter for natural motion blur
        acc = np.zeros((H, W, 4), np.float32)
        for k in range(SUBFRAMES):
            ts = t + (k + 0.5) / SUBFRAMES * SHUTTER / fps - SHUTTER / fps / 2
            acc += self.seraph(ts)
        ser = acc / SUBFRAMES
        out = self.glow(ser[..., 3], t)
        out = out * (1 - ser[..., 3:4]) + ser
        fea = self.feathers_layer(t)
        out = fea + out * (1 - fea[..., 3:4])
        a = out[..., 3:4]
        rgb = out[..., :3] / np.maximum(a, 1e-6)
        rgba = np.concatenate([np.clip(rgb, 0, 1), np.clip(a, 0, 1)], 2)
        return (rgba * 255 + 0.5).astype(np.uint8)


def over_bg(rgba, bg):
    a = rgba[..., 3:4].astype(np.float32) / 255
    return (rgba[..., :3] * a + np.array(bg, np.float32) * (1 - a) + 0.5).astype(np.uint8)


def sheet(film, times, path, cols=4, bg=PREVIEW_BG):
    tiles = []
    for t in times:
        f = cv2.resize(over_bg(film.frame(t), bg), (480, 270), interpolation=cv2.INTER_AREA)
        im = Image.fromarray(f)
        ImageDraw.Draw(im).text((8, 6), f't={t:.2f}', fill=(200, 0, 0))
        tiles.append(np.asarray(im))
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
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
    elif cmd == 'cycle':
        # one beat of the isolated puppet at the hover, 12 frames, on light and dark backgrounds
        t0 = 3.5
        f0 = beat_freq(t0)
        tiles = []
        for i in range(12):
            t = t0 + i / 12 / f0
            fr = film.seraph(t)
            p = base_position(t)
            x0, y0 = int(p[0]) - 250, int(p[1]) - 230
            crop = fr[max(0, y0):y0 + 480, x0:x0 + 500]
            a = crop[..., 3:4]
            bg = np.array((40, 36, 44) if i % 2 else (236, 232, 226), np.float32) / 255
            img = crop[..., :3] + bg * (1 - a)
            tiles.append((np.clip(img, 0, 1) * 255).astype(np.uint8))
        hmin = min(t_.shape[0] for t_ in tiles)
        tiles = [t_[:hmin] for t_ in tiles]
        Image.fromarray(np.concatenate([np.concatenate(tiles[:6], 1), np.concatenate(tiles[6:], 1)], 0)).save(sys.argv[2])
    elif cmd == 'render':
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
        procs = [(subprocess.Popen(c, stdin=subprocess.PIPE), alpha) for c, alpha in jobs]
        n = int(round(DURATION * fps))
        for i in range(n):
            f = film.frame(i / fps, fps)
            prev = over_bg(f, PREVIEW_BG)
            for p, alpha in procs:
                p.stdin.write((f if alpha else prev).tobytes())
        for p, _ in procs:
            p.stdin.close(); p.wait()
        print('wrote', stem + '.{mov,webm,-preview.mp4}', n, 'frames')

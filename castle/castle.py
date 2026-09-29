"""Castle in the fog — 12 s, 1920x1080, 60 fps, silent.

The castle emerges from a white fog, ravens fly past it, the sky darkens, two bolts of lightning
strike (the second close), ravens scatter from the towers, rain sets in and a light comes on in a
tower window. Built on the reference photo (refs/castle.webp); fog, ravens, lightning and rain are
animated on top of it. Every frame is a pure function of t.

  python3 castle.py still 8.55 out/a.png
  python3 castle.py sheet 0:11.9:0.5 out/sheet.png
  python3 castle.py raven out/raven.png          # one wing beat of the raven puppet
  python3 castle.py render castle [--fps 60]      # castle.mp4 (+ castle.webm)
"""
import sys, os, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'refs')
W, H = 1920, 1080
DURATION = 12.0
SUB = 3                                   # fog textures are computed at 1/3 resolution

# ---------------------------------------------------------------- timeline (seconds)
T_REVEAL = (0.8, 4.8)                     # the fog parts, spires first
T_DARKEN = (5.8, 7.1)                     # the sky darkens, the fog thickens
BOLTS = [dict(t=7.2, strength=0.55, bolt=0), dict(t=8.5, strength=1.0, bolt=1)]
FLICKERS = [(10.25, 0.22), (11.35, 0.15)] # far-off lightning inside the clouds (time, strength)
T_RAIN = (8.45, 9.4)
T_WINDOW = (10.4, 11.4)
PUSH = (0.0, 12.0, 1.045)                 # slow push in on the castle
ZOOM_C = (1075.0, 420.0)

WARM = np.array([1.0, 0.72, 0.42], np.float32)
BOLT_COL = np.array([0.93, 0.95, 1.0], np.float32)


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def span(t, a, b):
    return min(max((t - a) / (b - a), 0.0), 1.0)


def periodic_noise(h, w, scale, seed, aniso=1.0):
    """Smooth noise that wraps around in both directions (filtered in the Fourier domain)."""
    rng = np.random.default_rng(seed)
    f = np.fft.fft2(rng.standard_normal((h, w)))
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.fftfreq(w)[None, :]
    k = np.sqrt((kx * aniso) ** 2 + ky ** 2)
    f *= np.exp(-(k * scale) ** 2)
    n = np.real(np.fft.ifft2(f))
    return ((n - n.mean()) / n.std()).astype(np.float32)


def shift(tex, dx, dy):
    """Sub-pixel scroll of a periodic texture."""
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(tex, M, (tex.shape[1], tex.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_WRAP)


def bezier(P, u):
    P = np.asarray(P, np.float64)
    return (1 - u) ** 3 * P[0] + 3 * (1 - u) ** 2 * u * P[1] + 3 * (1 - u) * u * u * P[2] + u ** 3 * P[3]


# ---------------------------------------------------------------- the raven
class Raven:
    """Puppet from refs/raven3.png (side view, wings up): body and wing. The wing flaps by being
    scaled through its root line (up = +1, edge-on = 0, down = negative) with a little sweep
    back of the tips; a second, far wing follows a touch behind so the bird never looks wingless."""

    def __init__(self):
        g = np.asarray(Image.open(os.path.join(REF, 'raven3.png')).convert('L')).astype(np.float32) / 255
        c = g[70:355, 250:585]
        m = np.clip((0.34 - c) / 0.14, 0, 1)
        n, lab, st, _ = cv2.connectedComponentsWithStats((m > 0.5).astype(np.uint8), 8)
        big = 1 + int(np.argmax(st[1:, 4]))
        keep = cv2.dilate((lab == big).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        m = m * keep
        # tuck the dangling legs away (thin, bottom left)
        legs = np.zeros_like(m, bool)
        legs[222:, 60:150] = True
        opened = cv2.morphologyEx((m > 0.5).astype(np.uint8), cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
        m = np.where(legs, m * opened, m)
        h, w = m.shape
        self.root_y = 178.0
        yy, xx = np.mgrid[0:h, 0:w]
        # wing and body overlap by a few rows at the root, so there is never a seam
        self.wing = (m * ((yy < self.root_y + 5) & (xx > 105))).astype(np.float32)
        self.body = (m * ~((yy < self.root_y - 3) & (xx > 105))).astype(np.float32)
        self.shoulder_x = 210.0
        self.size = (w, h)
        self.centre = np.array([200.0, 205.0])                          # body centre (sprite px)

    @staticmethod
    def stroke(ph):
        """+1 wings up ... -0.8 wings down; the downstroke takes 40 % of the beat."""
        s = ph % 1
        if s < 0.4:
            v = np.cos(np.pi * s / 0.4)
        else:
            v = -np.cos(np.pi * (s - 0.4) / 0.6)
        return 0.1 + 0.9 * v

    def draw(self, dst, x, y, scale, heading, ph, alpha=1.0, flip=False):
        """Adds the raven's coverage to `dst` (float, full frame). (x, y) is the body centre on
        screen, heading the flight direction in radians (screen, y down), ph the beat phase."""
        w, h = self.size
        s = self.stroke(ph)
        lift = -6.0 * (s - 0.1)                                         # body rises on the downstroke
        # sprite -> screen: centre at origin, mirror when flying left, rotate to the heading
        ang = heading if not flip else heading - np.pi
        ca, sa = np.cos(ang), np.sin(ang)
        mx = -1.0 if flip else 1.0
        G = np.array([[ca * scale * mx, -sa * scale, 0], [sa * scale * mx, ca * scale, 0], [0, 0, 1]], np.float64)
        G[0, 2] = x
        G[1, 2] = y
        C = np.array([[1, 0, -self.centre[0]], [0, 1, -self.centre[1] + lift], [0, 0, 1]], np.float64)
        R = int(max(w, h) * scale * 0.85) + 6
        x0, y0 = int(x) - R, int(y) - R
        T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
        box = np.zeros((2 * R, 2 * R), np.float32)

        def put(img, M):
            np.maximum(box, cv2.warpAffine(img, (T @ G @ M)[:2], (2 * R, 2 * R), flags=cv2.INTER_LINEAR), out=box)

        for k, (ds, dx) in enumerate(((-0.12, 16.0), (0.0, 0.0))):         # far wing first
            sk = self.stroke(ph - 0.04) * 0.92 if k == 0 else s
            sweep = 0.35 * (1 - sk)                                          # tips trail on the way down
            Wm = np.array([[1, -sweep * np.sign(sk + 1e-6) * 0.0, 0], [0, sk, self.root_y * (1 - sk)], [0, 0, 1]], np.float64)
            Wm[0, 1] = sweep * 0.6                                           # x shifts with height
            Wm[0, 2] = -sweep * 0.6 * self.root_y + dx
            put(self.wing, C @ Wm)
        put(self.body, C)
        ys, xs = slice(max(0, y0), min(H, y0 + 2 * R)), slice(max(0, x0), min(W, x0 + 2 * R))
        if ys.stop <= ys.start or xs.stop <= xs.start:
            return
        sub = box[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
        np.maximum(dst[ys, xs], sub * alpha, out=dst[ys, xs])


def path_pose(P, u):
    p = bezier(P, u)
    q = bezier(P, min(1, u + 0.01)) - bezier(P, max(0, u - 0.01))
    return p, np.arctan2(q[1], q[0])


# (start, end, path control points, scale, beats per second, phase, layer)
# layer "near" is drawn over the fog, "far" under it
FLIGHTS = [
    dict(t=(3.0, 5.7), P=[(2150, 560), (1500, 470), (700, 520), (-260, 400)], scale=0.62, f=3.0, ph=0.1, layer='near'),
    dict(t=(3.35, 6.6), P=[(2080, 250), (1180, 110), (900, 420), (-180, 170)], scale=0.3, f=3.6, ph=0.55, layer='near'),
    dict(t=(3.9, 6.9), P=[(2020, 430), (1500, 360), (800, 400), (-120, 320)], scale=0.17, f=4.2, ph=0.3, layer='far'),
    # the second bolt scares ravens off the towers
    dict(t=(8.55, 10.6), P=[(1066, 300), (1150, 150), (1500, 60), (2050, -80)], scale=0.13, f=5.0, ph=0.0, layer='far'),
    dict(t=(8.6, 10.9), P=[(1180, 400), (1320, 300), (1700, 280), (2060, 150)], scale=0.11, f=5.4, ph=0.4, layer='far'),
    dict(t=(8.62, 10.5), P=[(930, 360), (820, 220), (500, 120), (-120, 60)], scale=0.14, f=4.8, ph=0.7, layer='far'),
    dict(t=(8.7, 11.2), P=[(1010, 330), (960, 170), (700, 40), (300, -120)], scale=0.1, f=5.6, ph=0.2, layer='far'),
]


# ---------------------------------------------------------------- lightning
def bolt_path(start, end, seed, rough=0.28, depth=7):
    """Midpoint-displacement bolt with branches: list of polylines and their weights."""
    rng = np.random.default_rng(seed)

    def split(a, b, d):
        if d == 0:
            return [a, b]
        m = (a + b) / 2
        L = np.linalg.norm(b - a)
        n = np.array([-(b - a)[1], (b - a)[0]]) / (L + 1e-9)
        m = m + n * rng.normal(0, rough * L * 0.5)
        return split(a, m, d - 1)[:-1] + split(m, b, d - 1)

    main = np.array(split(np.array(start, float), np.array(end, float), depth))
    lines = [(main, 1.0)]
    for _ in range(5):
        i = int(rng.integers(len(main) // 6, len(main) * 3 // 4))
        a = main[i]
        dirn = main[min(len(main) - 1, i + 8)] - a
        ang = np.arctan2(dirn[1], dirn[0]) + rng.choice([-1, 1]) * rng.uniform(0.35, 0.8)
        L = rng.uniform(0.15, 0.35) * np.linalg.norm(np.array(end) - np.array(start))
        b = a + L * np.array([np.cos(ang), np.sin(ang)])
        lines.append((np.array(split(a, b, depth - 2)), 0.45))
    return lines


def bolt_mask(lines, width):
    S = 16
    core = np.zeros((H, W), np.uint8)
    for pts, wgt in lines:
        cv2.polylines(core, [np.round(pts * S).astype(np.int32)], False, int(255 * wgt),
                      max(1, int(width * (0.55 + 0.45 * wgt))), cv2.LINE_AA, 4)
    core = core.astype(np.float32) / 255
    glow = cv2.GaussianBlur(core, (0, 0), 9) * 3 + cv2.GaussianBlur(core, (0, 0), 40) * 6
    return core, np.clip(glow, 0, 1)


def flash_env(t, t0):
    """Strobing lightning flash: a bright first stroke, a dimmer restrike, a fading glow."""
    u = t - t0
    if u < 0 or u > 0.9:
        return 0.0
    v = 1.0 * np.exp(-u / 0.05)
    v += 0.7 * np.exp(-max(0.0, u - 0.11) / 0.04) * (u > 0.11)
    v += 0.35 * np.exp(-max(0.0, u - 0.26) / 0.12) * (u > 0.26)
    return float(min(1.0, v))


# ---------------------------------------------------------------- the film
class Film:
    def __init__(self):
        img = np.asarray(Image.open(os.path.join(REF, 'castle.webp')).convert('L')).astype(np.float32) / 255
        self.photo = cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
        # how "open sky / fog" each pixel of the photo is (the castle and trees are dark)
        self.light = smooth((cv2.GaussianBlur(self.photo, (0, 0), 1.2) - 0.3) / 0.28)
        h, w = H // SUB, W // SUB
        self.n1 = periodic_noise(h, w, 170, 1, aniso=2.4)                # broad, stretched banks
        self.n2 = periodic_noise(h, w, 80, 2, aniso=1.8)
        self.n3 = periodic_noise(h, w, 32, 3, aniso=1.3)
        yy = np.linspace(0, 1, h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
        xx = np.ones((h, 1), np.float32) * np.linspace(0, 1, w, dtype=np.float32)[None, :]
        self.yy, self.xx = yy, xx
        # reveal order: the spires first, then the walls, the lower slopes last
        cx, cy = ZOOM_C[0] / W, 250 / H
        dist = np.sqrt(((xx - cx) * 1.3) ** 2 + (yy - cy) ** 2)
        self.reveal = np.clip(0.55 * yy + 0.35 * dist + 0.1 * periodic_noise(h, w, 150, 4), 0, None)
        self.reveal = (self.reveal - self.reveal.min()) / (self.reveal.max() - self.reveal.min())
        self.raven = Raven()
        self.bolts = [bolt_mask(bolt_path((1560, -20), (1440, 520), 11, depth=7), 3),
                      bolt_mask(bolt_path((560, -30), (760, 700), 23, rough=0.32, depth=8), 6)]
        rng = np.random.default_rng(5)
        n = 1100
        self.rain = dict(x=rng.uniform(-300, W + 200, n), y=rng.uniform(0, H + 200, n),
                         v=rng.uniform(1700, 2600, n), L=rng.uniform(18, 46, n), a=rng.uniform(0.08, 0.26, n))
        self.grain = [cv2.GaussianBlur(np.random.default_rng(30 + k).standard_normal((H, W)).astype(np.float32),
                                       (0, 0), 0.9) for k in range(3)]
        self.grain = [g / g.std() for g in self.grain]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.vignette = 1 - 0.28 * (((xx - W / 2) / (W * 0.7)) ** 2 + ((yy - H / 2) / (H * 0.75)) ** 2) ** 1.2
        self.win = np.exp(-(((xx - 926 * W / 1672) / 3.2) ** 2 + ((yy - 318 * H / 940) / 4.5) ** 2))
        self.win_glow = np.exp(-(((xx - 926 * W / 1672) / 16) ** 2 + ((yy - 318 * H / 940) / 18) ** 2))

    # -- layers ---------------------------------------------------------------------------
    def fog(self, t):
        """Fog opacity (full res) and a slightly varying fog tone."""
        d1 = shift(self.n1, -6.0 * t, 1.5 * np.sin(t * 0.3))
        d2 = shift(self.n2, -11.0 * t + 40, 0)
        d3 = shift(self.n3, -17.0 * t, 1.5 * t)
        tex = 0.55 * d1 + 0.3 * d2 + 0.15 * d3
        # the fog that stays: thicker low down, drifting banks
        base = np.clip(0.05 + 0.3 * self.yy ** 2 + 0.13 * tex * (0.4 + self.yy), 0, 0.8)
        # the veil that clears
        p = smooth(span(t, *T_REVEAL))
        edge = self.reveal + 0.12 * tex
        veil = smooth((edge - p * 1.45 + 0.42) / 0.42)
        thick = 0.22 * smooth(span(t, *T_DARKEN)) * np.clip(0.6 + 0.4 * tex, 0, 1)
        a = np.clip(np.maximum(base, veil * 0.97) + thick * (1 - veil), 0, 0.985)
        a = cv2.resize(a, (W, H), interpolation=cv2.INTER_CUBIC)
        tone = cv2.resize(0.74 + 0.05 * d2, (W, H), interpolation=cv2.INTER_LINEAR)
        return np.clip(a, 0, 1), tone

    def ravens(self, t, layer, fps):
        cov = np.zeros((H, W), np.float32)
        for k, fl in enumerate(FLIGHTS):
            if fl['layer'] != layer:
                continue
            a, b = fl['t']
            n = 3
            for j in range(n):
                ts = t + ((j + 0.5) / n - 0.5) / fps
                if ts < a or ts > b:
                    continue
                u = (ts - a) / (b - a)
                p, head = path_pose(fl['P'], u)
                bob = 5 * fl['scale'] * np.sin(2 * np.pi * 0.9 * ts + k)
                flip = np.cos(head) < 0
                tmp = np.zeros((H, W), np.float32)
                self.raven.draw(tmp, p[0], p[1] + bob, fl['scale'], head, fl['ph'] + fl['f'] * (ts - a), flip=flip)
                cov += tmp / n
        return np.clip(cov, 0, 1)

    def flash(self, t):
        f = sum(b['strength'] * flash_env(t, b['t']) for b in BOLTS)
        for tf, s in FLICKERS:
            u = t - tf
            if 0 <= u < 0.6:
                f += s * (np.exp(-u / 0.06) + 0.6 * np.exp(-max(0, u - 0.14) / 0.08) * (u > 0.14))
        return float(min(1.0, f))

    def frame(self, t, fps=60):
        # the photo, pushed in slowly, with the window light
        z = 1 + (PUSH[2] - 1) * smooth(span(t, PUSH[0], PUSH[1]))
        plate = self.photo.copy()
        wl = smooth(span(t, *T_WINDOW))
        M = cv2.getRotationMatrix2D(ZOOM_C, 0, z)
        plate = cv2.warpAffine(plate, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        light = cv2.warpAffine(self.light, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        rgb = np.repeat(plate[..., None], 3, 2)
        # far ravens: in front of the castle, behind the fog
        far = self.ravens(t, 'far', fps)
        rgb = rgb * (1 - far[..., None]) + 0.04 * far[..., None]
        # bolts: behind the castle and the trees, softened by the fog in front
        fog_a, tone = self.fog(t)
        for b, (core, glow) in zip(BOLTS, self.bolts):
            e = flash_env(t, b['t']) * b['strength']
            if e <= 0:
                continue
            vis = light * (1 - 0.75 * fog_a)
            k = np.clip((core * 1.2 + glow * 0.5) * e * vis, 0, 1)[..., None]
            rgb = rgb * (1 - k) + BOLT_COL * k
        # fog
        rgb = rgb * (1 - fog_a[..., None]) + (tone * fog_a)[..., None]
        # the storm comes: everything darkens
        dk = 1 - 0.5 * smooth(span(t, *T_DARKEN))
        rgb = rgb * dk
        # lightning lights the sky and the fog; the castle stays a black silhouette
        f = self.flash(t)
        if f > 0:
            lit = np.clip(light * (1 - fog_a) + fog_a * 0.9, 0, 1)
            rgb = rgb + f * (0.95 - rgb) * lit[..., None] * 0.9
        for b, (core, glow) in zip(BOLTS, self.bolts):
            e = flash_env(t, b['t']) * b['strength']
            if e > 0.05:
                k = np.clip(core * e * light * (1 - 0.6 * fog_a), 0, 1)[..., None]
                rgb = rgb * (1 - k) + k * 1.0
                # a darker, cooler sky around the stroke so a white bolt reads against a lit sky
        # window light
        if wl > 0:
            flick = 0.85 + 0.15 * np.sin(2 * np.pi * 3.1 * t) * np.sin(2 * np.pi * 1.7 * t + 1)
            g = cv2.warpAffine((self.win * 0.75 + self.win_glow * 0.09) * (1 - 0.6 * fog_a), M, (W, H))
            rgb = rgb + (WARM * (g * wl * flick)[..., None])
        # near ravens over everything
        near = self.ravens(t, 'near', fps)
        rgb = rgb * (1 - near[..., None]) + 0.03 * near[..., None]
        # rain
        ra = smooth(span(t, *T_RAIN))
        if ra > 0:
            r = self.rain
            lay = np.zeros((H, W), np.uint8)
            slant = 0.28
            y = (r['y'] + r['v'] * t) % (H + 200) - 100
            x = r['x'] - slant * (r['y'] + r['v'] * t) % (W + 500)
            x = (r['x'] - slant * r['v'] * t) % (W + 500) - 250
            for xi, yi, Li, ai in zip(x, y, r['L'], r['a']):
                c = int(255 * ai * ra * (1 + 2.0 * f))
                cv2.line(lay, (int(xi * 16), int(yi * 16)),
                         (int((xi + slant * Li) * 16), int((yi - Li) * 16)), min(255, c), 1, cv2.LINE_AA, 4)
            rl = lay.astype(np.float32)[..., None] / 255
            rgb = rgb + (0.85 - rgb) * rl
        # grain and vignette
        ph = t * 7.0
        wts = [0.5 + 0.5 * np.cos(2 * np.pi * (ph - j / 3)) for j in range(3)]
        gr = sum(w_ * g_ for w_, g_ in zip(wts, self.grain)) / sum(wts)
        rgb = rgb * self.vignette[..., None] + 0.018 * gr[..., None]
        return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def sheet(film, times, path, cols=4):
    tiles = []
    for t in times:
        im = Image.fromarray(cv2.resize(film.frame(t), (480, 270), interpolation=cv2.INTER_AREA))
        ImageDraw.Draw(im).text((6, 6), f't={t:.2f}', fill=(255, 60, 60))
        tiles.append(np.asarray(im))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)]
    Image.fromarray(np.concatenate(rows, 0)).save(path)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'raven':
        rv = Raven()
        tiles = []
        for i in range(12):
            c = np.zeros((300, 400), np.float32)
            full = np.zeros((H, W), np.float32)
            rv.draw(full, 200, 150, 0.8, 0.0, i / 12)
            tiles.append(((1 - full[:300, :400]) * 235).astype(np.uint8))
        Image.fromarray(np.concatenate([np.concatenate(tiles[:6], 1), np.concatenate(tiles[6:], 1)], 0)).save(sys.argv[2])
        sys.exit()
    film = Film()
    if cmd == 'still':
        Image.fromarray(film.frame(float(sys.argv[2]))).save(sys.argv[3])
    elif cmd == 'sheet':
        a, b, s = map(float, sys.argv[2].split(':'))
        sheet(film, list(np.arange(a, b + 1e-6, s)), sys.argv[3])
    elif cmd == 'render':
        stem = sys.argv[2]
        fps = int(sys.argv[sys.argv.index('--fps') + 1]) if '--fps' in sys.argv else 60
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        src = [ff, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
               '-r', str(fps), '-i', '-']
        jobs = [src + ['-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p',
                       '-movflags', '+faststart', stem + '.mp4']]
        procs = [subprocess.Popen(c, stdin=subprocess.PIPE) for c in jobs]
        n = int(round(DURATION * fps))
        for i in range(n):
            f = film.frame(i / fps, fps)
            for p in procs:
                p.stdin.write(f.tobytes())
        for p in procs:
            p.stdin.close(); p.wait()
        print('wrote', stem + '.mp4', n, 'frames')

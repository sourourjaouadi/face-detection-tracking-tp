"""TD Pratique - Face detection & tracking in videos (classical CV only).

python td.py step1 --video video.mp4 [--compare-alpha]
python td.py step2 --video video.mp4
python td.py step3
python td.py step4 --video video.mp4
python td.py step5 [--desc orb]
python td.py step6 --video video.mp4 --harris-filter
python td.py all   --video video.mp4 --no-show      (runs everything)

Keys in windows: ESC/q = quit, SPACE = pause/resume.
Run step2 BEFORE step3 and step5 (it creates the face crops).
"""
import argparse
import glob
import os

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

OUT = "out"
FACES = os.path.join(OUT, "faces")
os.makedirs(FACES, exist_ok=True)

FACE_SIZE = (64, 64)
DENS_MIN_DEFAULT = 0.0005
FONT = cv2.FONT_HERSHEY_SIMPLEX
SHOW = True
CASCADE = None
EYES = None

K_OPEN = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
K_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
HOG = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)


# ============================================================ utilities ====
def load_cascade(name):
    f = {"default": "haarcascade_frontalface_default.xml",
         "alt2": "haarcascade_frontalface_alt2.xml"}[name]
    c = cv2.CascadeClassifier(cv2.data.haarcascades + f)
    if c.empty():
        raise SystemExit("Cascade not found. Install OpenCV 4.x: pip install \"opencv-python<5\"")
    return c


def has_eyes(gray, box):
    """Second classical check: a real face has an eye in its upper part."""
    x, y, w, h = box
    if w < 80:
        return True                                  # too small for reliable eye detection
    top = cv2.equalizeHist(gray[y:y + int(h * 0.6), x:x + w])
    eyes = EYES.detectMultiScale(top, 1.1, 3, minSize=(int(w * 0.1), int(w * 0.1)),
                                 maxSize=(int(w * 0.4), int(w * 0.4)))
    return len(eyes) >= 1


def fit(img, mw, mh):
    """Downscale for DISPLAY only so the whole frame is visible on screen."""
    h, w = img.shape[:2]
    s = min(mw / w, mh / h, 1.0)
    if s >= 1.0:
        return img
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def show(name, img, a, pos=None, mw=None, mh=None):
    if not SHOW:
        return
    cv2.imshow(name, fit(img, mw or a.max_w, mh or a.max_h))
    if pos:
        cv2.moveWindow(name, *pos)


def wait(a):
    """True if user wants to quit. SPACE pauses."""
    if not SHOW:
        return False
    k = cv2.waitKey(a.delay) & 0xFF
    if k in (27, ord("q")):
        return True
    if k == 32:
        while True:
            k2 = cv2.waitKey(0) & 0xFF
            if k2 == 32:
                break
            if k2 in (27, ord("q")):
                return True
    return False


def put(img, text, org=(5, 20), color=(0, 255, 255)):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.putText(img, text, org, FONT, 0.6, color, 2)
    return img


def draw(frame, boxes, color, label=None, th=2):
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, th)
        if label:
            cv2.putText(frame, label, (int(x), max(12, int(y) - 5)), FONT, .5, color, 1)


def stream(a):
    """Yield (index, frame). Frames wider than --work-w are resized (whole frame kept)."""
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {a.video}")
    a.fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    a.n_frames = n
    W, H = int(cap.get(3)), int(cap.get(4))
    print(f"video: {W}x{H} @ {a.fps:.1f} fps, {n} frames")
    i = 0
    try:
        while True:
            if a.max_frames and i >= a.max_frames:
                break
            ok, f = cap.read()
            if not ok:
                break
            if a.work_w and f.shape[1] > a.work_w:
                s = a.work_w / f.shape[1]
                f = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            if i == 0:
                print(f"working size: {f.shape[1]}x{f.shape[0]}")
            yield i, f
            i += 1
    finally:
        cap.release()


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def contained(a, b):
    """Fraction of the smaller box covered by the other."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return ix * iy / (min(aw * ah, bw * bh) + 1e-9)


def nms(boxes, thr=0.3):
    keep = []
    for b in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
        if all(iou(b, k) < thr and contained(b, k) < 0.6 for k in keep):
            keep.append(b)
    return keep


def pad_box(r, f, W, H):
    x, y, w, h = r
    px, py = int(w * f), int(h * f)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(W, x + w + px), min(H, y + h + py)
    return (x0, y0, x1 - x0, y1 - y0)


# ============================================================== STEP 1 ====
def make_bg():
    return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16,
                                              detectShadows=True)


def raw_mask(bg, frame, lr):
    m = bg.apply(frame, learningRate=lr)
    _, m = cv2.threshold(m, 200, 255, cv2.THRESH_BINARY)   # drop shadows (=127)
    return m


def morph(m0):
    opened = cv2.dilate(cv2.erode(m0, K_OPEN), K_OPEN)             # opening
    closed = cv2.erode(cv2.dilate(opened, K_CLOSE), K_CLOSE)       # closing (bigger, elliptic)
    return opened, closed


def get_rois(mask, smin):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    return [tuple(int(v) for v in stats[i, :4]) for i in range(1, n)
            if stats[i, cv2.CC_STAT_AREA] >= smin]


# ============================================================== STEP 2 ====
def detect_faces(gray, regions, min_face, neighbors, verify=False, max_n=0):
    """Viola-Jones on each region; boxes returned in FULL-FRAME coordinates."""
    faces = []
    for (x, y, w, h) in regions:
        if min(w, h) < min_face:
            continue
        roi = cv2.equalizeHist(gray[y:y + h, x:x + w])
        found = CASCADE.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=neighbors,
                                         minSize=(min_face, min_face))
        for (fx, fy, fw, fh) in found:
            faces.append((x + int(fx), y + int(fy), int(fw), int(fh)))
    faces = nms(faces)
    if verify:
        faces = [f for f in faces if has_eyes(gray, f)]
    if max_n:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:max_n]
    return faces


class Detector:
    """Whole detection chain: MOG2 -> morphology -> ROIs -> Viola-Jones."""

    def __init__(self, a):
        self.a, self.bg, self.i, self.n_fallback = a, make_bg(), 0, 0

    def mask(self, frame):
        lr = -1 if self.i < self.a.warmup else self.a.alpha   # fast learning at start
        self.i += 1
        return raw_mask(self.bg, frame, lr)

    def __call__(self, frame):
        a = self.a
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        m0 = self.mask(frame)
        opened, m = morph(m0)
        rois = get_rois(m, a.smin or int(0.0015 * W * H))
        full = [(0, 0, W, H)]
        if a.full_frame:
            search = full
        else:
            search = [pad_box(r, a.pad, W, H) for r in rois]
            if not search and not a.strict:          # no motion: still look everywhere
                search = full
                self.n_fallback += 1
        faces = detect_faces(gray, search, a.min_face, a.neighbors,
                             not a.no_verify, a.max_per_frame)
        return dict(gray=gray, m0=m0, opened=opened, mask=m, rois=rois, faces=faces)


def hog_vis(gray64, cell=8, nbins=9, scale=8):
    """HOG visualisation: one oriented-line glyph per 8x8 cell."""
    g = np.float32(gray64)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = ang % 180
    h, w = g.shape
    canvas = np.zeros((h * scale, w * scale), np.uint8)
    for i in range(h // cell):
        for j in range(w // cell):
            sl = (slice(i * cell, (i + 1) * cell), slice(j * cell, (j + 1) * cell))
            hist, _ = np.histogram(ang[sl].ravel(), bins=nbins, range=(0, 180),
                                   weights=mag[sl].ravel())
            hist = hist / (hist.max() + 1e-6)
            cx, cy = (j + .5) * cell * scale, (i + .5) * cell * scale
            for b in range(nbins):
                th = np.deg2rad((b + .5) * 180 / nbins + 90)   # edge = gradient + 90 deg
                dx, dy = np.cos(th), np.sin(th)
                r = hist[b] * cell * scale / 2
                cv2.line(canvas, (int(cx - dx * r), int(cy - dy * r)),
                         (int(cx + dx * r), int(cy + dy * r)), 255, 1)
    return canvas


def face_crop(gray, box):
    x, y, w, h = box
    return cv2.equalizeHist(cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE))


# ============================================================== STEP 4 ====
def harris_density(gray_box):
    """Harris response map -> number of corner blobs / ROI area."""
    if gray_box.shape[0] < 8 or gray_box.shape[1] < 8:
        return 0.0, None
    g = np.float32(gray_box) / 255.0
    r = cv2.cornerHarris(g, 2, 3, 0.04)
    thr = max(0.01 * float(r.max()), 1e-4)            # relative + absolute threshold
    n, _ = cv2.connectedComponents(np.uint8(r > thr))
    return (n - 1) / float(gray_box.shape[0] * gray_box.shape[1]), r


def get_dens_min(a):
    if a.dens_min is not None:
        return a.dens_min
    p = f"{OUT}/s4_dens_min.txt"
    if os.path.exists(p):
        return float(open(p).read())
    return DENS_MIN_DEFAULT


# ================================================================ RUNS =====
def run_step1(a):
    det = Detector(a)
    if a.snap < 0:                                    # -1 = middle of the video
        cap = cv2.VideoCapture(a.video)
        a.snap = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2
        cap.release()
    print(f"snapshot will be taken at frame {a.snap}")
    alphas = [0.001, 0.005, 0.05]
    cmp_bgs = {al: make_bg() for al in alphas} if a.compare_alpha else {}
    last, saved = None, False

    def save(frame, m0, opened, m, vis, extra=None):
        cv2.imwrite(f"{OUT}/s1_M0.png", m0)
        cv2.imwrite(f"{OUT}/s1_opened.png", opened)
        cv2.imwrite(f"{OUT}/s1_closed.png", m)
        cv2.imwrite(f"{OUT}/s1_rois.png", vis)
        panels = [put(fit(x, 480, 270).copy(), t) for x, t in
                  [(m0, "M0 raw"), (opened, "opening"), (m, "closing"), (vis, "ROIs")]]
        cv2.imwrite(f"{OUT}/s1_panel.png", np.hstack(panels))
        if extra is not None:
            cv2.imwrite(f"{OUT}/s1_alpha_compare.png", extra)
        print("snapshots saved in out/ (s1_panel.png is the report figure)")

    for i, frame in stream(a):
        H, W = frame.shape[:2]
        m0 = det.mask(frame)
        opened, m = morph(m0)
        rois = get_rois(m, a.smin or int(0.0015 * W * H))
        vis = frame.copy()
        draw(vis, rois, (0, 255, 0))
        put(vis, f"frame {i}  ROIs: {len(rois)}")
        extra = None
        if cmp_bgs:
            ms = [raw_mask(bg, frame, -1 if i < a.warmup else al) for al, bg in cmp_bgs.items()]
            extra = np.hstack([put(fit(mk, 480, 270).copy(), f"alpha={al}")
                               for mk, al in zip(ms, cmp_bgs)])
            show("alpha compare", extra, a, mw=1440, mh=270)
        last = (frame, m0, opened, m, vis, extra)
        if i == a.snap:
            save(*last)
            saved = True
        show("M0 (raw)", m0, a, (0, 0), 640, 360)
        show("opening", opened, a, (650, 0), 640, 360)
        show("closing = cleaned", m, a, (0, 400), 640, 360)
        show("ROIs", vis, a, (650, 400), 640, 360)
        if wait(a):
            break
    if not saved and last:
        save(*last)


def run_step2(a):
    for f in glob.glob(f"{FACES}/*.png"):
        os.remove(f)                                   # fresh dataset for this video
    det, saved, crops, hog_done = Detector(a), 0, [], False
    people = cv2.HOGDescriptor()
    people.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    n_people = 0
    for i, frame in stream(a):
        r = det(frame)
        gray = r["gray"]
        vis = frame.copy()
        draw(vis, r["rois"], (0, 255, 0))
        draw(vis, r["faces"], (0, 0, 255), "VJ")
        if not a.no_people and i % 2 == 0:             # HOG + linear SVM on ROIs
            for (x, y, w, h) in r["rois"]:
                if w >= 64 and h >= 128:
                    sc = min(1.0, 640.0 / h)                     # speed + fewer false hits
                    roi = cv2.resize(frame[y:y + h, x:x + w], None, fx=sc, fy=sc)
                    if roi.shape[0] < 128 or roi.shape[1] < 64:
                        continue
                    rects, _ = people.detectMultiScale(roi, hitThreshold=a.hog_thr,
                                                       winStride=(8, 8), padding=(8, 8),
                                                       scale=1.05, groupThreshold=2)
                    cand = [(int(px / sc), int(py / sc), int(pw / sc), int(ph / sc))
                            for (px, py, pw, ph) in rects]
                    for (px, py, pw, ph) in nms(cand):
                        draw(vis, [(x + px, y + py, pw, ph)], (255, 0, 0), "HOG")
                        n_people += 1
        for box in r["faces"]:
            f64 = face_crop(gray, box)
            if not hog_done:
                d = HOG.compute(f64).ravel()
                print("HOG face descriptor length:", d.size)          # 1764
                hog_done = True
            if i % a.stride == 0 and saved < a.max_faces:
                cv2.imwrite(f"{FACES}/f{saved:04d}.png", f64)
                saved += 1
                if len(crops) < 6:
                    crops.append(f64)
        put(vis, f"frame {i}  ROIs:{len(r['rois'])}  faces:{len(r['faces'])}  saved:{saved}")
        if crops:
            show("HOG vis (first face)", hog_vis(crops[0]), a, (0, 0), 512, 512)
        show("Step2: ROI (green) + Viola-Jones (red) + HOG people (blue)", vis, a)
        if wait(a):
            break
    if crops:
        row1 = np.hstack([cv2.resize(c, (256, 256)) for c in crops])
        row2 = np.hstack([cv2.resize(hog_vis(c), (256, 256)) for c in crops])
        cv2.imwrite(f"{OUT}/s2_hog_montage.png", np.vstack([row1, row2]))
    print(f"{saved} face crops saved in {FACES}  |  fallback frames: {det.n_fallback}"
          f"  |  HOG people hits: {n_people}")
    if saved < 10:
        print("WARNING: <10 faces. Try --full-frame, --neighbors 3, --min-face 20, or a better video.")


def load_faces():
    files = sorted(glob.glob(f"{FACES}/*.png"))
    if len(files) < 10:
        raise SystemExit("Need >=10 face crops in out/faces: run step2 first.")
    X = np.array([cv2.imread(f, 0).ravel() for f in files], np.float32) / 255.0
    return X, files


def tile(img_f, label):
    im = np.uint8(np.clip(img_f.reshape(FACE_SIZE[::-1]), 0, 1) * 255)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    cv2.putText(im, label, (2, 10), FONT, .35, (0, 255, 255), 1)
    return im


def plot_curve(xs, ys, path, title, ylabel):
    W, H = 640, 360
    img = np.full((H, W, 3), 255, np.uint8)
    x0, y0, x1, y1 = 70, 35, W - 20, H - 45
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 1)
    vmax = max(ys) * 1.05 + 1e-12
    pts = []
    for n, (k, v) in enumerate(zip(xs, ys)):
        px = int(x0 + (x1 - x0) * n / max(1, len(xs) - 1))
        py = int(y1 - (y1 - y0) * v / vmax)
        pts.append((px, py))
        cv2.circle(img, (px, py), 3, (0, 0, 255), -1)
        cv2.putText(img, str(k), (px - 8, y1 + 15), FONT, .38, (0, 0, 0), 1)
    cv2.polylines(img, [np.array(pts, np.int32)], False, (255, 0, 0), 1)
    cv2.putText(img, title, (x0, 20), FONT, .5, (0, 0, 0), 1)
    cv2.putText(img, f"{vmax:.4f}", (2, y0 + 5), FONT, .4, (0, 0, 0), 1)
    cv2.putText(img, "0", (50, y1), FONT, .4, (0, 0, 0), 1)
    cv2.putText(img, "k (number of eigenfaces)", (x0 + 150, H - 8), FONT, .45, (0, 0, 0), 1)
    cv2.putText(img, ylabel, (2, y0 - 12), FONT, .4, (0, 0, 0), 1)
    cv2.imwrite(path, img)


def run_step3(a):
    X, _ = load_faces()
    n_comp = min(100, len(X) - 1)
    pca = PCA(n_components=n_comp).fit(X)
    mean, comps = pca.mean_, pca.components_
    cum = np.cumsum(pca.explained_variance_ratio_)
    ks = sorted({k for k in [1, 2, 5, 10, 20, 30, 50, n_comp] if k <= n_comp})
    print("cumulative explained variance:",
          {k: round(float(cum[k - 1]), 3) for k in ks})

    eig = [tile((c - c.min()) / (c.max() - c.min() + 1e-9), f"e{i}")
           for i, c in enumerate(comps[:15])]
    tiles = [tile(mean, "mean")] + eig                  # works for any number of faces
    while len(tiles) % 8:
        tiles.append(np.zeros_like(tiles[0]))
    eig_img = np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)])
    cv2.imwrite(f"{OUT}/s3_eigenfaces.png", eig_img)

    coefs_all = pca.transform(X)
    mses = []
    for k in ks:                                         # quality vs k over ALL faces
        rec = mean + coefs_all[:, :k] @ comps[:k]
        mses.append(float(np.mean((rec - X) ** 2)))
    print("k    MSE       PSNR(dB)")
    for k, m in zip(ks, mses):
        print(f"{k:<4d} {m:.6f}  {10 * np.log10(1.0 / max(m, 1e-12)):.1f}")
    plot_curve(ks, mses, f"{OUT}/s3_mse_vs_k.png", "Reconstruction error vs k", "mean MSE")

    rows = []
    for idx in sorted({0, len(X) // 2, len(X) - 1}):
        row = [tile(X[idx], "orig")]
        for k in ks:
            row.append(tile(mean + coefs_all[idx, :k] @ comps[:k], f"k={k}"))
        rows.append(np.hstack(row))
    rec_img = np.vstack(rows)
    cv2.imwrite(f"{OUT}/s3_reconstruction.png", rec_img)
    print("saved: s3_eigenfaces.png, s3_reconstruction.png, s3_mse_vs_k.png")

    if SHOW:
        up = lambda im: cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        show("Eigenfaces (mean + first 15)", up(eig_img), a, (0, 0), 1100, 500)
        show("Reconstruction: orig | k=1 ... k=max", up(rec_img), a, (0, 520), 1300, 450)
        show("MSE vs k", cv2.imread(f"{OUT}/s3_mse_vs_k.png"), a, (1120, 0), 640, 360)
        print("Press any key in a window to close.")
        cv2.waitKey(0)


def run_step4(a):
    det, dens_min = Detector(a), get_dens_min(a)
    rng = np.random.default_rng(0)
    d_face, d_roi, d_bg = [], [], []
    for i, frame in stream(a):
        r = det(frame)
        gray, vis = r["gray"], frame.copy()
        H, W = gray.shape
        for (x, y, w, h) in r["rois"]:
            d, _ = harris_density(gray[y:y + h, x:x + w])
            d_roi.append(d)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 1)
            cv2.putText(vis, f"ROI {d:.4f}", (x, y + 12), FONT, .4, (255, 255, 0), 1)
        for (x, y, w, h) in r["faces"]:
            d, _ = harris_density(gray[y:y + h, x:x + w])
            d_face.append(d)
            keep = d >= dens_min
            col = (0, 255, 0) if keep else (0, 0, 255)
            cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)
            cv2.putText(vis, f"{d:.4f} {'KEEP' if keep else 'REJECT'}", (x, max(12, y - 5)),
                        FONT, .5, col, 1)
            for _ in range(15):                          # same-size background patch (control)
                if w >= W or h >= H:
                    break
                bx, by = int(rng.integers(0, W - w)), int(rng.integers(0, H - h))
                if not r["mask"][by:by + h, bx:bx + w].any():
                    d_bg.append(harris_density(gray[by:by + h, bx:bx + w])[0])
                    break
        put(vis, f"frame {i}  Harris density min={dens_min:.5f}")
        show("Step4 Harris density (green keep / red reject / cyan ROI)", vis, a)
        if wait(a):
            break

    def st(v):
        return f"mean={np.mean(v):.5f} std={np.std(v):.5f} n={len(v)}" if v else "n=0"
    print("faces      :", st(d_face))
    print("ROIs       :", st(d_roi))
    print("background :", st(d_bg))
    if d_face and d_bg:
        lo_face, hi_bg = float(np.percentile(d_face, 5)), float(np.percentile(d_bg, 95))
        if lo_face > hi_bg:
            thr = (lo_face + hi_bg) / 2
            print(f"classes separable -> DENS_MIN = {thr:.5f} (saved; step6 uses it automatically)")
        else:
            thr = 0.0                                  # do not reject real faces
            print("classes OVERLAP -> density alone is a weak filter: threshold set to 0 "
                  "(filter inactive). Pass --dens-min X to force one. Say this in the report.")
        open(f"{OUT}/s4_dens_min.txt", "w").write(f"{thr:.6f}")


def run_step5(a):
    X, _ = load_faces()
    imgs = [cv2.resize(np.uint8(x.reshape(FACE_SIZE[::-1]) * 255), (128, 128)) for x in X]
    if a.desc == "orb":
        ext, dim = cv2.ORB_create(edgeThreshold=15, patchSize=15), 32
        make_kps = lambda: [cv2.KeyPoint(float(x), float(y), 15)
                            for y in range(16, 113, 8) for x in range(16, 113, 8)]
    else:
        ext, dim = cv2.SIFT_create(), 128
        make_kps = lambda: [cv2.KeyPoint(float(x), float(y), 16)       # dense grid
                            for y in range(8, 121, 8) for x in range(8, 121, 8)]
    descs = []
    for im in imgs:
        _, d = ext.compute(im, make_kps())
        descs.append(np.zeros((0, dim), np.float32) if d is None else d.astype(np.float32))
    allD = np.vstack(descs)
    print(f"{a.desc.upper()} descriptors:", allD.shape)
    K = a.K
    if len(allD) < K:
        raise SystemExit(f"Only {len(allD)} descriptors for K={K}: need more faces.")
    km = KMeans(n_clusters=K, n_init=3, random_state=0).fit(allD)
    H = np.array([np.bincount(km.predict(d) if len(d) else [], minlength=K) for d in descs],
                 np.float32)
    H /= np.maximum(H.sum(1, keepdims=True), 1)                    # frequency histograms
    np.save(f"{OUT}/s5_hists.npy", H)
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)
    sim = Hn @ Hn.T
    nn = [int(j) for j in np.argsort(-sim[0])[1:4]]
    print("vocabulary size K =", K, "| histogram matrix:", H.shape)
    print("nearest faces to face 0 (cosine):", [(j, round(float(sim[0, j]), 3)) for j in nn])
    print("most similar pair (cosine): %.3f | least similar: %.3f"
          % (np.max(sim - 2 * np.eye(len(sim))), np.min(sim)))

    bars = np.zeros((220, K * 6, 3), np.uint8)
    for k, v in enumerate(H[0] / (H[0].max() + 1e-9)):
        cv2.rectangle(bars, (k * 6, 210), (k * 6 + 4, 210 - int(v * 195)), (0, 255, 0), -1)
    cv2.putText(bars, "BoF histogram of face 0 (100 visual words)", (5, 15), FONT, .5,
                (255, 255, 255), 1)
    cv2.imwrite(f"{OUT}/s5_hist_face0.png", bars)
    row = np.hstack([cv2.resize(np.uint8(X[j].reshape(FACE_SIZE[::-1]) * 255), (128, 128))
                     for j in [0] + nn])
    cv2.imwrite(f"{OUT}/s5_neighbors.png", row)
    print("saved: s5_hist_face0.png, s5_neighbors.png (face 0 + 3 nearest), s5_hists.npy")


# ============================================================== STEP 6 ====
class Track:
    _next = 0

    def __init__(self, box):
        x, y, w, h = box
        cx, cy = x + w / 2, y + h / 2
        kf = cv2.KalmanFilter(4, 2)                              # state [x y vx vy]
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                        [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.processNoiseCov = np.diag([1e-2, 1e-2, 1e-1, 1e-1]).astype(np.float32)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
        self.kf, self.w, self.h = kf, float(w), float(h)
        self.id, self.missed, self.hits = Track._next, 0, 1
        self.pos, self.trail = (cx, cy), []
        Track._next += 1

    def predict(self):
        p = self.kf.predict()
        self.pos = (float(p[0, 0]), float(p[1, 0]))

    def update(self, box):
        x, y, w, h = box
        self.kf.correct(np.array([[x + w / 2], [y + h / 2]], np.float32))
        s = self.kf.statePost
        self.pos = (float(s[0, 0]), float(s[1, 0]))
        self.w, self.h = 0.7 * self.w + 0.3 * w, 0.7 * self.h + 0.3 * h
        self.missed, self.hits = 0, self.hits + 1

    def box(self):
        return (int(self.pos[0] - self.w / 2), int(self.pos[1] - self.h / 2),
                int(self.w), int(self.h))


def associate(tracks, dets, gate):
    """Hungarian assignment on centre distance, gated by distance (scaled with face size)."""
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))
    cost = np.array([[np.hypot(t.pos[0] - (d[0] + d[2] / 2), t.pos[1] - (d[1] + d[3] / 2))
                      for d in dets] for t in tracks])
    r, c = linear_sum_assignment(cost)
    matches = [(i, j) for i, j in zip(r, c)
               if cost[i, j] <= max(gate, 1.5 * max(tracks[i].w, tracks[i].h))]
    mt, md = {m[0] for m in matches}, {m[1] for m in matches}
    return (matches, [i for i in range(len(tracks)) if i not in mt],
            [j for j in range(len(dets)) if j not in md])


def run_step6(a):
    det, dens_min = Detector(a), get_dens_min(a)
    Track._next = 0
    tracks, writer, log, n_rej, created = [], None, [], 0, 0
    for i, frame in stream(a):
        r = det(frame)
        gray, faces = r["gray"], r["faces"]
        if a.harris_filter:
            kept = [b for b in faces
                    if harris_density(gray[b[1]:b[1] + b[3], b[0]:b[0] + b[2]])[0] >= dens_min]
            n_rej += len(faces) - len(kept)
            faces = kept

        for t in tracks:
            t.predict()
        matches, ut, ud = associate(tracks, faces, a.gate)
        for ti, dj in matches:
            if tracks[ti].missed > 0 and tracks[ti].hits >= a.confirm:
                log.append(f"frame {i}: ID{tracks[ti].id} re-acquired after {tracks[ti].missed} missed")
            tracks[ti].update(faces[dj])
        for ti in ut:
            tracks[ti].missed += 1
            if tracks[ti].missed == 1 and tracks[ti].hits >= a.confirm:
                log.append(f"frame {i}: ID{tracks[ti].id} not detected -> Kalman prediction")
        for dj in ud:
            t = Track(faces[dj])
            tracks.append(t)
            created += 1
        for t in tracks:
            if t.missed > a.M and t.hits >= a.confirm:
                log.append(f"frame {i}: ID{t.id} deleted (missed > M={a.M})")
        tracks = [t for t in tracks if t.missed <= a.M]

        vis = frame.copy()
        draw(vis, faces, (0, 0, 255), th=1)                       # raw detections (red)
        for t in tracks:
            if t.hits < a.confirm:
                continue                                          # tentative track
            t.trail.append((int(t.pos[0]), int(t.pos[1])))
            t.trail = t.trail[-40:]
            if t.missed == 0:
                col, tag = (0, 255, 0), f"ID{t.id}"
            elif t.missed <= a.N:
                col, tag = (0, 255, 255), f"ID{t.id} pred({t.missed})"   # occlusion
            else:
                continue                                          # N < missed <= M: silent
            draw(vis, [t.box()], col, tag)
            for p, q in zip(t.trail[:-1], t.trail[1:]):
                cv2.line(vis, p, q, col, 1)
        active = sum(1 for t in tracks if t.hits >= a.confirm and t.missed <= a.N)
        put(vis, f"frame {i}  active tracks: {active}  green=detected yellow=predicted red=raw")

        if writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(f"{OUT}/s6_tracking.mp4",
                                     cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
        writer.write(vis)
        show("Step6 tracking", vis, a)
        if wait(a):
            break
    if writer:
        writer.release()
    open(f"{OUT}/s6_log.txt", "w").write("\n".join(log))
    print(f"tracks created: {created} | harris-rejected detections: {n_rej} "
          f"| fallback frames: {det.n_fallback}")
    print("video: out/s6_tracking.mp4 | lifecycle events: out/s6_log.txt")


# ================================================================== CLI ====
def main():
    global SHOW, CASCADE, EYES
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=[f"step{i}" for i in range(1, 7)] + ["all"])
    p.add_argument("--video", default="video.mp4")
    # display / speed
    p.add_argument("--max-w", type=int, default=1280, help="display fit width")
    p.add_argument("--max-h", type=int, default=720, help="display fit height")
    p.add_argument("--work-w", type=int, default=640, help="resize frames wider than this (faster)")
    p.add_argument("--delay", type=int, default=1, help="waitKey ms (30 = slower playback)")
    p.add_argument("--no-show", action="store_true", help="headless, save outputs only")
    p.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    # step 1
    p.add_argument("--alpha", type=float, default=0.005, help="MOG2 learningRate")
    p.add_argument("--warmup", type=int, default=30, help="frames with automatic learning rate")
    p.add_argument("--smin", type=int, default=0, help="min area px (0 = 0.15%% of frame)")
    p.add_argument("--snap", type=int, default=-1, help="frame to save (-1 = middle of video)")
    p.add_argument("--compare-alpha", action="store_true")
    # detection
    p.add_argument("--full-frame", action="store_true", help="Viola-Jones on the whole frame")
    p.add_argument("--strict", action="store_true", help="no full-frame fallback if no motion")
    p.add_argument("--pad", type=float, default=0.15, help="ROI padding ratio")
    p.add_argument("--min-face", type=int, default=80, help="min face size px")
    p.add_argument("--neighbors", type=int, default=6)
    p.add_argument("--cascade", choices=["default", "alt2"], default="alt2")
    p.add_argument("--no-verify", action="store_true", help="disable eye check on faces")
    p.add_argument("--max-per-frame", type=int, default=0, help="keep N largest faces (0=all)")
    p.add_argument("--hog-thr", type=float, default=0.8, help="HOG people SVM threshold")
    p.add_argument("--no-people", action="store_true", help="skip HOG people detector (step2)")
    p.add_argument("--stride", type=int, default=2, help="save a face crop every N frames")
    p.add_argument("--max-faces", type=int, default=300)
    # step 4/5
    p.add_argument("--dens-min", type=float, default=None)
    p.add_argument("--desc", choices=["sift", "orb"], default="sift")
    p.add_argument("--K", type=int, default=100)
    # step 6
    p.add_argument("--harris-filter", action="store_true")
    p.add_argument("--N", type=int, default=5, help="frames drawn as Kalman prediction")
    p.add_argument("--M", type=int, default=30, help="frames before deletion (M>N)")
    p.add_argument("--gate", type=float, default=100)
    p.add_argument("--confirm", type=int, default=3, help="hits before a track is shown")
    a = p.parse_args()
    if a.M <= a.N:
        raise SystemExit("Need M > N")

    SHOW = not a.no_show
    CASCADE = load_cascade(a.cascade)
    EYES = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    steps = [f"step{i}" for i in range(1, 7)] if a.step == "all" else [a.step]
    for s in steps:
        print(f"\n===== {s} =====")
        globals()["run_" + s](a)
        if SHOW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
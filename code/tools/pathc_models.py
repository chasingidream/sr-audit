# -*- coding: utf-8 -*-
"""Path C：Δ_floor 可预测性 —— 非线性模型 + 留站点泛化。

目标：把 R² 从线性基准（0.49 / 空间 CV 0.459）推向 0.4+ 的**诚实泛化**口径。

模型阶梯（全部零依赖，仅 numpy）
  1. OLS（基准）
  2. 岭回归（L2）
  3. 特征工程 + 交互项 + 二次项，再岭
  4. 梯度提升（自实现 CART 回归树 + 随机森林/GBDT，纯 numpy）
  5. KNN（局部记忆，检验是否有可用的空间邻域信号）

诚实性口径
  * 空间留块 CV（Δlat×Δlon，含"一个大陆"级别的块 30x60）
  * 留采集战役 CV（leave-one-campaign-out）：全球制图真正的泛化语境
  * 同时报两个因变量：delta_floor（dB）与 corr_bic_hr（尺度无关）
"""
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np

# 空间留块 CV 的块数随分辨率急剧上升（5x5 度 → 663 块）；GBDT 要逐折重训，
# 663 折 × 150 树在 Windows 进程池下会崩。故对树模型只跑到 20x20（88 块）为止，
# 5x5 仅对闭式解模型（ridge/knn）执行。这与"空间 CV 要看块尺度敏感性"的结论不冲突：
# 关键是粗块（30x60 = 大陆级）能不能站住。
GBDT_MAX_BLOCKS = 120
_NW = max(1, min(6, (os.cpu_count() or 4)))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(ROOT, "experiments")
META = os.path.join(ROOT, "data", "raw", "metadata.csv")
CAMP_RE = re.compile(r"^(ASMSpotter|Amnesty POI|Landcover|UNHCR)")


def campaign(cov):
    m = CAMP_RE.match(str(cov).strip())
    return m.group(1) if m else "OTHER"


def load():
    import csv
    q = {}
    for line in open(os.path.join(EXP, "recon_quality.jsonl"), encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            q[r["cov"]] = r
    meta = {}
    with open(META, "r", encoding="utf-8", errors="replace", newline="") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        idx = {n: i for i, n in enumerate(hdr)}
        for row in rd:
            if not row:
                continue
            sid = row[0].strip()
            if not sid or sid in meta:
                continue
            rec = {}
            for key, col in [("area_km2", "area"), ("lccs", "LCCS")]:
                try:
                    rec[key] = float(row[idx[col]])
                except (KeyError, ValueError, IndexError):
                    pass
            try:
                b = json.loads(row[idx["bounds"]])
                rec["lon_span"] = abs(b[2] - b[0])
                rec["lat_span"] = abs(b[3] - b[1])
            except (KeyError, ValueError, IndexError, json.JSONDecodeError):
                pass
            meta[sid] = rec
    d = json.load(open(os.path.join(EXP, "stratified_floor_ws.json"), encoding="utf-8"))
    rows = []
    for s in d["scenes"]:
        r = dict(s)
        r["campaign"] = campaign(s["cov"])
        qq = q.get(s["cov"])
        if qq:
            r.update({k: qq[k] for k in
                      ("psnr_bic", "psnr_mean", "var_hr_n", "corr_bic_hr", "contrast_lr")})
        r.update(meta.get(s["cov"], {}))
        rows.append(r)
    return rows


# ---------------------------------------------------------------- 特征工程
# ⚠ 代数耦合黑名单：Δ_floor = PSNR(bic) − PSNR(mean)，而
#   PSNR(mean) = −10·log10 VAR(HR)。实测 corr(var_hr_n, −psnr_mean) = 1.000
#   ⇒ var_hr_n / log_var_hr / log_psnr_mean 是**恒等式级**的伪预测子，
#     用它们能让 R² 虚高到 0.69，但那是把因变量的一项搬进自变量。
#   全球制图的诚实口径必须把它们排除（scale-free 复核）。
LEAKY = {"var_hr_n", "log_var_hr", "log_psnr_mean", "psnr_bic", "psnr_mean", "delta_floor"}
BASE = ["ndvi_mean", "ndvi_std", "bright_lr", "spec_het", "bright_hr", "texture_hr",
        "abs_lat", "cloud", "log_area", "log_lccs", "log_tex",
        "log_contrast", "abs_dt_log", "lon_span", "lat_span"]


def raw_feats(r):
    d = {}
    for k in ["ndvi_mean", "ndvi_std", "bright_lr", "spec_het", "bright_hr", "texture_hr",
              "cloud", "lat", "lon", "delta_days"]:
        v = r.get(k)
        if v is None:
            return None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(v):
            return None
        d[k] = v
    for k in ["area_km2", "lccs"]:
        v = r.get(k)
        d[k] = np.nan if v is None else float(v)
    # 仅供 include_leaky=True 使用（与 Δ_floor 代数耦合，默认剔除）
    v = r.get("var_hr_n")
    d["log_var_hr"] = (np.log10(v) if (v is not None and v > 0) else np.nan)
    return d


def engineer(d, level=2, include_leaky=False):
    """level 0=原始; 1=+log/abs; 2=+二次与关键交互。

    include_leaky=False 时剔除与 Δ_floor 代数耦合的项（见 LEAKY 注释）。
    """
    f = {}
    lat, lon = d["lat"], d["lon"]
    f["ndvi_mean"] = d["ndvi_mean"]
    f["ndvi_std"] = d["ndvi_std"]
    f["bright_lr"] = d["bright_lr"]
    f["spec_het"] = d["spec_het"]
    f["bright_hr"] = d["bright_hr"]
    tex = max(d["texture_hr"], 1e-12)
    f["log_tex"] = np.log10(tex)
    f["abs_lat"] = abs(lat)
    f["cloud"] = d["cloud"]
    f["abs_dt_log"] = np.log10(abs(d["delta_days"]) + 1.0)
    f["lon"] = lon
    f["log_area"] = np.log10(max(d["area_km2"], 1e-3)) if np.isfinite(d["area_km2"]) else np.nan
    f["log_lccs"] = np.log10(max(d["lccs"], 1.0)) if np.isfinite(d["lccs"]) else np.nan
    if include_leaky:
        f["log_var_hr"] = d.get("log_var_hr", np.nan)
    if level >= 2:
        f["lat_x_dt"] = abs(lat) * f["abs_dt_log"]
        f["tex_x_dt"] = f["log_tex"] * f["abs_dt_log"]
        f["ndvi_x_tex"] = d["ndvi_mean"] * f["log_tex"]
        f["bright_x_tex"] = d["bright_hr"] * f["log_tex"]
        f["ndvi_sq"] = d["ndvi_mean"] ** 2
        f["tex_sq"] = f["log_tex"] ** 2
        f["bright_hr_sq"] = d["bright_hr"] ** 2
        f["cloud_x_ndvi"] = d["cloud"] * d["ndvi_mean"]
        f["spec_x_tex"] = d["spec_het"] * f["log_tex"]
        f["ndvistd_x_tex"] = d["ndvi_std"] * f["log_tex"]
    return f


def build(rows, y, level, extra=(), include_leaky=False):
    recs = []
    for r in rows:
        d = raw_feats(r)
        if d is None:
            recs.append(None)
            continue
        f = engineer(d, level, include_leaky=include_leaky)
        f.update({k: r.get(k) for k in extra})
        recs.append(f)
    # keys 取所有样本字段的并集（否则第一个样本的 NaN 会漏掉合法特征）
    keys = None
    for f in recs:
        if f is None:
            continue
        ks = [k for k, v in f.items() if v is not None]
        keys = ks if keys is None else [k for k in keys if k in ks]
    X = np.full((len(recs), len(keys)), np.nan)
    for i, f in enumerate(recs):
        if f is None:
            continue
        for j, k in enumerate(keys):
            v = f.get(k)
            if v is not None and np.isfinite(v):
                X[i, j] = v
    m = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[m], y[m], np.where(m)[0], keys


# ---------------------------------------------------------------- 模型
def standardize(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def fit_ridge(Xtr, ytr, lam):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Z = (Xtr - mu) / sd
    n, p = Z.shape
    yc = ytr - ytr.mean()
    A = Z.T @ Z + lam * np.eye(p)
    b = np.linalg.solve(A, Z.T @ yc)
    return mu, sd, b, ytr.mean()


def pred_ridge(m, Xte):
    mu, sd, b, y0 = m
    return ((Xte - mu) / sd) @ b + y0


class Tree:
    """极简 CART 回归树（平方误差），用于 GBDT。"""

    __slots__ = ("f", "t", "l", "r", "v")

    def __init__(self):
        self.f = self.t = self.l = self.r = None
        self.v = 0.0

    def fit(self, X, y, depth, min_leaf, max_feat, rng):
        self.v = float(y.mean())
        if depth <= 0 or len(y) < 2 * min_leaf:
            return self
        n, p = X.shape
        best = None
        feats = rng.choice(p, size=max(1, min(max_feat, p)), replace=False) \
            if max_feat and max_feat < p else np.arange(p)
        # 向量化分裂搜索：对每个候选特征，一次性扫完 9 个分位阈值
        for j in feats:
            col = X[:, j]
            qs = np.unique(np.percentile(col, np.linspace(10, 90, 9)))
            if len(qs) == 0:
                continue
            m = col[:, None] <= qs[None, :]                 # (n, nq) 布尔
            nl = m.sum(0)
            nr = n - nl
            ok = (nl >= min_leaf) & (nr >= min_leaf)
            if not ok.any():
                continue
            yc = y.astype(np.float64)
            sl = np.where(m, yc[:, None], 0.0).sum(0)
            s2l = np.where(m, (yc ** 2)[:, None], 0.0).sum(0)
            sr, s2r = yc.sum() - sl, (yc ** 2).sum() - s2l
            sse = (s2l - sl * sl / np.maximum(nl, 1)) + \
                  (s2r - sr * sr / np.maximum(nr, 1))
            sse = np.where(ok, sse, np.inf)
            k = int(np.argmin(sse))
            if best is None or sse[k] < best[0]:
                best = (float(sse[k]), int(j), float(qs[k]), col <= qs[k])
        if best is None:
            return self
        _, j, t, m = best
        self.f, self.t = int(j), t
        self.l = Tree().fit(X[m], y[m], depth - 1, min_leaf, max_feat, rng)
        self.r = Tree().fit(X[~m], y[~m], depth - 1, min_leaf, max_feat, rng)
        return self

    def predict(self, X):
        out = np.empty(len(X))
        for i in range(len(X)):
            nd = self
            while nd.f is not None:
                nd = nd.l if X[i, nd.f] <= nd.t else nd.r
            out[i] = nd.v
        return out


def fit_gbdt(Xtr, ytr, n_trees=200, lr=0.06, depth=4, min_leaf=20,
             subsample=0.7, max_feat=0.6, seed=0):
    rng = np.random.RandomState(seed)
    p = Xtr.shape[1]
    mf = max(1, int(round(max_feat * p)))
    base = float(ytr.mean())
    pred = np.full(len(ytr), base)
    trees = []
    # 预重采样索引（每棵树共享同一套 bootstrap 比例，但独立抽）
    for _ in range(n_trees):
        resid = ytr - pred
        idx = rng.choice(len(ytr), size=max(1, int(subsample * len(ytr))), replace=False)
        t = Tree().fit(Xtr[idx], resid[idx], depth, min_leaf, mf, rng)
        pred += lr * t.predict(Xtr)
        trees.append(t)
    return base, trees, lr

def pred_gbdt(m, X):
    base, trees, lr = m
    out = np.full(len(X), base)
    for t in trees:
        out += lr * t.predict(X)
    return out


def r2_of(y, pred):
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def _fold_worker(job):
    """(Xtr, ytr, Xte) → 预测。用于进程池并行各折。"""
    Xtr, ytr, Xte, kind, kw = job
    if kind == "ridge":
        lam = kw.get("lam", 1.0)
        return pred_ridge(fit_ridge(Xtr, ytr, lam), Xte)
    if kind == "knn":
        return naive_knn(kw.get("k", 15))(Xtr, ytr, Xte)
    if kind == "gbdt":
        return pred_gbdt(fit_gbdt(Xtr, ytr, **kw), Xte)
    raise ValueError(kind)


def spec_of(fitter):
    return getattr(fitter, "_spec", None)


def cv_spatial(rows, Xfull, y, dlat, dlon, fitter, min_tr=40, workers=None):
    keys = [(int((r["lat"] + 90) // dlat), int((r["lon"] + 180) // dlon)) for r in rows]
    uniq = sorted(set(keys))
    spec = spec_of(fitter)
    if spec[0] == "gbdt" and len(uniq) > GBDT_MAX_BLOCKS:
        return float("nan"), 0            # 见 GBDT_MAX_BLOCKS 注释
    jobs, slots = [], []
    for k in uniq:
        te = np.array([i for i, kk in enumerate(keys) if kk == k])
        tr = np.array([i for i, kk in enumerate(keys) if kk != k])
        if len(tr) < min_tr or len(te) == 0:
            continue
        jobs.append((Xfull[tr], y[tr], Xfull[te], spec[0], spec[1]))
        slots.append(te)
    if not jobs:
        return np.nan, 0
    if spec[0] == "gbdt":
        # GBDT 折数多时逐折重训代价高，且 Windows 进程池在数百折下会崩。
        # 折数 > 40 时改为**单进程顺序**（每折仍很快：~3.8s / 150 树 / 3000 行）；
        # 实测 88 折 ≈ 90 s、31 折 ≈ 35 s，完全可接受。
        outs = []
        for i, j in enumerate(jobs):
            outs.append(_fold_worker(j))
            if (i + 1) % 25 == 0:
                print(f"      [cv {dlat}x{dlon}] {i + 1}/{len(jobs)} folds", flush=True)
    elif spec[0] == "knn" and len(jobs) > 4:
        outs = [_fold_worker(j) for j in jobs]
    else:
        outs = [_fold_worker(j) for j in jobs]
    pred = np.full(len(y), np.nan)
    for te, o in zip(slots, outs):
        pred[te] = o
    m = np.isfinite(pred)
    return (r2_of(y[m], pred[m]), int(m.sum())) if m.sum() >= 30 else (np.nan, int(m.sum()))


def cv_group(rows, Xfull, y, key="campaign", fitter=None, workers=None,
             ret_pred=False):
    spec = spec_of(fitter)
    jobs, slots, groups = [], [], []
    for g in sorted({r[key] for r in rows}):
        te = np.array([i for i, r in enumerate(rows) if r[key] == g])
        tr = np.array([i for i, r in enumerate(rows) if r[key] != g])
        if len(tr) < 40 or len(te) == 0:
            continue
        jobs.append((Xfull[tr], y[tr], Xfull[te], spec[0], spec[1]))
        slots.append(te)
        groups.append(g)
    if jobs:
        outs = [_fold_worker(j) for j in jobs]      # 4 折，顺序即可
    else:
        outs = []
    pred = np.full(len(y), np.nan)
    per = {}
    for g, te, o in zip(groups, slots, outs):
        pred[te] = o
        per[g] = r2_of(y[te], o)
    m = np.isfinite(pred)
    r2 = r2_of(y[m], pred[m]) if m.sum() >= 30 else np.nan
    if ret_pred:
        return r2, per, pred
    return r2, per


def make_ridge(lam):
    def f(Xtr, ytr, Xte):
        return pred_ridge(fit_ridge(Xtr, ytr, lam), Xte)
    f._spec = ("ridge", {"lam": lam})
    return f


def make_gbdt(**kw):
    def f(Xtr, ytr, Xte):
        return pred_gbdt(fit_gbdt(Xtr, ytr, **kw), Xte)
    f._spec = ("gbdt", dict(kw))
    return f


def naive_knn(k=15):
    def f(Xtr, ytr, Xte):
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        A = (Xtr - mu) / sd
        B = (Xte - mu) / sd
        out = np.empty(len(B))
        for i in range(len(B)):
            d = ((A - B[i]) ** 2).sum(1)
            nn = np.argpartition(d, min(k, len(d) - 1))[:k]
            out[i] = ytr[nn].mean()
        return out
    f._spec = ("knn", {"k": k})
    return f


def _dump(report):
    json.dump(report, open(os.path.join(EXP, "pathc_models.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


def main():
    rows = load()
    print(f"[load] n={len(rows)}", flush=True)
    report = {"n": len(rows), "note": "leak-free = 剔除与 Δ_floor 代数耦合的 var_hr_n 系"}
    _dump(report)          # 立刻落盘，后续每步覆盖，防中途被 kill 丢结果

    # 先做一次"泄漏对照"：同一模型在含/不含代数耦合特征下的样本内与泛化差距
    y0 = np.array([r["delta_floor"] for r in rows], float)
    print("\n" + "=" * 96)
    print("### 对照：代数耦合特征（log VAR(HR)）对 R² 的虚高作用")
    fit_g = make_gbdt(n_trees=60, lr=0.15, depth=4, min_leaf=30, seed=1)
    for tag, leaky in [("leak-free", False), ("+ logVAR(HR)", True)]:
        Xf, yf, keep, keys = build(rows, y0, 1, include_leaky=leaky)
        sub = [rows[i] for i in keep]
        ins = r2_of(yf, fit_g(Xf, yf, Xf))
        c30 = cv_spatial(sub, Xf, yf, 30, 60, fit_g)[0]
        lc, per = cv_group(sub, Xf, yf, "campaign", fit_g)
        print(f"  {tag:<14s} p={Xf.shape[1]:2d} in-R2={ins:.3f}  CV30x60={c30:.3f}  "
              f"leaveCamp={lc:.3f}   " +
              " ".join(f"{k}={v:.2f}" for k, v in per.items() if np.isfinite(v)))
        report[f"leak_check|{tag}"] = {"p": int(Xf.shape[1]), "in_r2": ins,
                                       "cv_30x60": c30, "cv_leave_campaign": lc,
                                       "per_campaign": per, "keys": keys}
        _dump(report)

    for yname in ["delta_floor", "corr_bic_hr"]:
        y_all = np.array([r.get(yname, np.nan) for r in rows], float)
        print("\n" + "=" * 96)
        print(f"### y = {yname}")
        print(f"{'model':<30s} {'p':>3s} {'n':>5s} {'in':>6s} "
              f"{'CV5x5':>7s} {'CV10':>7s} {'CV20':>7s} {'CV30x60':>8s} {'leaveCamp':>10s}")
        res = {}
        for level in (1, 2):
            Xf, yf, keep, keys = build(rows, y_all, level)
            sub = [rows[i] for i in keep]
            print(f"  -- feature level {level}: p={Xf.shape[1]}  n={len(yf)}", flush=True)
            cands = [("OLS (lam=0)", make_ridge(0.0)),
                     ("ridge lam=10", make_ridge(10.0)),
                     ("knn k=15", naive_knn(15)),
                     ("GBDT n60 lr.15 d4", make_gbdt(n_trees=60, lr=0.15, depth=4,
                                                     min_leaf=30, seed=1)),
                     ("GBDT n80 lr.12 d5", make_gbdt(n_trees=80, lr=0.12, depth=5,
                                                     min_leaf=20, seed=2))]
            for name, fitter in cands:
                try:
                    is_tree = spec_of(fitter)[0] == "gbdt"
                    ins = r2_of(yf, fitter(Xf, yf, Xf))
                    cvs = {}
                    grids = [(20, 20), (30, 60)] if is_tree else \
                            [(5, 5), (10, 10), (20, 20), (30, 60)]
                    for a, b in grids:
                        cvs[f"{a}x{b}"] = cv_spatial(sub, Xf, yf, a, b, fitter)[0]
                    lc, per = cv_group(sub, Xf, yf, "campaign", fitter)
                except Exception as e:  # noqa: BLE001
                    print(f"    {name:<28s} FAILED {type(e).__name__}: {e}", flush=True)
                    continue
                print(f"    {name:<28s} {Xf.shape[1]:3d} {len(yf):5d} {ins:6.3f} "
                      f"{cvs.get('5x5', float('nan')):7.3f} "
                      f"{cvs.get('10x10', float('nan')):7.3f} "
                      f"{cvs.get('20x20', float('nan')):7.3f} "
                      f"{cvs.get('30x60', float('nan')):8.3f} {lc:10.3f}", flush=True)
                res[f"L{level}|{name}"] = {"p": int(Xf.shape[1]), "n": int(len(yf)),
                                           "in_r2": ins, "cv": cvs,
                                           "cv_leave_campaign": lc,
                                           "per_campaign": per}
                report[yname] = res
                _dump(report)

    json.dump(report, open(os.path.join(EXP, "pathc_models.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n[ok] wrote pathc_models.json")


if __name__ == "__main__":
    main()

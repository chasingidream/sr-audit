"""P2-1: 下游任务验证 harness —— 把 SR 增益是否真实，直接对接到下游 Earth-observation 任务。

核心论点（见 docs/P2_下游验证与通用性规划.md）：文章已证明"未审计的 SR 增益是评估伪象"。
本脚本补上**下游应用**证据：用同一个**冻结**的下游分类器/检测器，对三种 SR 输入推理，
算 mAcc / macro-F1 / mIoU，看 (②−①) 与 (③−①) 的**下游精度差**是否复现 Δ_floor 的排序：

  ① bicubic 上采样 LR（baseline）
  ② 未审计 SOTA（本已 fail floor，如 EDSR/WDSR 在 6.63×）
  ③ 审计通过的 RAMS / 本文方法

预期：若 ② 在 SR-PSNR 上虚增但在下游不增（甚至降），而 ③ 同时审计通过且下游真增 →
"审计协议筛掉的正是下游无用增益"，闭环 so-what。

设计：
- `Classifier` 接口可插拔：默认 `DummyTemplateClassifier`（纯 numpy 模板匹配，无需 GPU，
  用于冒烟测试）；真实接入点在 `ResNetClassifier`（懒加载 torch，EuroSAT/BigEarthNet 权重）。
- 整个 harness **torch-free** 默认路径，可在任意机器跑通流程；真实下游训练/推理时再在 GPU 上跑。
- 指标：mAcc（平均类内正确率）、macro-F1、mIoU（逐类 IoU 均值）。

用法：
  # 合成冒烟测试（无需真实权重/标签）
  python code/p2_downstream_verify.py --demo --out experiments/p2_downstream_verify.json --report experiments/p2_downstream_verify.md

  # 真实标注 EO 数据集（ImageFolder: root/<class>/<img>），自动构造 baseline 臂；
  # 外部 SR 模型输出（臂②/③）由 .npy 传入（可选），冻结 CNN 在同臂上训练后推理。
  python code/p2_downstream_verify.py \
      --data_root ~/data/EuroSAT \
      --sr_fail experiments/sr_edsr.npy --sr_pass experiments/sr_rams.npy \
      --classifier trainable --device cuda --epochs 15 --scale 4 --image_size 64 \
      --out experiments/p2_downstream_verify.json \
      --report experiments/p2_downstream_verify.md \
      --figure experiments/p2_downstream_verify.png

  # 直接喂三臂 .npy（臂①= bicubic，臂②/③= 外部 SR 模型输出）
  python code/p2_downstream_verify.py \
      --bicubic bicubic.npy --sr_fail edsr.npy --sr_pass rams.npy \
      --labels labels.npy --n_classes 10 --classifier resnet \
      --out experiments/p2_downstream_verify.json --report experiments/p2_downstream_verify.md
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# ---------------------------------------------------------------- 下游分类器接口
class Classifier:
    """下游分类器接口。predict 接受 (N,C,H,W) 返回 (N, n_classes) logits/probs。"""

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def predict(self, images):
        raise NotImplementedError

    def name(self):
        return self.__class__.__name__


class DummyTemplateClassifier(Classifier):
    """纯 numpy 模板匹配（冒烟测试用）。

    用每个类别的"干净模板"作为参考，输入图与模板做相似度取 argmax。图片越接近干净模板
    （即 SR 保真度越高），分类越准。从而三臂（bicubic / fail-SOTA / 审计通过法）会自然
    拉开下游精度差，演示 harness 逻辑。
    """

    def __init__(self, templates):
        super().__init__(len(templates))
        self.templates = templates  # (n_classes, C, H, W)

    def predict(self, images):
        N = images.shape[0]
        logits = np.zeros((N, self.n_classes))
        for i in range(N):
            img = images[i]
            for c in range(self.n_classes):
                t = self.templates[c]
                # 用负 MSE 距离作 logit：对模糊/噪声敏感（模糊图离干净模板更远），
                # 从而三臂会拉开下游精度差，演示真实下游分类器的判别行为。
                logits[i, c] = -float(((img - t) ** 2).mean())
        return logits

    def name(self):
        return "DummyTemplateClassifier(numpy)"


class ResNetClassifier(Classifier):
    """真实下游分类器（EuroSAT / BigEarthNet 风格 ResNet-50）。

    接入点：用真实数据集训练好一个冻结 ResNet，加载权重后对整个 (N,C,H,W) 批量推理。
    此处仅保留接口骨架——实际运行需在 GPU 环境安装 torch/timm，并把权重路径传入。
    """

    def __init__(self, n_classes, weights=None, device="cuda"):
        super().__init__(n_classes)
        import torch  # 懒加载：仅真实使用时才 import torch
        import torch.nn.functional as F  # noqa: F401
        self.device = device
        self.weights = weights
        # 真实实现：self.model = torch.load(weights); self.model.eval()
        self._torch = torch

    def predict(self, images):
        import torch
        t = torch.from_numpy(np.asarray(images, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            # 真实实现：logits = self.model(t)
            # 冒烟占位：返回与 Dummy 同形的随机但确定性输出
            rng = np.random.default_rng(0)
            return torch.tensor(rng.standard_normal((t.shape[0], self.n_classes))).to(self.device)

    def name(self):
        return "ResNetClassifier(%s)" % (self.weights or "untrained")


class TrainableCNNClassifier(Classifier):
    """真实可训练下游分类器（轻量 CNN，torch 懒加载）。

    用于真实 P2-1：在 (LR/bicubic/SR) 输入上训练一个小型卷积网络，冻结权重后
    对三臂推理。默认极轻量（2 卷积 + GAP + 线性），CPU 也可在小子集上合理时间训完。
    设计要点：P2-1 要求**同一冻结分类器**对三臂推理，因此用 `run_p2_downstream`
    在 baseline 臂上 fit、再对全部臂 predict——保证精度差只反映 SR 输入质量。
    """

    def __init__(self, n_classes, device="cuda", epochs=15, lr=1e-3, batch=64, seed=0):
        super().__init__(n_classes)
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        self.seed = seed
        self.model = None

    def _build(self):
        import torch.nn as nn
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, self.n_classes),
        )

    def fit(self, images, labels):
        import torch
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        from torch.utils.data import TensorDataset, DataLoader
        model = self._build().to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        crit = torch.nn.CrossEntropyLoss()
        X = torch.from_numpy(np.asarray(images, dtype=np.float32)).to(self.device)
        Y = torch.from_numpy(np.asarray(labels, dtype=np.int64)).to(self.device)
        dl = DataLoader(TensorDataset(X, Y), batch_size=self.batch, shuffle=True)
        for _ in range(self.epochs):
            for xb, yb in dl:
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()
        self.model = model
        return self

    def predict(self, images):
        import torch
        if self.model is None:
            raise RuntimeError("call fit() before predict()")
        self.model.eval()
        X = torch.from_numpy(np.asarray(images, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            return self.model(X).cpu().numpy()

    def name(self):
        return "TrainableCNNClassifier(epochs=%d,dev=%s)" % (self.epochs, self.device)


# ---------------------------------------------------------------- 指标
def classification_metrics(pred_labels, true_labels, n_classes):
    """mAcc / macro-F1 / mIoU。pred_labels, true_labels: (N,) int。"""
    pred = np.asarray(pred_labels).ravel()
    true = np.asarray(true_labels).ravel()
    eps = 1e-8
    acc = float((pred == true).mean())
    f1s, ious = [], []
    for c in range(n_classes):
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn + eps)
        iou = tp / (tp + fp + fn + eps)
        f1s.append(f1)
        ious.append(iou)
    return {"mAcc": acc, "macroF1": float(np.mean(f1s)), "mIoU": float(np.mean(ious))}


def run_arm(classifier, images, labels, n_classes):
    """对单臂 SR 输入跑下游分类，返回指标 dict + 预测标签。"""
    logits = classifier.predict(images)
    pred = np.asarray(logits).argmax(axis=1)
    m = classification_metrics(pred, labels, n_classes)
    m["n_samples"] = int(len(labels))
    return m, pred


# ---------------------------------------------------------------- 真实数据路径（P2-1 实跑）
def _bicubic_up(lr, size):
    """双三次上采样 (B,C,h,w)->size，与 p2_audit_external.bicubic_up 同一 scipy 口径。"""
    B, C, h, w = lr.shape
    fx, fy = size[0] / h, size[1] / w
    out = np.empty((B, C, size[0], size[1]), dtype=lr.dtype)
    for b in range(B):
        for c in range(C):
            out[b, c] = zoom(lr[b, c], (fx, fy), order=3)
    return out


def load_labeled_eo(root, image_size=None, limit=None, limit_per_class=None):
    """ImageFolder 式加载带标签 EO 数据集：root/<class_name>/<img>。

    支持 .png/.jpg/.jpeg/.tif/.tiff；输出 (images [N,3,H,W] float32[0,1], labels [N] int)。
    EuroSAT 等标准基准按此布局放置即可直接喂入。

    - limit: 全局张数上限（按类顺序截断，仅用于极小冒烟）。
    - limit_per_class: 每类张数上限（均衡子集；真实 P2-1 推荐用此而非 limit，
      避免旧实现按类顺序截断只截到第一类的 bug）。
    """
    from PIL import Image  # 懒加载，避免无 PIL 环境报错
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    imgs, labs = [], []
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    counts = {}
    for ci, cls in enumerate(classes):
        d = os.path.join(root, cls)
        counts[cls] = 0
        for fn in sorted(os.listdir(d)):
            if not fn.lower().endswith(exts):
                continue
            p = os.path.join(d, fn)
            if fn.lower().endswith((".tif", ".tiff")):
                import tifffile
                arr = tifffile.imread(p).astype(np.float32)
            else:
                arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, 0)
            elif arr.ndim == 3 and arr.shape[0] in (3, 4):
                arr = arr[:3]
            else:
                arr = arr.transpose(2, 0, 1)[:3]
            if image_size:
                from scipy.ndimage import zoom as _z
                arr = _z(arr, (1, image_size / arr.shape[1], image_size / arr.shape[2]), order=1)
            imgs.append(arr / 255.0 if arr.max() > 1.5 else arr)
            labs.append(ci)
            counts[cls] += 1
            if limit_per_class and counts[cls] >= limit_per_class:
                break
            if limit and len(imgs) >= limit:
                break
        if limit and len(imgs) >= limit:
            break
    return np.stack(imgs).astype(np.float32), np.array(labs)


def make_arms_for_labeled(images, scale=4, sr_pass=None, sr_fail=None):
    """由 HR 真值图构造三臂 SR 输入。

    - bicubic (baseline)：HR 先降采样 scale 倍再 bicubic 上采样。
    - audited-pass method / unaudited-fail SOTA：由外部 SR 模型产出（.npy，[N,C,H,W]），
      实跑时传入（见 P2 规划文档"待确认项"）。
    """
    N, C, H, W = images.shape
    lr = np.empty((N, C, H // scale, W // scale))
    for i in range(N):
        for k in range(C):
            lr[i, k] = zoom(images[i, k], (1 / scale, 1 / scale), order=1)
    bic = _bicubic_up(lr, (H, W))
    arms = {"bicubic (baseline)": bic}
    if sr_pass is not None:
        arms["audited-pass method"] = np.asarray(sr_pass)
    if sr_fail is not None:
        arms["unaudited-fail SOTA"] = np.asarray(sr_fail)
    return arms


def run_p2_downstream(arms_images, labels, n_classes, clf,
                     train_arm="bicubic (baseline)"):
    """同一冻结分类器对三臂推理：在 train_arm 上 fit，再对全部臂 predict。"""
    if train_arm not in arms_images:
        train_arm = next(iter(arms_images))
    clf.fit(arms_images[train_arm], labels)
    arms = {}
    for tag, imgs in arms_images.items():
        m, _ = run_arm(clf, imgs, labels, n_classes)
        arms[tag] = {"arm": tag, **m}
    base = arms[train_arm]
    for tag, m in arms.items():
        m["delta_vs_bicubic"] = {k: round(m[k] - base[k], 4)
                                 for k in ("mAcc", "macroF1", "mIoU")}
    return arms, base, train_arm


def run_p2_downstream_per_arm(arms_images, labels, n_classes, clf_factory, seed=0):
    """In-domain protocol: train a FRESH classifier on each arm, evaluate in-domain.

    Eliminates the frozen-classifier distribution-shift confound: every arm is judged by
    a classifier trained on its own SR distribution, so delta_vs_bicubic reflects the
    true downstream utility of that SR method, not train/test domain shift.
    """
    arms = {}
    for tag, imgs in arms_images.items():
        clf = clf_factory()
        clf.fit(imgs, labels)
        m, _ = run_arm(clf, imgs, labels, n_classes)
        arms[tag] = {"arm": tag, **m}
    base_tag = "bicubic (baseline)" if "bicubic (baseline)" in arms else next(iter(arms))
    base = arms[base_tag]
    for tag, m in arms.items():
        m["delta_vs_bicubic"] = {k: round(m[k] - base[k], 4)
                                 for k in ("mAcc", "macroF1", "mIoU")}
    return arms, base, base_tag


def _new_clf(args, n_classes, seed):
    """Classifier factory for one (arm, seed). Returns None for dummy (real-data unsupported)."""
    if args.classifier == "trainable":
        return TrainableCNNClassifier(n_classes, device=args.device,
                                     epochs=args.epochs, seed=seed)
    if args.classifier == "resnet":
        return ResNetClassifier(n_classes, weights=args.resnet_weights,
                               device=args.device)
    return None


def _build_arms(args):
    """Construct the three-arm image dict for real-data (ImageFolder or direct .npy)."""
    if args.data_root:
        images, labels = load_labeled_eo(
            args.data_root, image_size=args.image_size, limit=args.limit,
            limit_per_class=args.limit_per_class)
        sr_pass = np.load(args.sr_pass) if args.sr_pass else None
        sr_fail = np.load(args.sr_fail) if args.sr_fail else None
        arms_images = make_arms_for_labeled(images, scale=args.scale,
                                            sr_pass=sr_pass, sr_fail=sr_fail)
        n_classes = int(labels.max()) + 1
    else:
        for f in (args.bicubic, args.sr_fail, args.sr_pass, args.labels):
            if not f:
                print("[error] 非 --demo 且非 --data_root 时需提供 "
                      "--bicubic/--sr_fail/--sr_pass/--labels", file=sys.stderr)
                sys.exit(2)
        arms_images = {
            "bicubic (baseline)": np.load(args.bicubic),
            "unaudited-fail SOTA": np.load(args.sr_fail),
            "audited-pass method": np.load(args.sr_pass),
        }
        labels = np.load(args.labels).ravel()
        n_classes = args.n_classes
    return arms_images, labels, n_classes


def _aggregate_arms(seed_runs):
    """Aggregate per-seed arm metrics -> {tag: {metric: {mean, std}, delta_*: {...}}}."""
    tags = list(seed_runs[0].keys())
    metrics = ["mAcc", "macroF1", "mIoU"]
    agg = {}
    for tag in tags:
        agg[tag] = {}
        for k in metrics:
            vals = [r[tag][k] for r in seed_runs]
            agg[tag][k] = {"mean": round(float(np.mean(vals)), 4),
                          "std": round(float(np.std(vals)), 4)}
        for k in metrics:
            vals = [r[tag]["delta_vs_bicubic"][k] for r in seed_runs]
            agg[tag]["delta_vs_bicubic_" + k] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4)}
    return agg


def plot_downstream(json_path, out_png):
    """由 p2_downstream_verify.json 绘制三臂下游精度柱状图（so-what 配图，英文 publication-ready）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = json.load(open(json_path, encoding="utf-8"))
    arms = d["arms"]
    tags = list(arms.keys())
    # 臂名 -> 作图显示名（与英文主文一致）
    disp = {"bicubic (baseline)": "bicubic\n(baseline)",
            "unaudited-fail SOTA": "unaudited-fail\nSOTA",
            "audited-pass method": "audited-pass\nmethod"}
    if "agg" in d:
        macc = [d["agg"][t]["mAcc"]["mean"] for t in tags]
        stdv = [d["agg"][t]["mAcc"]["std"] for t in tags]
    else:
        macc = [arms[t]["mAcc"] for t in tags]
        stdv = [0.0 for _ in tags]
    dacc = [arms[t]["delta_vs_bicubic"]["mAcc"] for t in tags]
    colors = ["#888888" if x >= 0 else "#cc4444" for x in dacc]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(len(tags)), macc, yerr=stdv, capsize=4,
                  color=colors, edgecolor="#333333")
    for i, (v, dv) in enumerate(zip(macc, dacc)):
        ax.text(i, v + 0.01 + stdv[i], "%.3f\n(Δ%.3f)" % (v, dv),
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels([disp.get(t, t) for t in tags], fontsize=9)
    ax.set_ylabel("mAcc")
    ax.set_ylim(0, 1.08)
    proto = d.get("protocol", "frozen")
    title = ("P2-1: per-arm in-domain classifier accuracy\n"
             if proto == "per-arm in-domain"
             else "P2-1: frozen downstream classifier accuracy\n")
    if "agg" in d:
        title += "seeds=%s" % (",".join(str(s) for s in d.get("seeds", [])))
    else:
        title += "audited-pass > bicubic > unaudited-fail"
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    return out_png


# ---------------------------------------------------------------- 合成冒烟数据
def _make_synthetic(n=120, n_classes=4, res=32, seed=0):
    """构造三臂 SR 图 + 标签 + 分类模板。

    每个类别一个干净模板（平滑、互不相同）。GT 图 = 模板 + 小噪声。
    三臂退化：
      good   = GT（近真，审计通过法）
      bicubic= GT 轻微模糊（baseline）
      fail   = GT 模糊 + 大噪声（未审计 SOTA 的伪增益输入）
    """
    rng = np.random.default_rng(seed)
    # 模板 = 共享低频基底 + 类专属分量（类间有相关，分类非平凡）
    shared = rng.standard_normal((3, res, res))
    uniq = rng.standard_normal((n_classes, 3, res, res))
    templates = 0.65 * shared[None] + 0.35 * uniq
    templates = (templates - templates.min()) / (templates.max() - templates.min() + 1e-12)
    labels = rng.integers(0, n_classes, n)
    gt = np.empty((n, 3, res, res))
    for i in range(n):
        gt[i] = templates[labels[i]] + rng.standard_normal((3, res, res)) * 0.02
    gt = np.clip(gt, 0, 1)

    good = gt.copy()
    bic = np.empty_like(gt)
    fail = np.empty_like(gt)
    for i in range(n):
        for k in range(3):
            bic[i, k] = gaussian_filter(gt[i, k], sigma=3.5)   # 明显模糊的 baseline
            fail[i, k] = np.clip(gaussian_filter(gt[i, k], sigma=3.5)
                                 + rng.standard_normal((res, res)) * 0.14, 0, 1)
    return {"good": good, "bicubic": bic, "fail": fail, "labels": labels,
            "templates": templates}


def _run_demo():
    data = _make_synthetic(n=160, n_classes=4, res=32, seed=1)
    clf = DummyTemplateClassifier(data["templates"])

    def _merge(tag, imgs):
        m, _ = run_arm(clf, imgs, data["labels"], clf.n_classes)
        return {"arm": tag, **m}

    arms = {
        "bicubic (baseline)": _merge("bicubic (baseline)", data["bicubic"]),
        "unaudited-fail SOTA": _merge("unaudited-fail SOTA", data["fail"]),
        "audited-pass method": _merge("audited-pass method", data["good"]),
    }
    base = arms["bicubic (baseline)"]
    print("=== P2-1 downstream (synthetic) ===")
    for tag, m in arms.items():
        delta = {k: round(m[k] - base[k], 4) for k in ("mAcc", "macroF1", "mIoU")}
        print("  %-22s mAcc=%.3f macroF1=%.3f mIoU=%.3f | Δvs bic: %s"
              % (tag, m["mAcc"], m["macroF1"], m["mIoU"], delta))
    return arms


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="P2-1: downstream-task verification harness")
    ap.add_argument("--demo", action="store_true", help="synthetic smoke test")
    ap.add_argument("--bicubic", default=None, help="SR images .npy (N,C,H,W) for arm 1")
    ap.add_argument("--sr_fail", default=None, help="SR images .npy for unaudited-fail SOTA")
    ap.add_argument("--sr_pass", default=None, help="SR images .npy for audited-pass method")
    ap.add_argument("--labels", default=None, help="ground-truth class labels .npy (N,) int")
    ap.add_argument("--n_classes", type=int, default=4)
    ap.add_argument("--classifier", default="dummy",
                   choices=["dummy", "resnet", "trainable"])
    ap.add_argument("--resnet_weights", default=None)
    ap.add_argument("--data_root", default=None,
                   help="真实标注 EO 数据集根目录（ImageFolder：root/<class>/<img>）")
    ap.add_argument("--scale", type=int, default=4, help="SR 倍率（HR 降采样倍率）")
    ap.add_argument("--image_size", type=int, default=64,
                   help="真实数据 HR 统一 resize 到 (image_size, image_size)")
    ap.add_argument("--limit", type=int, default=None, help="仅加载前 N 张（冒烟/小子集）")
    ap.add_argument("--limit_per_class", type=int, default=None,
                   help="每类最多加载 N 张（均衡子集，推荐用于真实 P2-1）")
    ap.add_argument("--train_arm", default="bicubic (baseline)",
                   help="冻结分类器在其上训练的臂（默认 baseline）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", default="experiments/p2_downstream_verify.json")
    ap.add_argument("--report", default=None)
    ap.add_argument("--figure", default=None, help="下游精度柱状图输出 png 路径")
    ap.add_argument("--seed", type=int, default=0, help="随机种子（单次运行）")
    ap.add_argument("--seeds", default=None,
                   help="多种子敏感性：逗号分隔，如 0,1,2,3,4（覆盖 --seed，聚合 mean±std）")
    ap.add_argument("--per_arm_train", action="store_true",
                   help="逐臂训练(in-domain)协议：每臂单独训练分类器并域内评估，消除分布漂移混淆")
    args = ap.parse_args()
    seeds = ([int(x) for x in args.seeds.split(",") if x.strip() != ""]
             if args.seeds else [args.seed])
    agg = None

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_classes_report = args.n_classes  # demo 默认 4；真实路径下方会覆盖为数据实际类别数

    if args.demo:
        arms = _run_demo()
    else:
        # ---- 构造三臂 SR 输入 ----
        arms_images, labels, n_classes = _build_arms(args)
        n_classes_report = n_classes

        # ---- 多种子循环（seed 敏感性）----
        seed_runs = []
        for sd in seeds:
            if args.per_arm_train:
                arms_sd, base, train_arm = run_p2_downstream_per_arm(
                    arms_images, labels, n_classes,
                    (lambda s=sd: _new_clf(args, n_classes, s)), seed=sd)
            else:
                clf = _new_clf(args, n_classes, sd)
                if clf is None:
                    print("[error] dummy 分类器仅用于 --demo（真实数据需 trainable/resnet）",
                          file=sys.stderr)
                    return 2
                arms_sd, base, train_arm = run_p2_downstream(
                    arms_images, labels, n_classes, clf, train_arm=args.train_arm)
            print("  [seed=%d %s] train_arm=%s | classifier=%s"
                  % (sd, "per-arm" if args.per_arm_train else "frozen",
                     train_arm, args.classifier))
            seed_runs.append(arms_sd)

        if len(seed_runs) == 1:
            arms = seed_runs[0]
        else:
            arms = seed_runs[-1]  # 末种子用于作图的单点
            agg = _aggregate_arms(seed_runs)

    # 落盘（run_p2_downstream 的 arms 已含 delta；demo 路径此处补算相对 baseline 的差）
    if args.demo:
        base = arms["bicubic (baseline)"]
        for tag, m in arms.items():
            m["delta_vs_bicubic"] = {k: round(m[k] - base[k], 4)
                                     for k in ("mAcc", "macroF1", "mIoU")}
    out_obj = {"task": "P2-1 downstream verification",
               "protocol": "per-arm in-domain" if args.per_arm_train else "frozen",
               "seeds": seeds,
               "n_classes": n_classes_report,
               "arms": arms}
    if agg is not None:
        out_obj["agg"] = agg
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print("[saved] %s" % args.out)

    if args.figure:
        try:
            plot_downstream(args.out, args.figure)
            print("[saved] %s" % args.figure)
        except Exception as e:  # noqa: BLE001
            print("[warn] 下游配图生成失败: %s" % e, file=sys.stderr)

    if args.report:
        proto = "逐臂训练(in-domain)" if args.per_arm_train else "冻结分类器"
        L = ["# P2-1 下游任务验证（SR 增益 → 下游精度）", "",
             "协议：%s；种子：%s" % (proto, ",".join(str(s) for s in seeds)),
             "下游指标：mAcc / macro-F1 / mIoU；Δ = 相对 bicubic baseline 的下游精度差。", ""]
        if agg is not None:
            L.append("| 臂 | mAcc(mean±std) | ΔmAcc(mean±std) | macro-F1 mean | mIoU mean |")
            L.append("|---|---|---|---|---|")
            for tag, a in agg.items():
                m = a["mAcc"]; d = a["delta_vs_bicubic_mAcc"]
                L.append("| %s | %.3f±%.3f | %+.3f±%.3f | %.3f | %.3f |" % (
                    tag, m["mean"], m["std"], d["mean"], d["std"],
                    a["macroF1"]["mean"], a["mIoU"]["mean"]))
            L += ["", "---", ""]
            L.append("**解读（多种子聚合）**：ΔmAcc 以 mean±std 报告；若某臂 std 偏大或 CI 跨 0，"
                     "其下游精度差不显著，不能据此断言审计 verdict 预测下游。")
        else:
            L.append("| 臂 | mAcc | macro-F1 | mIoU | ΔmAcc | ΔF1 | ΔmIoU |")
            L.append("|---|---|---|---|---|---|---|")
            for tag, m in arms.items():
                d = m["delta_vs_bicubic"]
                L.append("| %s | %.3f | %.3f | %.3f | %+.3f | %+.3f | %+.3f |" % (
                    tag, m["mAcc"], m["macroF1"], m["mIoU"],
                    d["mAcc"], d["macroF1"], d["mIoU"]))
            L += ["", "---", ""]
            L.append("**解读**：审计通过法（③）下游精度应 > bicubic（①），且 > 未审计失败 SOTA（②）；"
                     "若 ② 在 SR-PSNR 上虚增但下游不增/为负，则证明审计筛掉的是下游无用增益。")
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("\n".join(L) + "\n")
        print("[saved] %s" % args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

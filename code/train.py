"""Training entry point for the spatiotemporal SR model.

Usage:
    python train.py                 # uses configs/default.py, synthetic smoke test
    python train.py --stage rsa    # freeze backbone, train RSA only
    python train.py --epochs 5 --smoke  # 2-batch quick check

NOTE on this machine: torch CUDA segfaults unless C:\\Windows\\System32\\vcomp140.dll
is hidden during the run. The block at the very top renames it before importing
torch and restores it on exit, so you can run training directly and safely.
"""
import os
import sys
import atexit
import argparse
import time
import multiprocessing as _mp

# --------------------------------------------------------------------------- #
# vcomp140.dll workaround -- MUST run before `import torch` (see project MEMORY)
# --------------------------------------------------------------------------- #
# IMPORTANT (Windows): torch CUDA segfaults unless C:\Windows\System32\vcomp140.dll
# is hidden during the run. This block must run BEFORE `import torch`.
# It must ONLY run in the MAIN process: when DataLoader uses num_workers>0 on
# Windows it spawns worker processes that re-import this module. If a worker
# "recovers" vcomp140.dll (renames .hide -> .dll) and THEN imports torch, the
# worker segfaults. So we guard with current_process().name == "MainProcess":
# the main process hides it once; workers inherit the already-hidden state and
# never touch it, so their torch import is safe.
_DLL = r"C:\Windows\System32\vcomp140.dll"
_HIDDEN = _DLL + ".hide"
_IS_MAIN = _mp.current_process().name == "MainProcess"


def _hide_dll():
    """Hide vcomp140.dll so torch CUDA import is safe. Returns True if hidden."""
    if os.path.exists(_HIDDEN) and not os.path.exists(_DLL):
        # recover from a previous crashed run (main process only)
        try:
            os.rename(_HIDDEN, _DLL)
        except Exception:
            pass
    if os.path.exists(_DLL):
        try:
            os.rename(_DLL, _HIDDEN)
            return True
        except Exception:
            print("[warn] could not hide vcomp140.dll; torch CUDA may segfault.")
    return False


def _restore_dll():
    if os.path.exists(_HIDDEN):
        try:
            os.rename(_HIDDEN, _DLL)
        except Exception:
            print("[warn] could not restore vcomp140.dll; please rename "
                  f"{_HIDDEN} -> {_DLL} manually.")


if _IS_MAIN:
    _hide_dll()
    atexit.register(_restore_dll)

# now safe (main: hidden above; workers: already hidden by main)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs.default import CONFIG
from models.sr_model import build_model
from data.dataset import get_dataset
from losses import CompositeLoss
from metrics import calc_psnr_ssim


def resolve(base, p):
    return p if os.path.isabs(p) else os.path.join(base, p)


def freeze_except(model, names):
    for n, p in model.named_parameters():
        p.requires_grad = any(n.startswith(x) for x in names)


def build_optimizer(model, cfg):
    return torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                             weight_decay=cfg["weight_decay"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="2-batch quick check")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--cfg", default=None,
                    help="path to a python config module (overrides configs.default)")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子（多种子复跑用，stage 名带 _s<N> 后缀隔离存档）")
    args = ap.parse_args()

    # 固定随机种子，保证消融可复现（训练增广/权重初始化/DataLoader 顺序）
    import random as _random, numpy as _np, torch as _torch
    _random.seed(args.seed); _np.random.seed(args.seed)
    _torch.manual_seed(args.seed)
    _torch.backends.cudnn.deterministic = True

    # 允许用 --cfg 指定别的配置文件（消融/真实数据场景）
    if args.cfg:
        import importlib.util
        spec = importlib.util.spec_from_file_location("user_cfg", args.cfg)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg = dict(mod.CONFIG)
    else:
        cfg = dict(CONFIG)
    if args.stage:
        cfg["stage"] = args.stage
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.resume:
        cfg["resume"] = args.resume

    base = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        cfg.get("project_root", "..")))
    # 每个 stage 单独存盘/记日志，避免三档消融的 ckpt/log 互相覆盖
    save_dir = os.path.join(resolve(base, cfg["save_dir"]), cfg["stage"])
    log_dir = os.path.join(resolve(base, cfg["log_dir"]), cfg["stage"])
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}  stage={cfg['stage']}  cuda_available="
          f"{torch.cuda.is_available()}")

    # 人为未对齐变体：misaligned* / xs_misaligned* 档在输入 LR 施加随机仿射，
    # 模拟跨时相/跨传感器配准误差；对齐档（其他 stage）不加。
    _stage = cfg["stage"]
    _is_mis = isinstance(_stage, str) and (
        _stage.startswith("misaligned") or _stage.startswith("xs_misaligned")
        or _stage.startswith("mtxa_mis"))
    geo_jitter = cfg.get("geo_jitter") if _is_mis else None
    # 跨传感器：对"全部" LR 帧注入失配（HR 作锚点）；仅 misaligned* 只对 frame1..T-1。
    cfg["geo_jitter_all"] = _stage.startswith("xs_misaligned")
    # 对齐模块模式（option 2）：必须在 build_model 之前确定（构造参数）。
    #   xs_*   : cross-attn + HR 锚点（跨传感器；注意 V 来自 HR = GT -> 会泄漏，
    #            仅用于对照，不能用于无泄漏验证）
    #   mtxa_* : cross-attn + frame0 锚点（多时相自对齐，锚点非 GT -> 无泄漏，
    #            这是判断 cross-attn 对齐是否可行的干净实验）
    if isinstance(_stage, str) and _stage.startswith("xs_"):
        cfg["align_mode"] = "cross_attn"
        cfg["align_ref"] = "hr"
    elif isinstance(_stage, str) and _stage.startswith("mtxa"):
        cfg["align_mode"] = "cross_attn"
        cfg["align_ref"] = "frame0"
    else:
        cfg["align_mode"] = cfg.get("align_mode", "cost_volume")
        cfg["align_ref"] = cfg.get("align_ref", "frame0")

    # data
    train_ds = get_dataset(cfg, project_root=base, deterministic=False, geo_jitter=geo_jitter)
    val_ds = get_dataset(cfg, project_root=base, deterministic=True, geo_jitter=geo_jitter)  # 验证集固定 crop，指标可复现
    val_frac = float(cfg.get("val_frac", 0.1))   # 默认 0.1（历史口径）；P3 n≥300 用 0.5
    n_train = max(1, min(len(train_ds) - 1, int((1.0 - val_frac) * len(train_ds))))
    train_ds = torch.utils.data.Subset(train_ds, list(range(n_train)))
    val_ds = torch.utils.data.Subset(val_ds, list(range(n_train, len(val_ds))))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                             num_workers=cfg["num_workers"], drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"])

    # 验证集全局 data_range：取 GT(HR target) 在整段 val 上的全局动态范围，
    # 作为所有 stage 统一的 PSNR/SSIM 满量程，保证跨方法绝对可比（消除虚高）。
    val_dr = 1.0
    with torch.no_grad():
        mn, mx = float("inf"), float("-inf")
        for vb in val_loader:
            t = vb["target"].float()
            mn = min(mn, float(t.min())); mx = max(mx, float(t.max()))
        if mx - mn > 1e-8:
            val_dr = mx - mn
    print(f"[info] val_dr={val_dr:.6f} (global HR dynamic range over val set)")

    # model
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] model params={n_params/1e3:.1f}K")

    # stage gating — 隔离 RSA / 多时相 / HR 参考引导 的边际贡献
    #   sisr    : 纯单帧超分（无自对齐、无多时相融合、无 HR 参考）= 真 baseline
    #   mt      : 多时相融合 + HR 参考引导，但不用 RSA 自对齐
    #   mt+rsa  : 全开（多时相 + RSA 自对齐 + HR 参考引导；end2end/full 同义）
    stage = cfg["stage"]
    # 变体后缀别名（迭代剥离直到稳定）：
    #   "_l1"(纯L1损失) / "_ct"(ConvTranspose上采样) / "_rbc"(残差on-bicubic) /
    #   "_zi"(残差零初始化) / "_s<N>"(多种子，如 mtxa_aligned_rbc_zi_s7)
    # 门控按剥离后的基础名执行；save_dir/log 用全名隔离。
    import re as _re
    _SUFFIXES = ("_l1", "_ct", "_rbc", "_zi")
    _changed = True
    while _changed:
        _changed = False
        for _suf in _SUFFIXES:
            if stage.endswith(_suf):
                stage = stage[:-len(_suf)]
                _changed = True
        if _re.search(r"_s\d+$", stage):
            stage = _re.sub(r"_s\d+$", "", stage)
            _changed = True
    use_rsa, use_ref = True, True
    cfg["loss"]["w_flow"] = 0.0  # 仅 misaligned_rsa 档开启已知-GT 流监督
    if stage in ("sisr", "baseline"):
        freeze_except(model, ["extractor", "fusion", "recon", "guidance"])
        use_rsa, use_ref = False, False
        cfg["loss"]["w_cyclic"] = 0.0
    elif stage == "mt_no_ref":
        # 多时相融合本身：T>1 多帧融合，但无 RSA 自对齐、无 HR 参考引导。
        # 与 sisr(T=1 单帧) 对照 -> 干净隔离"多时相融合"的边际贡献；
        # 与 mt(+HR参考) 对照 -> 隔离 HR 参考引导的贡献。
        freeze_except(model, ["extractor", "fusion", "recon", "guidance"])
        use_rsa, use_ref = False, False
        cfg["loss"]["w_cyclic"] = 0.0
    elif stage == "mt":
        freeze_except(model, ["extractor", "fusion", "recon", "guidance"])
        use_rsa, use_ref = False, True
        cfg["loss"]["w_cyclic"] = 0.0
    elif stage == "misaligned_norsa":
        # 人为失配 + 无 RSA：LR 帧带几何扰动直接融合 -> 下界（量化"无对齐"的代价）
        freeze_except(model, ["extractor", "fusion", "recon", "guidance"])
        use_rsa, use_ref = False, False
        cfg["loss"]["w_cyclic"] = 0.0
    elif stage == "misaligned_rsa":
        # 人为失配 + RSA 自对齐：rsa 可训，cyclic 正则驱动自对齐 -> 验证 RSA 价值。
        # 关键：use_ref=False 去掉 GT 泄漏，仅看 RSA 能否把失配帧对齐救回。
        # 路线②修正：cyclic 自监督在"几何失配+内容差异"并存的多时相数据上不成立
        # （内容差远大于几px几何差，梯度不驱动对齐）。故改用 KNOWN-GT 流监督
        # （dataset 回传施加的位移作 GT），直接验证 RSA 能否恢复已知失配。
        freeze_except(model, ["extractor", "fusion", "recon", "rsa", "guidance"])
        use_rsa, use_ref = True, False
        cfg["loss"]["w_cyclic"] = 0.0
        cfg["loss"]["w_flow"] = 1.0
    elif stage in ("xs_aligned", "xs_misaligned_noxa", "xs_misaligned_xa"):
        # 路线②③：跨传感器 + cross-attention 对齐（替换 cost-volume 流，option 2）。
        # HR 作对齐锚点（仅用于对齐，不泄漏进重建 -> use_ref=False）。
        # 对全部 LR 帧注入已知几何失配（geo_jitter_all，GT 已知、无泄漏）。
        cfg["align_mode"] = "cross_attn"
        cfg["align_ref"] = "hr"
        cfg["loss"]["w_cyclic"] = 0.0
        cfg["loss"]["w_flow"] = 0.0
        if stage == "xs_misaligned_noxa":
            # 失配 + 无对齐（直接融合）-> 下界，量化"失配代价"
            freeze_except(model, ["extractor", "fusion", "recon", "guidance"])
            use_rsa, use_ref = False, False
        else:
            # xs_aligned / xs_misaligned_xa：cross-attn 对齐（HR 锚点）
            freeze_except(model, ["extractor", "fusion", "recon", "xattn", "guidance"])
            use_rsa, use_ref = True, False
    elif stage in ("mtxa_aligned", "mtxa_mis_noxa", "mtxa_mis_xa"):
        # 无泄漏版 cross-attn 对齐验证（option 2 干净实验）：
        # 锚点 = frame0（LR 帧，非 GT）-> 不泄漏；失配只加在 frame1..T-1。
        # 对照：mtxa_mis_xa − mtxa_mis_noxa = cross-attn 对齐的干净贡献；
        #       若 mtxa_mis_xa ≈ mtxa_aligned 说明 cross-attn 完全救回失配。
        cfg["align_mode"] = "cross_attn"
        cfg["align_ref"] = "frame0"
        cfg["loss"]["w_cyclic"] = 0.0
        cfg["loss"]["w_flow"] = 0.0
        # use_ref=False：HR 不参与，SR 完全由 LR 帧重建 -> 无 GT 泄漏
        if stage == "mtxa_mis_noxa":
            freeze_except(model, ["extractor", "fusion", "recon", "guidance"])
            use_rsa, use_ref = False, False
        else:
            freeze_except(model, ["extractor", "fusion", "recon", "xattn", "guidance"])
            use_rsa, use_ref = True, False
    elif stage in ("mt+rsa", "end2end", "full"):
        use_rsa, use_ref = True, True
    else:  # 未知 stage 视为全开
        use_rsa, use_ref = True, True
    print(f"[info] stage={stage}  use_rsa={use_rsa}  use_ref={use_ref}")

    loss_fn = CompositeLoss(cfg["loss"]).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.1)
    writer = SummaryWriter(log_dir)

    start_ep = 0
    if cfg.get("resume") and os.path.exists(cfg["resume"]):
        ckpt = torch.load(cfg["resume"], map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_ep = ckpt["epoch"] + 1
        print(f"[info] resumed from epoch {start_ep}")
    # 续训时把余弦调度对齐到断点（否则每段 LR 重置到峰值，破坏收敛）
    if cfg.get("resume") and os.path.exists(cfg["resume"]):
        try:
            scheduler.last_epoch = start_ep
        except Exception:
            pass

    epochs = 2 if args.smoke else cfg["epochs"]
    global_step = 0
    for epoch in range(start_ep, epochs):
        model.train()
        t0 = time.time()
        running = {}
        n_batches = 2 if args.smoke else len(train_loader)
        for bi, batch in enumerate(train_loader):
            if args.smoke and bi >= 2:
                break
            lrs = batch["lrs"].to(device)            # [B,T,C,h,w]
            hr_ref = batch["hr_ref"].to(device)      # [B,C,H,W]
            target = batch["target"].to(device)
            if stage == "sisr":
                lrs = lrs[:, :1]                     # 纯单帧超分
            optimizer.zero_grad()
            # 注意：始终传真实 hr_ref 给模型。HR 是否"泄漏进重建"由 use_ref 内部门控
            # （guidance 仅在 use_ref=True 时叠加）。use_ref=False 的跨传感器档
            # （xs_*）靠 HR 作对齐锚点、但不泄漏 -> 干净验证 RSA/cross-attn 贡献。
            pred, aux = model(lrs, hr_ref, return_aux=True,
                              use_rsa=use_rsa, use_ref=use_ref)
            loss, parts = loss_fn(pred, target, aux=aux, model=model, lrs=lrs)
            # 路线② RSA 验证：已知-GT 流监督（仅 misaligned_rsa 档，w_flow>0）。
            # dataset 回传施加的位移作 GT 流，逼 RSA 预测接近它 -> 直接证明 RSA 机制能恢复失配。
            if use_rsa and cfg["loss"].get("w_flow", 0.0) > 0 and "gt_flow" in batch:
                gf = batch["gt_flow"].to(device)              # [B,T,2,h,w]
                pred_flows = aux.get("flows") or []
                if pred_flows:
                    gt = gf[:, 1:]                            # frames 1..T-1
                    fl = sum((pf - gt[:, ti]).abs().mean()
                             for ti, pf in enumerate(pred_flows))
                    fl = fl / len(pred_flows) * cfg["loss"]["w_flow"]
                    loss = loss + fl
                    parts["flow"] = fl.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v
            global_step += 1
            if bi % 10 == 0:
                print(f"  ep{epoch} bi{bi} loss={parts['total']:.4f}")
        scheduler.step()

        # log
        for k, v in running.items():
            writer.add_scalar(f"train/{k}", v / max(1, n_batches), epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        # validation（用 GT=target 比较，统一 data_range=val_dr 消除虚高）
        if (epoch + 1) % cfg["val_every"] == 0 or args.smoke:
            model.eval()
            psnrs, ssims, flow_acc = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    lrs = batch["lrs"].to(device)
                    hr_ref = batch["hr_ref"].to(device)
                    target = batch["target"].to(device)
                    if stage == "sisr":
                        lrs = lrs[:, :1]
                    # 验证段同样始终传真实 hr_ref（use_ref 内部门控 guidance 泄漏）
                    pred, aux = model(lrs, hr_ref, use_rsa=use_rsa,
                                      use_ref=use_ref, return_aux=True)
                    # 不做 clamp：标准化后的数据动态范围远超 [0,1]（如 SEN2VENµS
                    # val_dr=4.27），若把 pred/target 双双 clamp 进 [0,1] 而
                    # data_range 仍用未 clamp 的 val_dr，MSE 被人为缩小 -> PSNR
                    # 严重虚高（实测 SEN2VENµS 虚高 ~15 dB、PROBA-V ~6.7 dB；
                    # 未标准化的 WorldStrat 因 clamp 是 no-op 而不受影响）。
                    # 与评测脚本 eval_skill_floor_cfg.py 口径统一：均不 clamp。
                    p, s = calc_psnr_ssim(pred, target, data_range=val_dr)
                    psnrs.append(p.item())
                    ssims.append(s.item())
                    # RSA 对齐诊断：平均预测位移幅度（特征空间 px）。
                    # 若 RSA 在学对齐 -> 该值应非零且接近施加的 jitter；norsa 档恒为 0。
                    if use_rsa and aux.get("flows"):
                        fa = torch.cat([f.abs().mean().unsqueeze(0)
                                        for f in aux["flows"]], 0).mean().item()
                        flow_acc.append(fa)
            if psnrs:
                mp, ms = sum(psnrs) / len(psnrs), sum(ssims) / len(ssims)
                writer.add_scalar("val/psnr", mp, epoch)
                writer.add_scalar("val/ssim", ms, epoch)
                fmsg = (f" flow={sum(flow_acc)/len(flow_acc):.3f}px"
                        if flow_acc else "")
                print(f"[val] ep{epoch} PSNR={mp:.2f} SSIM={ms:.4f}{fmsg} "
                      f"(dr={val_dr:.4f}, {time.time()-t0:.1f}s)")

        # checkpoint
        ckpt = {"model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "epoch": epoch, "cfg": cfg}
        torch.save(ckpt, os.path.join(save_dir, f"ckpt_ep{epoch}.pt"))
        print(f"[save] epoch {epoch} -> {save_dir}")

    writer.close()
    print("[done]")


if __name__ == "__main__":
    main()

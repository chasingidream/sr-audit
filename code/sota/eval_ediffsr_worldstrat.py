# -*- coding: utf-8 -*-
"""
Task #32: EDiffSR 双协议复审 (SOTA dual-protocol audit)

科学目标
--------
同一份官方权重、同一批世界区划数据，只改变「退化来源」，隔离出单一变量:
  A) synthetic x4 : HR -> bicubic 下采样 x4 -> EDiffSR 精化   == 该方法原生 Regime
  B) real  x6.63  : Sentinel-2 LR(64) -> bicubic 上采样到 424 -> EDiffSR 精化 == 真实跨传感器 Regime
两档下 EDiffSR 相对双三次的增益之差，即「增益缩水」的量化。

口径铁律(本项目已栽 4 次，不可让)
  1. 禁批池化: 一律以「逐样本 PSNR 平均」为准 + 配对 t 检验(CI/p/Cohen's d)
  2. 同时报技能地板(预测 GT 均值)，判断方法是否真越过"傻瓜预测器"

官方权重已知坑(不处理则结果全假)
  * 权重缺 enhance.rcab.* 共 8 个 key -> 无法 strict=True 载入
  * 直接 strict=False 会让 enhance 随机初始化并叠加到输出(x + enhance(x))，
    结果彻底失真 -> 必须替换为「输出全零」模块，数学上等价于训练时架构
  * IRSDE 内部 max_sigma/255 -> 权重工作于 [0,1] 值域
  * read_tif 对 LR 默认走 /65535(错的)，WorldStrat 的 S2 L2A 也是 DN=反射率x10000
"""
import argparse
import json
import os
import sys
import time

REPO = r""
CODES = os.path.join(REPO, r"third_party\_partial_extract\EDiffSR-main\codes")
SISR = os.path.join(CODES, r"config\sisr")
WEIGHTS = os.path.join(REPO, r"third_party\ediffsr\180000_G.pth")
MANIFEST = os.path.join(REPO, r"data\processed\worldstrat_manifest.json")
OUT_DIR = os.path.join(REPO, "experiments", "sota_review")

DLL = r"C:\Windows\System32\vcomp140.dll"
HID = DLL + ".hide"
_hidden = False


def _hide():
    global _hidden
    if os.path.exists(DLL):
        try:
            os.rename(DLL, HID)
            _hidden = True
            print("[guard] vcomp140.dll 已隐藏")
        except Exception as e:
            print("[guard] 隐藏失败:", e)


def _restore():
    global _hidden
    if _hidden and os.path.exists(HID):
        try:
            os.rename(HID, DLL)
            print("[guard] vcomp140.dll 已还原")
        except Exception as e:
            print("[guard] 还原失败(需手动处理!):", e)


ENHANCE_KEYS = {
    "enhance.rcab.0.weight", "enhance.rcab.0.bias",
    "enhance.rcab.2.weight", "enhance.rcab.2.bias",
    "enhance.rcab.3.attention.1.weight", "enhance.rcab.3.attention.1.bias",
    "enhance.rcab.3.attention.3.weight", "enhance.rcab.3.attention.3.bias",
}


def build_and_load_net(device):
    """载入官方权重，并忠实恢复训练时的架构(去掉 enhance 分支)。"""
    import torch
    import torch.nn as nn

    sys.path.insert(0, SISR)
    from models.modules.DenoisingNAFNet_arch import ConditionalNAFNet

    net = ConditionalNAFNet(img_channel=3, width=64,
                            enc_blk_nums=[14, 1, 1, 1],
                            middle_blk_num=1,
                            dec_blk_nums=[1, 1, 1, 1],
                            upscale=1)
    sd = torch.load(WEIGHTS, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    missing, unexpected = net.load_state_dict(sd, strict=False)

    # ---- 严格核查: 只允许 enhance 这一组缺失，且不允许出现多余 key ----
    miss, unex = set(missing), set(unexpected)
    if unex:
        raise RuntimeError("ckpt 存在架构外 key: %r" % sorted(unex)[:8])
    extra = miss - ENHANCE_KEYS
    if extra:
        raise RuntimeError("除 enhance 外还缺 key(配置可能不对): %r" % sorted(extra)[:8])
    print("[ckpt] 已载入; 缺失仅 enhance.rcab.*(%d 个) -> 替换为全零分支" % len(miss))

    class _Zero(nn.Module):
        def forward(self, x):
            return 0.0 * x

    net.enhance = _Zero()
    return net.to(device).eval()


def ediffsr_refine(net, sde, cond_up):
    """官方推理流程: cond_up 为已放大到目标分辨率的 LR"""
    import torch
    with torch.no_grad():
        state = sde.noise_state(cond_up)
        sde.set_model(net)
        sde.set_mu(cond_up)
        out = sde.reverse_sde(state, save_states=False)
    return out.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8, help="评测样本数(0=全部)")
    ap.add_argument("--val-start", type=int, default=3600,
                    help="从 manifest 该下标开始取(Task24 沿用后 10%% 验证区)")
    ap.add_argument("--T", type=int, default=1, help="LR 帧数(EDiffSR 是单帧方法)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--contrast", action="store_true",
                    help="开启逐通道 min-max 拉伸到 [0,1]，验证动态范围假设")
    ap.add_argument("--quiet-diag", dest="quiet_diag", action="store_true",
                    help="关闭逐样本诊断输出")
    args = ap.parse_args()

    _hide()
    try:
        import torch
        import numpy as np
        import torch.nn.functional as F

        # 注意包遮蔽: EDiffSR 的 codes/ 下也有同名的 data/ 与 models/ 包。
        # 必须先在项目 code/ 进 sys.path 之前导入 EDiffSR 侧模块，否则项目模块会被遮蔽。
        sys.path.insert(0, CODES)   # 提供 utils (codes/utils)
        sys.path.insert(0, SISR)    # 提供 models (codes/config/sisr/models)
        from utils.sde_utils import IRSDE
        from models.modules.DenoisingNAFNet_arch import ConditionalNAFNet

        # ---- 项目侧模块(扁平导入，遵循本项目惯例) ----
        sys.path.insert(0, os.path.join(REPO, "code"))
        from data.dataset import WorldStratDataset
        from eval_metrics import calc_psnr, calc_cc, calc_ssim
        from scipy import stats as _st

        # 项目版 eval_paired_floor 不能被导入: 它 from models.sr_model import ...，
        # 而此刻 sys.modules['models'] 已是 EDiffSR 的包(见上方包遮蔽说明)。
        # 这里按同一套数学本地实现: 配对 t 检验 + 95%CI + Cohen's d。
        def paired(xs, ys):
            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)
            n = min(len(xs), len(ys))
            xs, ys = xs[:n], ys[:n]
            if n < 2:
                return None
            diffs = xs - ys
            mean = float(diffs.mean())
            sd = float(diffs.std(ddof=1))
            se = sd / (n ** 0.5)
            if sd <= 0:
                return dict(mean_diff=mean, ci95=(mean, mean),
                            p=float("nan"), cohen_d=float("nan"), n=n)
            _, p = _st.ttest_rel(xs, ys)
            lo, hi = _st.t.interval(0.95, n - 1, loc=mean, scale=se)
            return dict(mean_diff=mean, ci95=(float(lo), float(hi)),
                        p=float(p), cohen_d=float(mean / sd), n=n)

        # ---- 修正辐射口径: WorldStrat 的 S2 L2A 与 SPOT 同为 DN=反射率x10000 ----
        import data.dataset as dsmod
        _orig_read = dsmod.read_tif

        def _patched_read(path, bands=None, div=None):
            return _orig_read(path, bands, div=div if div is not None else 10000.0)

        dsmod.read_tif = _patched_read
        print("[data] read_tif 已统一为 div=10000 (修正 LR 的 /65535 错误)")

        manifest = json.load(open(MANIFEST, encoding="utf-8"))
        start = args.val_start
        idxs = list(range(start, len(manifest)))
        if args.limit > 0:
            idxs = idxs[:args.limit]
        print("[data] 评测样本 idx: %s" % idxs)

        ds = WorldStratDataset(manifest, lr_size=64, T=args.T, deterministic=True,
                               hr_per_lr=6.63, hr_div=10000.0, norm="none")

        device = args.device
        net = build_and_load_net(device)
        sde = IRSDE(max_sigma=50, T=100, schedule="cosine", eps=0.005, device=device)

        results = {"A_synthetic_x4": {}, "B_real_x6.63": {}}

        # ---- A/B 两档共用评测核心 ----
        def run_condition(name, idxs):
            psnr_model, psnr_bic, psnr_mean = [], [], []
            cc_model, cc_bic = [], []
            n = 0
            t0 = time.time()
            for k, idx in enumerate(idxs):
                s = ds[idx]
                lr = s["lrs"][0:1].to(device)          # [1,3,64,64] 单帧
                gt = s["target"].to(device)            # [3,424,424]
                _, H, W = gt.shape               # [3,424,424]

                if args.contrast:
                    # 逐通道 min-max 拉伸到 [0,1] —— 验证"动态范围不匹配"假设。
                    # 用 GT 的通道极值定义映射，两个竞争预测器经历完全相同的变换，
                    # 故 model vs bicubic 的对比仍然公平。
                    gmin = gt.amin(dim=(1, 2), keepdim=True)
                    gmax = gt.amax(dim=(1, 2), keepdim=True)
                    rng = (gmax - gmin).clamp(min=1e-6)
                    gt = ((gt - gmin) / rng).clamp(0.0, 1.0)
                    lr = ((lr - gmin) / rng).clamp(0.0, 1.0)

                if name.startswith("A"):
                    # 合成 x4: 由 GT 下采样再放大，模拟该方法原生 regime
                    down = F.interpolate(gt[None], size=(H // 4, W // 4),
                                         mode="bicubic", align_corners=True)
                    cond = F.interpolate(down, size=(H, W), mode="bicubic",
                                         align_corners=True)
                else:
                    # 真实跨传感器: LR(64) 直接放大到 HR 分辨率
                    cond = F.interpolate(lr, size=(H, W), mode="bicubic",
                                         align_corners=True)

                out = ediffsr_refine(net, sde, cond).clamp(0.0, 1.0)
                bic = cond.clamp(0.0, 1.0)

                # --- 诊断: 模型相对其输入条件改动了多少 ---
                if not args.quiet_diag:
                    d_as_cond = float(calc_psnr(
                        out.detach().cpu().numpy().astype(np.float64),
                        bic.detach().cpu().numpy().astype(np.float64),
                        data_range=1.0))
                    print("                [diag] PSNR(out, cond)=%.2f dB" % d_as_cond)
                # 技能地板: 逐通道常数图(预测 GT 均值)
                flat = gt.mean(dim=(1, 2), keepdim=True).expand_as(gt)[None]

                # 必须统一带 batch 维: 项目 calc_cc 只看 p.ndim 决定是否扩维,
                # pred 4 维 + gt 3 维时环内 g[:,b] 会误切第 1 轴(得 3x424 而非 424^2)
                gt_np = gt[None].detach().cpu().numpy().astype(np.float64)
                for pred, bucket in ((out, psnr_model), (bic, psnr_bic), (flat, psnr_mean)):
                    p = pred.detach().cpu().numpy().astype(np.float64)
                    bucket.append(float(calc_psnr(p, gt_np, data_range=1.0)))
                cc_model.append(float(calc_cc(out.detach().cpu().numpy().astype(np.float64), gt_np)))
                cc_bic.append(float(calc_cc(bic.detach().cpu().numpy().astype(np.float64), gt_np)))
                n += 1
                print("  [%s] sample %d/%d ok  model=%.2f bic=%.2f mean=%.2f  (%.1fs)"
                      % (name, n, len(idxs), psnr_model[-1], psnr_bic[-1], psnr_mean[-1],
                         time.time() - t0))
            return {"psnr_model": psnr_model, "psnr_bicubic": psnr_bic,
                    "psnr_mean": psnr_mean, "cc_model": cc_model, "cc_bicubic": cc_bic}

        for name in ("A_synthetic_x4", "B_real_x6.63"):
            print("\n================ %s ================" % name)
            results[name] = run_condition(name, idxs)

        # ---- 汇总 + 配对检验 ----
        print("\n================ 汇总 ================")
        summary = {}
        for name, r in results.items():
            pm = np.array(r["psnr_model"])
            pb = np.array(r["psnr_bicubic"])
            pf = np.array(r["psnr_mean"])
            pt = paired(pm, pb)
            cin = pt["ci95"] if pt else (float("nan"), float("nan"))
            ccm = float(np.mean(r["cc_model"]))
            ccb = float(np.mean(r["cc_bicubic"]))
            gain = float(pm.mean() - pb.mean())
            summary[name] = {
                "per_sample_psnr_model": float(pm.mean()),
                "per_sample_psnr_bicubic": float(pb.mean()),
                "per_sample_psnr_meanfloor": float(pf.mean()),
                "gain_vs_bicubic_db": gain,
                "paired_ci": [float(cin[0]), float(cin[1])],
                "paired_p": float(pt["p"]) if pt else float("nan"),
                "cohens_d": float(pt["cohen_d"]) if pt else float("nan"),
                "cc_model": ccm,
                "cc_bicubic": ccb,
                "g1_pairing_cc": ccb,
                "g2_structure_ratio": float(ccm / ccb) if ccb != 0 else None,
                "n": int(len(pm)),
            }
            s = summary[name]
            print("%s: model=%.2f  bicubic=%.2f  meanFloor=%.2f  gain=%+.2f dB"
                  % (name, s["per_sample_psnr_model"], s["per_sample_psnr_bicubic"],
                     s["per_sample_psnr_meanfloor"], s["gain_vs_bicubic_db"]))
            print("   配对 t: CI=%s p=%.4f d=%.3f | G1 CC(bic,GT)=%.3f | G2 ratio=%.3f"
                  % (s["paired_ci"], s["paired_p"],
                     s["cohens_d"] if s["cohens_d"] is not None else float("nan"),
                     s["g1_pairing_cc"],
                     s["g2_structure_ratio"] if s["g2_structure_ratio"] else float("nan")))

        gainA = summary["A_synthetic_x4"]["gain_vs_bicubic_db"]
        gainB = summary["B_real_x6.63"]["gain_vs_bicubic_db"]
        summary["SHRINKAGE"] = {"gain_synthetic_x4": gainA, "gain_real_x663": gainB,
                                "shrink_abs_db": round(gainB - gainA, 3)}
        print("\n增益缩水: 合成x4 %+.2f dB  ->  真实x6.63 %+.2f dB  (差 %+.2f dB)"
              % (gainA, gainB, gainB - gainA))

        os.makedirs(OUT_DIR, exist_ok=True)
        op = os.path.join(OUT_DIR, "ediffsr_raw.json")
        json.dump(summary, open(op, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print("[saved]", op)

    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        _restore()


if __name__ == "__main__":
    import traceback
    main()

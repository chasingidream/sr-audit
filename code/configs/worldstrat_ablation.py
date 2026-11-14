"""WorldStrat 真实数据第一阶段消融配置（A+B 修正版）。

数据：data/processed/worldstrat_manifest.json（3927 个 LR-HR 配对，
每 coverage 16 个时相 Sentinel-2）。本机 RTX 4060 Ti 8GB，单卡，故默认用
子集 + 小 batch。

六档 stage ladder（隔离各模块边际贡献；前四档为"已配准/对齐"基线，五六档为路线② RSA 验证）：
  对齐基线（无人为扰动）：
    sisr      : 纯单帧超分（T=1，无自对齐、无多时相融合、无 HR 参考）= 真 baseline
    mt_no_ref : 多时相融合（T>1），但无 RSA 自对齐、无 HR 参考引导
    mt        : 多时相融合 + HR 参考引导，但不用 RSA 自对齐（HR 泄漏，仅用于隔离贡献）
    mt+rsa    : 全开（多时相 + RSA 自对齐 + HR 参考引导）
  路线② 人为未对齐变体（对 frame1..T-1 施加随机仿射，frame0 锚点不变）：
    misaligned_norsa : 失配 + 无 RSA（直接融合）-> 下界，量化"无对齐"的代价
    misaligned_rsa   : 失配 + RSA 自对齐（cyclic 正则驱动）-> 验证 RSA 能否把性能拉回
  （结论：cost-volume RSA 架构性失效，见 2026-08-31 记忆）

路线②③ 跨传感器 + cross-attention 对齐（替换 cost-volume 流，option 2+3 耦合）：
  锚点改为 HR 参考（跨传感器同景同时相，WorldStrat 原生设定）；对全部 LR 帧注入已知
  几何失配（geo_jitter_all=True，GT 已知、无泄漏）；HR 仅作对齐锚点，不泄漏进重建。
    xs_aligned        : cross_attn + HR锚点，无失配  -> 上界
    xs_misaligned_noxa: 无对齐（直接融合）+ 失配    -> 下界（量化失配代价）
    xs_misaligned_xa  : cross_attn + HR锚点 + 失配   -> 验证 cross-attn 能否救回失配
  对照逻辑：xs_misaligned_xa − xs_misaligned_noxa = cross-attn 对齐的干净贡献；
            若 xs_misaligned_xa 接近 xs_aligned 说明跨传感器自对齐成功。

指标修复：PSNR/SSIM 用验证集 GT 全局动态范围 (val_dr) 作统一满量程，跨方法可比。

切换 stage：python train.py --cfg code/configs/worldstrat_ablation.py --stage xs_misaligned_xa
"""

# 子集规模：从 3927 个配对里取前 N 个做第一阶段消融（全量留给论文正式训练）。
SUBSET = 600

CONFIG = {
    # --- model（HAUNet-S 量级，控制显存）---
    "in_ch": 3,
    "feat": 48,
    "scale": 7,                 # PixelShuffle 整数倍；真实 GSD≈6.63x，用 7 最接近
    "out_size": 424,            # 末端插值到真实 HR 尺寸（64*6.63≈424，非 7 整数倍）
    "T": 5,
    "lr_size": 64,
    "n_groups": 3,
    "n_blocks": 3,
    "rsa_radius": 3,
    "use_hr_ref": True,
    "use_conf": True,
    # 对齐模块模式（option 2）："cost_volume"(原RSA) | "cross_attn"(内容鲁棒) | "none"
    "align_mode": "cost_volume",
    "align_ref": "frame0",        # cross_attn 的参考锚点："frame0" | "hr"(跨传感器)

    # --- data（真实 WorldStrat）---
    "dataset": {
        "type": "worldstrat",
        # manifest 相对 project_root（= 仓库根 ）
        "manifest": "data/processed/worldstrat_manifest.json",
        "layout": "zenodo",
        "subset": SUBSET,          # 取前 SUBSET 个配对
        "hr_bands": (0, 1, 2),     # HR(_ps) RGBA -> RGB
        "lr_bands": (3, 2, 1),     # LR 12波段 Sentinel-2：B4/B3/B2 = RGB
        # --- 几何/辐射修正（2026-09-04，commit 79f4606）---
        "hr_per_lr": 6.63,         # 真实 GSD 比 1054/159≈6.63x（10m->1.5m）
        "hr_div": 10000.0,         # HR uint16 DN = 反射率*10000（旧 /65535 错）
        # 跨传感器辐射标定 LR->HR 尺度（训练集拟合全局常量，无泄漏）：
        #   LR_calib = 0.68586*LR + 0.001835  (R2=0.55)
        "lr_calib": [0.68586, 0.001835],
    },

    # --- 人为未对齐变体（路线②：验证 RSA 自对齐有效性）---
    # misaligned* 档对输入 LR 的 frame1..T-1 施加随机平移（frame0 保持干净作 RSA 锚点），
    # 模拟真实跨时相/跨传感器配准误差。
    # 注意：
    #  * jitter 固定到每个样本（dataset 用 Random(idx*1000+ti)），对齐模块才能学逆变换；
    #  * max_rot=0 -> 纯平移，GT 流为常量 (tx,ty)，可直接做有监督对齐验证；
    #  * **幅度必须够大才可测量**：@lr_size=64 时 0.06≈3.8px 已被实测证明对 PSNR 零损害
    #    （misaligned_norsa 28.18 vs mt_no_ref 28.20），上下界重合 -> 无法判定对齐是否有效。
    #    故加大到 0.12≈7.7px，**超出 cost-volume 半径(3~4px) 的表示范围**，使下界明显下降；
    #    而 cross-attn 无位移上限，若它能救回 -> 强贡献叙事。
    "geo_jitter": {"max_shift": 0.12, "max_rot": 0.0},
    # 跨传感器实验（option 3）：对"全部" LR 帧注入失配（HR 作锚点）。默认 False（仅 frame1..T-1）。
    "geo_jitter_all": False,

    # --- training ---
    "stage": "end2end",            # baseline | rsa | end2end
    "batch_size": 4,               # 8GB 显存保守值
    "epochs": 30,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "num_workers": 2,
    "val_every": 5,
    "resume": None,

    # --- losses ---
    "loss": {
        "w_l1": 1.0,
        "w_char": 0.0,
        "w_perc": 0.0,             # LPIPS 默认关（本机权重下载慢）
        "w_fourier": 0.1,
        "w_wavelet": 0.1,
        "w_flow": 1.0,
        "w_cyclic": 0.1,
    },

    # --- io ---
    "save_dir": "experiments/ckpts",
    "log_dir": "experiments/logs",
    "project_root": "..",
}

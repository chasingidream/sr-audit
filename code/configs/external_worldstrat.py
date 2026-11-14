"""WorldStrat 对照配置（P1 三轴：跨传感器 + 多帧 + 6.63×）。

作用：与 external_probav.py / external_sen2venus.py 共享【同一架构、同一 30-epoch
余弦协议、同一 loss 权重】，只在数据相关项上不同，使"模型 − 双三次"的方法层增益
可在 2× → 3× → 6.63× 三个尺度上直接横比。

关键一致性约束（#22 铁律）：
  * feat=48 / n_groups=3 / n_blocks=3 / rsa_radius=3 与另两个配置完全相同；
  * epochs=30 / lr=2e-4 / batch=4 / CosineAnnealing 完全相同；
  * loss 权重 w_l1=1.0, w_fourier=0.1, w_wavelet=0.1, w_flow=0, w_cyclic=0
    —— 与 external_probav.py 严格一致（早期 worldstrat_ablation 用 w_flow=1.0
    会破坏跨数据集可比性，故此处清零）；
  * stage=mtxa_aligned → cross_attn + frame0 锚点 + use_ref=False（无 GT 泄漏），
    与 external_probav 的 mtxa_aligned 完全同构。

数据：WorldStrat 真实配对（修正后 2026-09-04，commit 79f4606）：
  scale=7(out_size=424) 反映真实 GSD 比 1054/159≈6.63×；hr_div=10000 修正反射率
  量程；lr_calib=[0.68586,0.001835] 为训练集拟合的【全局】仿射（#23 正在复核
  是否应改逐场景）。

独立 save_dir/log_dir：避免与旧 worldstrat_ablation 的 mtxa_aligned* 档互相覆盖。

切换档位：
  python code/train.py --stage mtxa_aligned --cfg code/configs/external_worldstrat.py
"""

CONFIG = {
    # --- model（与 external_probav / external_sen2venus 对齐）---
    "in_ch": 3,                 # WorldStrat HR RGB = SPOT 真彩
    "feat": 48,
    "scale": 7,                 # PixelShuffle 整数倍
    "out_size": 424,            # 末端插值到真实 HR 尺寸（64*6.63≈424，非 7 整数倍）
    "T": 5,                     # 取 5 帧多时相
    "lr_size": 64,              # 与另两个数据集统一
    "n_groups": 3,
    "n_blocks": 3,
    "rsa_radius": 3,
    "use_hr_ref": False,        # mtxa_aligned：HR 仅作上界参考，不泄漏进重建
    "use_conf": True,
    "align_mode": "cross_attn", # 多时相自对齐（cost-volume 已证架构性失效）
    "align_ref": "frame0",      # 锚点取 LR 帧，杜绝 GT 泄漏

    # --- geo_jitter（仅 mtxa_mis* 档生效，见 train.py line145-148）
    # 模拟跨时相/跨传感器配准误差：max_shift=0.12 -> 0.12*64(lr_size)=7.7px LR 平移。
    # 注意：必须放在顶层！否则 cfg.get("geo_jitter") 返回 None -> mis 档零失配，
    # 与 aligned 档等价，三组对照全部失效（2026-09-15 复核发现并修复）。
    # mtxa_mis_noxa（cross-attn OFF）退化基线 vs mtxa_mis_xa（cross-attn ON）自对齐贡献。
    "geo_jitter": {"max_shift": 0.12, "max_rot": 0.0},

    # --- data（WorldStrat 修正管线）---
    "dataset": {
        "type": "worldstrat",
        "manifest": "data/processed/worldstrat_manifest.json",
        "layout": "zenodo",
        "subset": 600,
        "hr_bands": (0, 1, 2),     # HR(_ps) RGBA -> RGB
        "lr_bands": (3, 2, 1),     # S2 12 波段：B4/B3/B2 = RGB
        "hr_per_lr": 6.63,         # 真实 GSD 比 1054/159≈6.63x
        "hr_div": 10000.0,         # HR DN = 反射率*10000（旧 /65535 错）
        # 跨传感器辐射处理（#23 落地，2026-09-10）：
        #  初版用逐场景【仿射】标定（per_scene_calib），但拟合 slope≈0.4 把 LR 量程压扁
        #  -> 模型回归均值、落后双三次 0.83 dB（诊断见 _diag_model_range.py）。
        #  -> 改用逐场景【归一化】norm="per_scene"：用该场景 LR 的 (mu,sd) 把 LR 与 HR
        #     都标准化为 unit-variance，与 SEN2VENµS/PROBA-V 的 norm="per_scene" 完全同构，
        #     使三数据集预处理口径一致、且网络只需学"空间 SR + 平滑增益"。
        #  scene_norm 由 code/compute_worldstrat_scene_norm.py 生成（list[scene]->[mu,sd]）。
        "lr_calib": [0.68586, 0.001835],   # norm 优先；仅 norm="none" 且 per_scene_calib=False 时回退
        "per_scene_calib": False,
        "scene_calib": "data/processed/worldstrat_scene_calib.json",
        "norm": "per_scene",
        "scene_norm": "data/processed/worldstrat_scene_norm.json",
    },

    # --- training（与另两配置严格一致）---
    "stage": "mtxa_aligned",
    "batch_size": 4,
    "epochs": 30,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "num_workers": 2,
    "val_every": 5,
    "resume": None,

    "loss": {
        "w_l1": 1.0,
        "w_char": 0.0,
        "w_perc": 0.0,
        "w_fourier": 0.1,
        "w_wavelet": 0.1,
        "w_flow": 0.0,
        "w_cyclic": 0.0,
    },

    # 独立子目录：避免与旧 worldstrat_ablation 的 mtxa_aligned* 档互相覆盖
    "save_dir": "experiments/ckpts_worldstrat",
    "log_dir": "experiments/logs_worldstrat",
    "project_root": "..",
}

# 时空超分 · 代码说明

跨传感器 + 多时相 + 真实退化的遥感**时空超分辨率重建**。
核心创新点：**RSA 配准无关自对齐**（相对 STANet 去掉仿射假设、完全 registration-free）。
目标期刊层级：IEEE TGRS / ISPRS J / RSE / TIP / PR。详见 `docs/method_design.md`。

## 目录结构
```
code/
  configs/default.py        # 默认超参（可直接改或 --cfg 覆盖）
  models/
    backbone.py             # 轻量骨干：FeatureExtractor(RCAB) / TemporalFusion / Reconstruction
    rsa.py                  # RSA 配准无关自对齐（correlation cost volume + 可微 warp + 循环一致）
    sr_model.py             # 组装：Stage0 编码 -> Stage1 RSA -> Stage2 融合 -> Stage3 重建
  data/dataset.py           # 双管线：WorldStrat 真实 .tif / 模拟退化 / 合成（零下载冒烟）
  losses.py                 # L1/Charbonnier + LPIPS(惰性) + Fourier + Haar小波 + 流平滑 + 循环一致
  metrics.py                # YCbCr-Y 的 PSNR / SSIM
  train.py                  # 训练入口（含 vcomp140 运行时规避、分阶段、断点续训、TensorBoard）
  _smoke.py                 # CPU 冒烟测试（验证 forward/backward）
```

## 环境
使用托管 Python（已验证可用）：
```
PY=python   # 任意 3.10+ 解释器；需 torch/tensorboard
```
- 依赖已装：tifffile / opencv-python-headless / scikit-image / matplotlib / einops / lpips / pandas / tensorboard / numpy。
- **torch 需 CUDA 版**：默认 `pip install torch` 会装到 CPU 构建。本项目用 aliyun cu126 镜像的轮子：
  `torch-2.13.0+cu126-cp313-cp313-win_amd64.whl` 与 `torchvision-0.28.0+cu126-...whl`
  （下载后 `python -m pip install --no-deps <wheel>`）。
- **vcomp140 规避**：本机 torch CUDA 运行前必须把
  `C:\Windows\System32\vcomp140.dll` 改名（如 `.hide`），跑完改回，否则 segfault。
  `train.py` 已在导入 torch 前自动处理并在退出时还原，可直接 `python train.py`。

## 训练
```bash
cd code
# 1) 冒烟测试（CPU，无需数据/无需 CUDA）
python _smoke.py

# 2) 合成数据快速跑通（GPU，无需下载）
python train.py --epochs 2 --smoke

# 3) 正式训练（默认 synthetic，可改 configs/default.py 的 dataset）
python train.py --stage end2end --epochs 50

# 分阶段（对应 docs/method_design.md §6）：
#   baseline : 冻结 RSA，先复现 A2WNet-like 基线
#   rsa      : 冻结骨干，只训自对齐（循环一致自监督）
#   end2end  : 全量端到端
```

## 数据
- **synthetic**：内置，零下载，用于冒烟/调试。
- **simulated**：给一个高清图目录（png/jpg/tif），自动做多时相几何+辐射退化 + 双三次下采样生成 LR。
  在 `configs/default.py` 设 `dataset = {"type":"simulated","hr_dir":"/path/...","kwargs":{...}}`。
- **worldstrat**：真实配对（Sentinel-2 LR ↔ SPOT HR .tif）。用 `build_manifest_worldstrat(root)`
  或自定义 manifest（`[{"lr":[...], "hr": "..."}, ...]`）喂给 `WorldStratDataset`。
  注意 WorldStrat 全量数据 ~107GB（Kaggle/Zenodo），先用合成数据跑通 pipeline 再迁真实。

## 评测
`metrics.calc_psnr_ssim(pred, target)` 给出 Y 通道 PSNR/SSIM；训练脚本每 `val_every` 个 epoch
在验证集打印。LPIPS/FID 等感知指标与下游任务（分类/变化检测）在论文阶段补。

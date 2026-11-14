# SR 项目成果服务化（两套互补的 HTTP 服务）

把项目的"成果"封装成前端可直接调用的 HTTP 服务，分两层、互为补充：

| 服务 | 文件 | 输入 → 输出 | 说明 |
|---|---|---|---|
| **⭐ 统一网关（推荐）** | `gateway_service.py` | 一个端口同时提供 ①②③ | `/sr` 出图 + `/audit` 验真 + `/sr_audit` 一键（出图+预检徽章） |
| **① 图像超分推理** | `sr_infer_service.py` | 一张低分辨率图片 → 超分后的图片 | 模型层：加载训练好的超分模型，直接出图。 |
| **② SR-Audit 审计协议** | `audit_service.py` | 某个方法的评测证据（逐场景指标）→ 可信徽章 + regime | 协议层：判断"这个超分增益是否真实可信"，与论文报告标准口径一致。 |

> 为什么要两套？一张图超分出来，**不代表它的增益是真的**（这正是本论文的核心论点：跨传感器高倍率下很多"增益"是 illusory 的）。所以 ① 负责"出图"，② 负责"验真"——统一网关把两者 + 一键预检合并到一个端口。

---

## 0. 统一网关（`gateway_service.py`，推荐入口）

```bash
python code/service/gateway_service.py --port 8770
# 打开 http://127.0.0.1:8770/ 统一演示页（三个面板：出图 / 验真 / 一键）
```

**路由**：`GET /`(演示页)｜`/health`｜`/axes`(三轴模型档案)｜`/schema`｜`/example/audit`｜
`POST /audit`、`/assess`(转发审计服务)｜`POST /sr?model=&compare=`(出图)｜
`POST /sr_audit?model=`(一键：SR 图 base64 + 无 GT 预检徽章)｜`POST /load`(预装载)。

**三轴模型档案**（Tier-B 末轮权重，SHA256 与 `experiments/RELEASE_MANIFEST.md` 逐位一致）：

| 轴 | 权重 | 预处理 | 任意图上传 |
|---|---|---|---|
| `ws` WorldStrat 6.63× | `ckpts_worldstrat/mtxa_aligned/ckpt_ep29.pt` | per_scene（用上传图自身统计） | ✓ |
| `probav` PROBA-V 3× | `ckpts_probav_n300_s42/mtxa_aligned/ckpt_ep29.pt` | global (mu=0.08206, sd=0.07568)，亮度近似 RED | ✓ |
| `s2v` SEN2VENµS 2× | `ckpts_s2v/sisr/ckpt_ep29.pt` | global (mu=0.07142, sd=0.11740) | ✗（4 波段，仅评测路径） |

**一键预检（/sr_audit）= 无 GT 降级检查 P1–P4**：P1 结构一致性 CC(SR,bicubic)、P2 动态范围 std 比（G3 阈值）、P3 周期伪影 FFT 比（G4 阈值，复用 `sr_audit.periodicity_scan`）、P4 高频能量比（信息项）。**它不是论文 G1–G5**（后者需 GT 配对评测），只回答"这张图本身有没有结构发散/统计异常/周期伪影"。

**真实数据实测示例**：`python code/service/axis_realdata_demo.py` —— 从各轴验证集导出真实 LR 场景 → 反归一化存 PNG → 经网关出图 + 预检 + 三轴审计卡 `/assess` 验真，产物落 `experiments/service_axis_demo/`。实测结果：WS 一键预检独立复现 7px 块网格伪影（P3 WARN 3.63）；PROBA-V 全 PASS；三轴徽章 VERIFIED|positive / VERIFIED|saturated / QUALIFIED|floor-failure。

Python 客户端：`gateway_call_example.py`（一次演示 /health、/sr、/audit、/sr_audit 四个调用）。

---

## 1. 图像超分推理服务（`sr_infer_service.py`）

把训练好的超分模型封装成「上传图片 → 返回超分后图片」的服务。

```bash
python code/service/sr_infer_service.py --port 8766
# 打开 http://127.0.0.1:8766/ 即前端上传示例
```

- 默认加载 `experiments/ckpts/ckpt_ep29.pt`（WorldStrat RGB / scale=4 的早期权重；真实三轴权重属 Tier-B Zenodo 产物，本地缺，用 `--ckpt` 指定即可）。
- 预处理复刻 `data/dataset.py`：RGB `/255` → `[0,1]`；`norm="per_scene"` 时用**输入图自身**的 (mu,sd) 标准化（LR/HR 共用，故任意单图可部署、无需 GT）；多时相 T 帧由单图 repeat 补齐。
- 返回模型 SR 图（PNG）+ bicubic 上采样基线；`?compare=1` 返回 `[bicubic | 模型SR]` 并排对比图。
- 设备自动选 CUDA / CPU；torch CUDA 前用 `_dll_guard` 隐藏 `vcomp140.dll` 防 segfault。

**端点**
| 端点 | 说明 |
|---|---|
| `GET /` | 前端上传示例页 |
| `GET /health` | 健康检查（含 device / 是否加载） |
| `POST /sr` | 上传图片（multipart 字段 `image`，或 JSON `{"image":base64,"compare":bool}`）→ 返回 SR 图（image/png）；`?compare=1` 返回并排对比图；响应头 `X-SR-Meta` 带 scale/out_size/device 等元信息 |

**调用示例**
- 前端：`sr_demo.html`（文件选择 → 实时显示结果）。
- Python：`sr_call_example.py`（标准库 urllib，无需 requests）。

---

## 2. SR-Audit 审计协议服务（`audit_service.py`）

把论文的核心交付物——**SR-Audit 评测审计协议**（G1–G5 门控 + 可信徽章 + 科学 regime 判定）——封装成前端可直接调用的 HTTP 服务。

> 协议层负责"验真"：调用方提供某个超分方法的**评测证据**（逐场景指标），服务返回门控判定、可信徽章（VERIFIED / QUALIFIED / VOID / UNVERIFIED）与科学 regime（positive / saturated / floor-failure），以及配套的 Δ_floor、配对 t 检验（CI / p / Cohen's d）等金标准统计量。徽章与 regime 判定**复用** `code/sr_audit_report.py` 的 `derive()`，与论文报告标准单一来源、口径一致。

---

## 3. 运行（独立审计服务）

```bash
# 需要：CPython 3.8+ 与 numpy（本机用受管解释器）
python code/service/audit_service.py --port 8765
# 启动后访问 http://127.0.0.1:8765/ 即前端调用示例页
```

服务同时提供（均带 CORS `*`，可被任意前端 fetch）：

| 端点 | 方法 | 说明 |
|---|---|---|
| `/` | GET | 自带示例、可实时调用 `/audit` 的前端演示页 |
| `/health` | GET | 健康检查 + 协议版本 |
| `/schema` | GET | 输入契约（JSON 字段说明） |
| `/audit` | POST | 从逐场景证据计算门控 + 徽章 + regime（**主端点**） |
| `/assess` | POST | 直接对已有审计卡片跑 `derive()`（等价 `sr_audit_report.py`） |

---

## 4. 输入契约（POST /audit）

请求体是 JSON，关键字段：

```json
{
  "method": "方法名",
  "task": "任务描述（可选）",
  "scale": 7, "val_dr": 15.06, "split": "val",   // 可选元数据
  "scenes": [
    {
      "psnr_model": 29.38,    // 必填：模型 PSNR(dB)
      "psnr_bicubic": 30.82,  // 必填：双三次基线 PSNR(dB)
      "psnr_mean": 33.08,     // 必填：均值预测器"地板"PSNR(dB)
      "cc_bicubic": 0.303,    // 可选 -> G1 配对完整性
      "cc_model": 0.231,      // 可选 -> G2 结构保真
      "std_model": 0.18, "std_gt": 0.24,  // 可选 -> G3 输出统计（也可直接给 std_ratio）
      "bias_abs": 0.889,      // 可选 -> G3 细节
      "grid_ratio_model": 1.05, "grid_period_model": 6,  // 可选 -> G4 周期伪影
      "leak_stage": false     // 可选 -> G5 泄漏风险（HR 是否进网络）
    }
    // ... 更多场景
  ]
}
```

- **缺失的证据 → 对应门判为 `n/a`**（不会误判 FAIL），便于优雅降级。
- 每个场景的 `psnr_*`、`cc_*`、`std_*`、`grid_*` 都是任何评测流水线都会输出的量，无需权重/GPU。

---

## 5. 输出示例（POST /audit，WorldStrat 6.63× floor-failure 复刻）

```json
{
  "standard": "SR-Audit Report Standard v1",
  "input": { "method": "CrossAttnAlign (mtxa_mis_xa)", "n_test": 60, "scale": 7, "val_dr": 15.06 },
  "stats": {
    "psnr": { "mean_floor": 33.07, "bicubic": 30.81, "model": 29.27 },
    "delta_floor_db": -2.26,
    "floor_gap_db": -3.80,
    "gain_db": -1.54,
    "paired": { "n": 60, "mean_diff": -1.54, "ci": [-1.68, -1.40], "p": 1.4e-34, "cohen_d": -3.43, "t_stat": -20.2, "df": 59 }
  },
  "gates": {
    "g1": { "verdict": "PASS", "detail": "CC(bic,GT)=0.303 (阈值>=0.30)" },
    "g2": { "verdict": "WARN", "detail": "model CC / bicubic CC = 0.76" },
    "g3": { "verdict": "PASS", "detail": "std_ratio=0.76, |bias|=0.8892" },
    "g4": { "verdict": "PASS", "detail": "ratio=1.05 @ period=6px" },
    "g5": { "verdict": "PASS", "detail": "未见泄漏" }
  },
  "audit": {
    "credibility": "QUALIFIED",
    "regime": "floor-failure",
    "gate_verdict": "WARN",
    "reasons": [
      "gate WARN: G2",
      "baseline collapsed: Δ_floor=-2.29 <= 0, model−bicubic not interpretable as skill",
      "model does not clear the mean-predictor skill floor"
    ],
    "headline": "CrossAttnAlign (mtxa_mis_xa) | regime=floor-failure | badge=QUALIFIED | gain=-1.56 dB | p=1.44e-34 | Δ_floor=-2.29 dB"
  }
}
```

完整样例见 `example_request.json`（输入）与 `example_response.json`（输出）。

### 判定口径（与论文一致）

- **G1 配对完整性**：`CC(bicubic(LR),GT) ≥ 0.30` → PASS / FAIL。
- **G2 结构保真**：`CC(model)/CC(bicubic) ≥ 0.9` PASS，≥ 0.7 WARN，否则 FAIL。
- **G3 输出统计**：`std_ratio ∈ [0.5, 1.8]` → PASS / FAIL。
- **G4 周期伪影**：FFT 最强谐波比值 `< 2.5` → PASS / WARN。
- **G5 泄漏风险**：HR 进网络或 PSNR > 45 dB → FAIL / PASS。
- **可信徽章**：任一 FAIL → `VOID`（数字作废）；有 WARN → `QUALIFIED`；全 PASS → `VERIFIED`；无门控 → `UNVERIFIED`。
- **科学 regime**：`Δ_floor ≤ 0` 或 `passes_floor=False` → `floor-failure`；gain 不显著（p ≥ 0.05）→ `saturated`；显著且过地板 → `positive`。

---

## 6. 前端调用示例

打开 `frontend_example.html`（或直接访问 `http://127.0.0.1:8765/`）：左侧填评测证据，点「运行审计」即实时渲染徽章 / regime / 门控表 / 统计量。核心调用就是一段 `fetch`：

```js
const resp = await fetch("http://127.0.0.1:8765/audit", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload)
});
const result = await resp.json();
// result.audit.credibility / result.audit.regime / result.stats.gain_db
```

Python 客户端见 `call_example.py`（仅用标准库 `urllib`，无需安装 `requests`）。

---

## 7. 文件清单

| 文件 | 作用 |
|---|---|
| `gateway_service.py` | **统一网关**：一个端口同时提供出图 + 验真 + 一键（推荐入口） |
| `gateway_demo.html` | 网关统一演示页（`GET /` 即返回它，三个面板） |
| `gateway_call_example.py` | 网关 Python 客户端示例（/health、/sr、/audit、/sr_audit） |
| `axis_realdata_demo.py` | 真实数据实测脚本：各轴验证集场景 → 网关出图/预检/验真 |
| `audit_service.py` | 审计协议服务（stdlib http.server + numpy，零额外依赖） |
| `frontend_example.html` | 审计服务前端示例页 |
| `call_example.py` | 审计服务 Python 客户端示例（urllib） |
| `example_request.json` | 审计输入样例（60 景，复刻 WorldStrat floor-failure） |
| `example_response.json` | 对应输出样例 |
| `sr_infer_service.py` | 图像超分推理服务（上传图 → 返回超分 PNG；依赖 torch + 权重；支持多模型档案） |
| `sr_demo.html` | 推理服务前端示例页（`GET /` 即返回它，可上传图片） |
| `sr_call_example.py` | 推理服务 Python 客户端示例（urllib 发 multipart FormData） |
| `README.md` | 本文件 |

> 配套评测工具：`code/audit_axis.py` —— 三轴通用审计器（sr_audit.py 的 --cfg 泛化版），产出 `/assess` 直接可判的双 schema 审计卡（`experiments/audit_axis_*.json`）。
> 注：**审计层**（audit_service.py / gateway 的 /audit）不依赖 torch；**推理层**（sr_infer_service.py / gateway 的 /sr、/sr_audit）需要 torch 与 Tier-B 末轮权重。

> 注：**审计服务**（audit_service.py）不依赖 torch / 权重 / GPU；**推理服务**（sr_infer_service.py）需要 torch 与一份训练好的权重（`--ckpt` 指定，默认 `experiments/ckpts/ckpt_ep29.pt`）。两者互补：审计服务评估"数字可信度"（G1–G5 + 徽章 + regime），推理服务产出"超分后的图像"。若需把本项目已训练的模型指标直接转成审计卡片，请用 `code/sr_audit_report.py` 处理 `experiments/audit_*.json`；本服务与之共用同一套徽章/regime 逻辑。

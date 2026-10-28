# Provenance & License Audit / 来源与许可审计档案（P0）

> 仓库：SpectralOptics（自 GemologyNotes-iOS-v3 提取的光学渲染引擎）
> 审计日期：2026-10-28
> 结论先行：**引擎全部着色器血统均处于上游 MIT 授权窗口内，可整体以 MIT 开源。**

---

## 1. 审计范围

P0 目标是在提取代码之前固定许可链：

1. 确定着色器翻译所依据的上游版本（精确到 blob）；
2. 确认我方全部修改都发生在上游 MIT 授权边界之内；
3. 归档上游许可文本与翻译源快照作为永久证据；
4. 记录我方原创增量清单。

---

## 2. 上游许可边界（已核查）

上游仓库：`yuyou-dev/OpenGemCutting`。

| 时间 | 上游事件 | 许可状态 |
|---|---|---|
| 2026-09-11 ~ 10-08 | v1.0.0 → v2.1.0（边界提交 `e8cfa6c`） | **MIT**（LEGACY_MIT_LICENSE.txt，版权行 "Copyright (c) 2026 Yuyou and OpenGemCutting contributors"） |
| 2026-10-14 | v2.1.1 许可迁移 | 新原创内容改为 Personal Non-Commercial License 1.0；**v2.1.0 及此前内容继续保留 MIT** |

上游 LICENSING.md（2026-10 现行版）关键原文（已存档 `upstream/LICENSING.md`）：

> 既有 MIT 发布：截至迁移前公开 main `e8cfa6c906001c3a28fc6f786a437cbbf3a07e2e`
> （v2.1.0 发布合并）以及此前按 MIT 提供的内容，继续保留原有授权。旧文本完整保存于
> LEGACY_MIT_LICENSE.txt。相同代码换一个许可证文件不能消除旧授权。

即：**MIT 保留边界 = 提交 `e8cfa6c`（2026-10-08）**，此后才进入非商业许可。

---

## 3. 翻译源定位（精确到 blob）

我方着色器头注释声明翻译自 `OpticsViewport.jsx` 的 `FRAGMENT_SHADER`。
通过本机上、游克隆（`/Users/soli/Documents/WORKS/OpenGemCutting`，HEAD 恰为
MIT 边界提交 `e8cfa6c`）定位：

| 项 | 值 |
|---|---|
| 源文件 | `src/components/OpticsViewport.jsx` |
| 翻译时点 | 2026-09-18（我方提交 `bce31cc`） |
| 该时点文件最后变更 | `dbc623e`（2026-09-14） |
| 对应 Blob SHA-1 | `fff810ce4e825df209882cf5f696c86dcd6f879d`（744 行） |
| 下一次上游变更 | `fb9b7fb`（2026-09-21，外部贡献者 EthanSMC，WebGPU 后端——上游 LICENSING.md 明文该提交亦按当时 MIT 提供） |
| 祖先关系 | `dbc623e` 是 `e8cfa6c`（MIT 边界）的祖先（`git merge-base --is-ancestor` 验证通过） |

**翻译源快照已归档**：`upstream/OpticsViewport.jsx.fff810ce.txt`。

结构对应抽查（`git show fff810ce` vs 我方 MSL）——上游 GLSL 与我方 MSL 逐项对应：

| 上游 GLSL（fff810ce） | 我方 MSL（optics_lab.metal） |
|---|---|
| `MAX_BOUNCES = 8` | `#define MAX_BOUNCES 8` |
| `EPSILON = 0.0012` | `#define EPSILON 0.0012` |
| `intersectConvex` / `nextBoundary` | 同名函数 |
| `traceGem(..., absorptionColor)` + dielectricFresnel | `traceGemSpectral`（光谱升级版） |
| 红/绿/蓝三族波长共享几何（`redIor/uIor/blueIor`） | 同构三族路径 |
| `traceMeshGem`（网格路径） | `traceMeshSpectral` |

头注释的来源声明与实际代码证据一致。

---

## 4. 我方着色器/渲染器变更历史审计

`optics_lab.metal` 与 `OpticsRenderer.swift` 完整 git 血统（`--follow`，5 个提交）：

| 提交 | 日期 | shader 增/删 | 性质 | 与上游关系 |
|---|---|---|---|---|
| `bce31cc` | 2026-09-18 | +275 | 初始：GLSL→MSL 逐行翻译 + 31-bin 光谱升级 | 翻译源 = fff810ce（MIT 窗口内 ✓） |
| `512ba40` | 2026-09-23 | +1258/−49 | 原创：偏光实验室（正交偏光 3D、Jones/密度矩阵、二轴晶载物台修正） | 上游无对应功能，纯原创 ✓ |
| `97df9c5` | 2026-10-02 | +1014/−49 | 原创：锥光干涉（Jones 累积/显示双管线）、晶胞渲染（解析球+胶囊）、光轴叠加、量子轨迹 pass、三偏光片 | 上游无对应功能，纯原创 ✓（时点仍在 MIT 边界 10-08 之前） |
| `5ae716d` | 2026-10-04 | +33/−19 | 原创：光轴由构造法给出（替代几何反解）、清理 | ✓ |
| `37f9ea4` | 2026-10-04 | +2/−1 | 原创：切磨工作台能力位收敛的微调 | ✓ |

**关键时间线结论**：着色器全部变更（2026-09-18 ~ 2026-10-04）早于上游 MIT
边界（2026-10-08）。即便后续提交曾参考上游 main，彼时上游全部内容仍为 MIT。
血统干净，无任何非商业许可内容混入的可能。

### 工作区未提交改动（截至审计日）

| 文件 | 改动 | 性质 |
|---|---|---|
| `OpticsMesh.swift` | +18/−5：`fixedNormalization`（切磨工作台取景钉毛坯帧） | 原创功能 ✓ |
| `OrbitGesture.swift` | +80：`OrbitPanGestureRecognizer`（pressLocation，按下点命中 gizmo） | 原创功能 ✓ |

两处均为对上游无来源关系的原创增量，随 P1 一并提取时按 MIT 提供。

---

## 5. 不进入本包的资产（许可隔离说明）

以下内容**不属于**本仓库，留在 GemologyNotes App 内：

- 宝石材质库 `optics_materials.json`、252 琢型库 `cuts/`（源自上游数据蒸馏，上游数据许可另行评估）；
- 切磨预设 `faceting-presets/`（facetdiagrams.com，MIT，属数据非代码）；
- SMORF 晶形库、CIF 晶体数据、宝石学文章与付费墙逻辑。

引擎包只含通用渲染代码，与上述数据零耦合（`CutModels.swift`/`OpticsHabitOrientation.swift` 不进包）。

---

## 6. P0 交付物清单

- [x] `LICENSE`（MIT，JI GUO (郭际)）
- [x] `THIRD_PARTY_NOTICES.md`（上游出处 + blob 定位 + 边界依据）
- [x] `docs/compliance/PROVENANCE.md`（本档案）
- [x] `docs/compliance/upstream/LEGACY_MIT_LICENSE.txt`（上游 MIT 原文存档）
- [x] `docs/compliance/upstream/LICENSING.md`（上游许可迁移说明存档，2026-10-28 取自 main）
- [x] `docs/compliance/upstream/OpticsViewport.jsx.fff810ce.txt`（翻译源快照，744 行）

## 7. 后续动作（P1 承接）

1. P1 提取代码时，把着色器头注释更新为指向本仓库 `THIRD_PARTY_NOTICES.md` 与上游 blob SHA；
2. Example 与 README 复述来源声明（对外可见面的义务）；
3. 上游 blob 快照与本档案随首个公开 tag 一起发布，保证任何时点可自证清白；
4. 若未来引擎需吸收上游 post-v2.1.1 的新内容（如 WebGPU 后端思路），必须先重新评估许可（其新内容为非商业许可）——**在 NOTICE 中明令禁止直接搬运**。

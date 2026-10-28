# Third-Party Notices / 第三方声明

本仓库（SpectralOptics）包含源自第三方项目的代码。以下列明其来源、许可与衍生关系。

---

## OpenGemCutting（部分衍生来源）

- **上游项目**：<https://github.com/yuyou-dev/OpenGemCutting>
- **上游许可**：MIT License，Copyright (c) 2026 Yuyou and OpenGemCutting contributors
  （原文副本见本仓库 `docs/compliance/upstream/LEGACY_MIT_LICENSE.txt`）
- **衍生关系**：本包中的 Metal 光线追踪着色器最初由上游
  `src/components/OpticsViewport.jsx` 中的 WebGL2 GLSL 300 ES fragment shader
  （`FRAGMENT_SHADER` 常量）逐行翻译为 Metal Shading Language，并在此基础上
  进行了原创的光谱升级（RGB 体色吸收 → 31 波长 / 400–700nm / 10nm 真光谱路径追踪）
  及后续原创扩展（偏光实验室、锥光干涉、晶胞渲染、量子轨迹绘制等）。
- **翻译时点所依据的上游版本**：
  - 文件：`src/components/OpticsViewport.jsx`
  - Blob SHA-1：`fff810ce4e825df209882cf5f696c86dcd6f879d`（744 行）
  - 所在提交：`dbc623e`（2026-09-14，"feat: add starter projects and refresh
    conversational design guidance"）
  - 该 blob 原文存档于 `docs/compliance/upstream/OpticsViewport.jsx.fff810ce.txt`
- **许可边界依据**：上游自 v2.1.1（2026-10-14）起将新原创内容迁移至
  OpenGemCutting Personal Non-Commercial License 1.0，但明文规定截至
  `e8cfa6c906001c3a28fc6f786a437cbbf3a07e2e`（v2.1.0 发布合并，2026-10-08）
  及此前按 MIT 提供的内容**继续保留 MIT 授权**，且该授权不因上游后续换发
  许可证文件而收回。上述翻译源 blob 位于该 MIT 边界之内。
  上游迁移说明原文存档于 `docs/compliance/upstream/LICENSING.md`。
- **本仓库对应部分的许可**：MIT（与本仓库 LICENSE 一致），衍生部分与原创部分均按
  MIT 提供。来源声明同时保留在着色器源码头注释中。

---

除上述来源外，本仓库其余代码均为原创（Copyright (c) 2026 JI GUO (郭际)，MIT）。

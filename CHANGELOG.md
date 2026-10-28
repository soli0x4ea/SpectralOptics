# Changelog

## 0.1.0 — 2026-10-28

首个公开版本。自「系统宝石学笔记」App（GemologyNotes-iOS-v3）提取的独立
Swift Package，提取流程 P0–P5 全部完成。

### OpticsCore

- 31 波长（400–700nm / 10nm）真光谱路径追踪：凸多面体（半空间平面表）、
  解析球面、三角网格（BVH，凹面支持）三条几何路径
- 偏光实验室：正交偏光 3D（Jones 矩阵 / 偏振密度矩阵、光率体定向、
  Fresnel 菱体相位）、锥光干涉图（Jones 解析 + 真实方向像 rgba16Float 累积）、
  钠光 / 白光双模式、三偏光片、旋光色散
- 晶胞渲染（解析原子球 + 胶囊化学键）、光轴指示叠加层（含逐像素遮挡）、
  白模与磨砂、量子轨迹绘制 pass
- 真实照明体 SPD（D65 / A / 烛光 / E / 钠光）→ CIE 1931 2° 积分 →
  linear sRGB → ACES 色调映射
- `OrbitCamera` 轨道相机、`PolarGemPose` 宝石姿态四元数
- 可测试性：CPU/GPU 双侧同构参考实现、Uniform 二进制布局断言

### OpticsOrbit（iOS / Catalyst）

- `OrbitMTKView` 视口、`OrbitGestureController` 手势状态机
  （单指轨道、双指平移、捏合缩放、双击复位、Catalyst 滚轮、覆盖层命中托管）

### 工程

- P0 来源与许可审计（`docs/compliance/PROVENANCE.md`，上游 OpenGemCutting
  MIT 窗口内核验，blob 级存档）
- P2 去域化清洗、P3 测试迁移（32 测试全绿）
- P4 README 使用指南 + macOS 示例 `OpticsExample`

# SpectralOptics

Metal 光谱光学渲染引擎（spectral path-traced optics renderer for iOS/macOS）——
自「系统宝石学笔记」App 提取的独立 Swift Package。**正在提取中（P0 合规阶段已完成，P1 代码提取未开始）。**

## 状态

| 阶段 | 状态 |
|---|---|
| P0 许可与来源合规 | ✅ 完成（见 `docs/compliance/PROVENANCE.md`） |
| P1 机械提取（SPM 骨架 + 引擎源码 public 化 + `makeLibrary(source:)` 加载） | ✅ 完成（MSL 独立编译 / 双平台 typecheck / SwiftPM build / macOS 运行时冒烟全绿） |
| P2 去域化清洗 / P3 测试迁移 / P4 文档与 Example / P5 发布 0.1.0 | ⬜ 未开始 |

## 能力（规划）

- 31 波长（400–700nm / 10nm）真光谱凸多面体/网格路径追踪（Fresnel、全内反射、B–G 色散、Beer–Lambert 主轴吸收）
- 正交偏光 3D（Jones 矩阵 / 偏振密度矩阵、光率体定向、Fresnel 菱体相位）
- 锥光干涉图（Jones 解析 + 真实方向像 rgba16Float 累积）、钠光/白光双模式
- 晶胞渲染（CPK 原子球 + 化学键解析求交）、光轴叠加层
- 量子轨迹绘制 pass（GPU 表现层；物理模拟留在宿主 App）
- 真实照明体 SPD（D65 / A / 烛光 / E / 钠光）→ CIE 1931 2° 积分 → linear sRGB → ACES 色调映射
- CPU/GPU 双侧同构参考实现 + uniform 布局断言（可测试性内建于设计）

## 许可

- 本仓库代码：MIT（`LICENSE`）
- 含源自 OpenGemCutting（MIT）的衍生部分，出处与许可边界见
  `THIRD_PARTY_NOTICES.md` 与 `docs/compliance/`。

# SpectralOptics

Metal 光谱光学渲染引擎（spectral path-traced optics renderer for iOS/macOS）——
自「系统宝石学笔记」App 提取的独立 Swift Package。

## 状态

| 阶段 | 状态 |
|---|---|
| P0 许可与来源合规 | ✅ 完成（见 `docs/compliance/PROVENANCE.md`） |
| P1 机械提取（SPM 骨架 + 引擎源码 public 化 + `makeLibrary(source:)` 加载） | ✅ 完成 |
| P2 去域化清洗 | ✅ 完成（App 内部概念引用已清除；域术语保留） |
| P3 测试迁移 | ✅ 完成（32 个测试全绿：Uniform 二进制布局、BVH/求交、偏光姿态、照明体表、**GPU↔CPU 逐像素 diff harness**、真实设备渲染器冒烟） |
| P4 文档与 Example | ✅ 完成（README 使用指南 + `swift run OpticsExample`） |
| P5 发布 0.1.0 | ✅ 完成（tag `0.1.0`，见 `CHANGELOG.md`） |

## 能力

- 31 波长（400–700nm / 10nm）真光谱凸多面体/网格路径追踪（Fresnel、全内反射、B–G 色散、Beer–Lambert 主轴吸收）
- 正交偏光 3D（Jones 矩阵 / 偏振密度矩阵、光率体定向、Fresnel 菱体相位）
- 锥光干涉图（Jones 解析 + 真实方向像 rgba16Float 累积）、钠光/白光双模式
- 晶胞渲染（CPK 原子球 + 化学键解析求交）、光轴叠加层
- 量子轨迹绘制 pass（GPU 表现层；物理模拟留在宿主 App）
- 真实照明体 SPD（D65 / A / 烛光 / E / 钠光）→ CIE 1931 2° 积分 → linear sRGB → ACES 色调映射
- CPU/GPU 双侧同构参考实现 + uniform 布局断言（可测试性内建于设计）

## 安装

```swift
// Package.swift
dependencies: [
    .package(path: "SpectralOptics")  // 或 git URL
]
```

两个 product：

| Product | 内容 | 平台 |
|---|---|---|
| `OpticsCore` | 渲染器、网格/BVH、姿态数学、照明体 SPD（纯 Metal，无 UIKit） | iOS 15+ / macOS 12+ |
| `OpticsOrbit` | `OrbitMTKView` + 轨道手势状态机（UIKit） | iOS / Catalyst |

## 快速上手（OpticsCore）

```swift
import MetalKit
import OpticsCore

let renderer = OpticsRenderer.make()!          // 或 OpticsRenderer(device:queue:)
let view = MTKView(frame: bounds, device: renderer.device)
renderer.attach(to: view)                       // 配置 pixelFormat / isPaused 等

// 几何：任意闭合三角网格（保留凹面，永不做凸包）
let mesh = OpticsMeshBuilder.build(.init(
    vertices: myVertices,                       // [SIMD3<Double>]，任意尺度
    faces: myFaces,                             // [[Int]]，扇形三角化
    faceNormals: myNormals))!                   // 每面参考外法向

// 渲染状态（值类型，整体替换）
var state = OpticsRenderState(
    planes: [],                                 // 凸体半空间平面表（与 mesh 三选一）
    ior: 2.417, dispersion: 0.044,
    absorptionAxisX: .zero, absorptionAxisY: .zero, absorptionAxisZ: .zero,
    spectrumAxes: nil,                          // [[SIMD4<Float>]]：三主轴 × 31 波长吸收
    illuminant: 0,                              // 0 D65 / 1 A / 2 烛光 / 3 E / 4 钠光
    environment: 0, background: SIMD3(0.94, 0.95, 0.95),
    exposure: 0, environmentRotation: 18,
    maxBounces: 6,
    camera: OrbitCamera(),
    polar: nil,                                 // 偏光实验室传 OpticsPolarRenderState
    mesh: mesh)                                 // 凹面网格路径
renderer.update(state)

// 相机 / 状态变化后：
state.camera = newCamera
renderer.update(state)
renderer.requestRender(in: view)
```

偏光实验室（正交偏光 3D / 锥光干涉）只需给 `polar` 传 `OpticsPolarRenderState`，
晶体光学参数（轴性、主折射率、半光轴角 2V、光轴定向）见该结构体字段文档；
`mode`：`0` 普通 / `1` 正交偏光 3D / `2` 锥光干涉图。

## 快速上手（OpticsOrbit，iOS / Catalyst）

```swift
import OpticsOrbit

let view = OrbitMTKView(frame: .zero, device: renderer.device)
renderer.attach(to: view)

let gesture = OrbitGestureController(camera: state.camera)
gesture.attach(to: view)
gesture.onChange = { camera in
    state.camera = camera
    renderer.update(state)
    renderer.requestRender(in: view)
}
// 偏光实验室等「手指扳物体」场景：
// gesture.dragTarget = .object; gesture.onObjectPan = { dxDeg, dyDeg in ... }
```

## Example

macOS 可执行示例（AppKit + MetalKit，钻石参数的八面体光谱路径追踪）：

```sh
swift run OpticsExample
```

拖拽旋转、滚轮缩放；交互语义与 `OpticsOrbit` 保持一致。

## 测试

```sh
swift test
```

32 个测试：Uniform 二进制布局断言、BVH 构建与求交、偏光姿态四元数、照明体 SPD 表、
GPU↔CPU 逐像素 diff harness（需 Metal 设备）、渲染器冒烟。

## 文档地图

- `docs/compliance/PROVENANCE.md` —— 来源与许可审计档案（P0）
- `THIRD_PARTY_NOTICES.md` —— 第三方声明（对外分发必读）
- `CHANGELOG.md` —— 版本历史
- 着色器与渲染器的契约（Uniform 布局、缓冲布局）见 `OpticsRenderer.swift`
  与 `optics_lab.metal` 的头注释——改任何一侧都必须同步另一侧。

## 许可与来源

- 本仓库代码：MIT（`LICENSE`，Copyright (c) 2026 JI GUO (郭际)）。
- 含源自 [OpenGemCutting](https://github.com/yuyou-dev/OpenGemCutting)（MIT）的
  衍生部分：Metal 着色器译自其 `OpticsViewport.jsx` 的 WebGL2 fragment shader
  （blob `fff810ce`，处于上游 MIT 授权窗口内），并在其上做了原创的光谱升级、
  偏光实验室、锥光干涉、晶胞渲染等扩展。出处、blob 定位与许可边界依据见
  `THIRD_PARTY_NOTICES.md` 与 `docs/compliance/`。

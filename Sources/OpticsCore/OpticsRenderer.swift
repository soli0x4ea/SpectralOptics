import Metal
import MetalKit
import simd

//
//  OpticsRenderer.swift
//  SpectralOptics
//
//  Metal 光谱光学渲染器。自 GemologyNotes（光学实验室）原样提取并 public 化。
//  物理模型与着色器族说明见 Resources/optics_lab.metal 头注释；
//  来源与许可见仓库 THIRD_PARTY_NOTICES.md / docs/compliance/PROVENANCE.md。
//
//  与着色器的契约（改任何一侧都必须同步另一侧）：
//   - OpticsUniforms ↔ optics_lab.metal 的 Uniforms（336 字节，断言在 init 尾部）
//   - OpticsMeshState ↔ intersectMeshBVH / traceMeshSpectral 的缓冲布局
//   - OpticsCellState / OpticsAxisState ↔ optics_cell_fragment / optics_axis_fragment
//

// MARK: - Uniforms（必须与 optics_lab.metal 的 Uniforms 二进制一致）

public struct OpticsUniforms {
    public var cameraPositionScale: SIMD4<Float>      // xyz position, w scale
    public var cameraForwardExposure: SIMD4<Float>    // xyz forward, w exp2 exposure
    public var cameraRightEnvRot: SIMD4<Float>        // xyz right,   w env rotation deg
    public var cameraUpFocal: SIMD4<Float>            // xyz up,      w focal offset
    public var bodyColorIor: SIMD4<Float>             // xyz body color（预留）, w ior
    public var observerDispersion: SIMD4<Float>       // xyz observer dir, w dispersion
    public var backgroundAbsorption: SIMD4<Float>     // xyz background, w unused
    public var absorptionAxisX: SIMD4<Float>          // xyz 晶体 X 主轴吸收, w unused
    public var absorptionAxisY: SIMD4<Float>          // xyz 晶体 Y 主轴吸收, w unused
    public var absorptionAxisZ: SIMD4<Float>          // xyz 晶体 Z 主轴（台面法线）, w unused
    public var panParams: SIMD4<Float>                // x panX, y panY, z aspect, w unused
    public var flags: SIMD4<Int32>                    // x environment, y bounces, z planeCount, w illuminant
    public var resolution: SIMD2<Float>
    // —— 偏光实验室（208 字节之后追加；普通模式全部为 0 时忽略）——
    public var polarParams: SIMD4<Float>              // x 模式0普通/1正交3D, y 分析镜绝对方位rad, z 载物台rad, w 板厚mm
    public var crystalA: SIMD4<Float>                 // x kind 0均质/1一轴+/2一轴-/3二轴+/4二轴-, y nO/n, z nE
    public var crystalB: SIMD4<Float>                 // x nA, y nB, z nG, w 半光轴角 V(rad)
    public var polarFlags: SIMD4<Float>               // x 钠光1/白光0, y 锥光视场半角rad, z 旋光谱点数, w 真实锥光累积帧序号
    public var crystalOrient: SIMD4<Float>            // x 光轴(二轴晶=光轴中线Bxa)倾角rad, y 方位角rad, z 光轴面滚转rad, w unused
    /// xyz 球心, w 半径；w > 0 时着色器走解析球面求交（此时 flags.z 平面数必须为 0）
    public var sphereParams: SIMD4<Float>
    /// 三偏光片（与 metal 侧 Uniforms 末尾同步追加）：x 下偏光片方位rad,
    /// y 中偏光片方位rad, z 中偏光片启用1/0, w 宝石自转角rad。
    /// 分析镜绝对方位在 polarParams.y；三者都是实验室系绝对角（0°=+x）。
    ///
    /// w 是「宝石自己在台上转」的记账（水平拖拽）：与 polarParams.z（载物台角）
    /// 同轴同向，样品总方位角 = 两者之和（shader 的 specimenSpin），而**可见的
    /// 载物台圆盘只读 polarParams.z** —— 转宝石不该连台面刻度一起转。
    public var polarLab: SIMD4<Float>
    /// 宝石在台面系里的姿态，单位四元数 (ix, iy, iz, r)：手指扳宝石的自由度
    /// （倾转 / 翻面）。世界←宝石 = Rz(polarParams.z + polarLab.w) · 本四元数，
    /// 推导见 PolarGemPose / optics_lab.metal 的 GemPose。
    /// 普通模式与锥光写 identity (0,0,0,1)。
    public var gemTumble: SIMD4<Float>

    public init(cameraPositionScale: SIMD4<Float>,
                cameraForwardExposure: SIMD4<Float>,
                cameraRightEnvRot: SIMD4<Float>,
                cameraUpFocal: SIMD4<Float>,
                bodyColorIor: SIMD4<Float>,
                observerDispersion: SIMD4<Float>,
                backgroundAbsorption: SIMD4<Float>,
                absorptionAxisX: SIMD4<Float>,
                absorptionAxisY: SIMD4<Float>,
                absorptionAxisZ: SIMD4<Float>,
                panParams: SIMD4<Float>,
                flags: SIMD4<Int32>,
                resolution: SIMD2<Float>,
                polarParams: SIMD4<Float>,
                crystalA: SIMD4<Float>,
                crystalB: SIMD4<Float>,
                polarFlags: SIMD4<Float>,
                crystalOrient: SIMD4<Float>,
                sphereParams: SIMD4<Float>,
                polarLab: SIMD4<Float>,
                gemTumble: SIMD4<Float>) {
        self.cameraPositionScale = cameraPositionScale
        self.cameraForwardExposure = cameraForwardExposure
        self.cameraRightEnvRot = cameraRightEnvRot
        self.cameraUpFocal = cameraUpFocal
        self.bodyColorIor = bodyColorIor
        self.observerDispersion = observerDispersion
        self.backgroundAbsorption = backgroundAbsorption
        self.absorptionAxisX = absorptionAxisX
        self.absorptionAxisY = absorptionAxisY
        self.absorptionAxisZ = absorptionAxisZ
        self.panParams = panParams
        self.flags = flags
        self.resolution = resolution
        self.polarParams = polarParams
        self.crystalA = crystalA
        self.crystalB = crystalB
        self.polarFlags = polarFlags
        self.crystalOrient = crystalOrient
        self.sphereParams = sphereParams
        self.polarLab = polarLab
        self.gemTumble = gemTumble
    }
}

/// 偏光实验室渲染参数（与 optics_lab.metal 偏光段对应；普通模式为 nil）
public struct OpticsPolarRenderState {
    /// 0 普通 / 1 正交偏光 3D / 2 锥光干涉图
    public var mode: Int
    public var analyzerAngleRad: Float
    /// 下偏光片（起偏镜）绝对方位（rad）
    public var polarizerAngleRad: Float
    /// 第三片偏振片（中偏光片）绝对方位（rad）
    public var midPolarizerAngleRad: Float
    /// 中偏光片是否插入光路
    public var midPolarizerOn: Bool
    /// 载物台角（rad）：只喂可见的盘面刻度/金指针与角标「台 x°」。
    /// 样品的总方位角还要加 `gemSpinRad`（见下）。
    public var stageRad: Float
    /// 宝石自转角（rad）：手指水平拖拽拨的。与 `stageRad` 同轴同向（绕世界 z），
    /// 所以样品的世界系姿态只取决于两者之和 —— 拆开纯为记账：**这个量不喂盘面、
    /// 不喂台角读数**，否则「转宝石时载物台刻度跟着转」。
    /// 喂给：宝石网格/光率体/光轴标记（正交 3D）与锥光的样品方位。
    public var gemSpinRad: Float = 0
    public var plateThicknessMm: Float
    /// 0 均质 / 1 一轴+ / 2 一轴- / 3 二轴+ / 4 二轴-
    public var crystalKind: Int
    public var nO: Float
    public var nE: Float
    public var nA: Float
    public var nB: Float
    public var nG: Float
    /// 二轴晶半光轴角 V = 2V/2（rad）
    public var halfVrad: Float
    /// 锥光使用钠光单色（false=D65 白光等色环）
    public var sodium: Bool
    /// 锥光视场半角（rad）
    public var fovHalfAngleRad: Float
    /// 旋光色散采样（波长 nm, °/mm），最多 6 对
    public var rotationSamples: [SIMD2<Float>]
    /// 光轴相对台面法线的定向（rad）：倾角 / 方位 / 二轴晶光轴面滚转
    public var axisTiltRad: Float
    public var axisAzimuthRad: Float
    public var axisRollRad: Float
    /// 宝石在台面系里的姿态（单位四元数 ix,iy,iz,r）；identity = 正立在台上。
    /// 与 axisTilt/axisAzimuth/axisRoll 是**两个独立的旋钮**：后者转的是宝石
    /// 内部的光率体（光轴相对晶体的定向，一个物理常数），前者转的是整颗宝石
    /// （网格几何 + 光率体一起刚体转动）。
    public var gemTumble: SIMD4<Float> = SIMD4<Float>(0, 0, 0, 1)

    public init(mode: Int,
                analyzerAngleRad: Float,
                polarizerAngleRad: Float,
                midPolarizerAngleRad: Float,
                midPolarizerOn: Bool,
                stageRad: Float,
                gemSpinRad: Float = 0,
                plateThicknessMm: Float,
                crystalKind: Int,
                nO: Float,
                nE: Float,
                nA: Float,
                nB: Float,
                nG: Float,
                halfVrad: Float,
                sodium: Bool,
                fovHalfAngleRad: Float,
                rotationSamples: [SIMD2<Float>],
                axisTiltRad: Float,
                axisAzimuthRad: Float,
                axisRollRad: Float,
                gemTumble: SIMD4<Float> = SIMD4<Float>(0, 0, 0, 1)) {
        self.mode = mode
        self.analyzerAngleRad = analyzerAngleRad
        self.polarizerAngleRad = polarizerAngleRad
        self.midPolarizerAngleRad = midPolarizerAngleRad
        self.midPolarizerOn = midPolarizerOn
        self.stageRad = stageRad
        self.gemSpinRad = gemSpinRad
        self.plateThicknessMm = plateThicknessMm
        self.crystalKind = crystalKind
        self.nO = nO
        self.nE = nE
        self.nA = nA
        self.nB = nB
        self.nG = nG
        self.halfVrad = halfVrad
        self.sodium = sodium
        self.fovHalfAngleRad = fovHalfAngleRad
        self.rotationSamples = rotationSamples
        self.axisTiltRad = axisTiltRad
        self.axisAzimuthRad = axisAzimuthRad
        self.axisRollRad = axisRollRad
        self.gemTumble = gemTumble
    }
}

/// 晶胞模式渲染输入（原子球/化学键/晶胞框；optics_cell_fragment 解析求交）。
/// 打包布局与 optics_lab.metal 晶胞段逐字段对应：
/// 原子每颗 2×SIMD4 —— (pos.xyz, radius) (rgb, 0)；
/// 胶囊每根 3×SIMD4 —— (a.xyz, radius) (b.xyz, 0) (rgb, 0)。
public struct OpticsCellState {
    public var atoms: [SIMD4<Float>]
    public var capsules: [SIMD4<Float>]
    /// 归一化后顶点最大模长（取景用，与 OpticsMeshState.radius 同口径）
    public var radius: Float

    public init(atoms: [SIMD4<Float>], capsules: [SIMD4<Float>], radius: Float) {
        self.atoms = atoms
        self.capsules = capsules
        self.radius = radius
    }

    public var atomCount: Int { atoms.count / 2 }
    public var capsuleCount: Int { capsules.count / 3 }

    public static let maxAtoms = 512
    public static let maxCapsules = 1024
}

/// 光轴指示层渲染输入：叠加在主结果之上的紫色发光虚线
/// （与量子轨迹光线同款：核心线 + 柔和辉光，屏幕空间画）。
/// 打包与 optics_lab.metal 轴段逐字段对应——每根 3×SIMD4：
/// (a.xyz, 线宽px) (b.xyz, 0) (rgb, alpha)。坐标系必须是网格归一化系
/// （AABB 居中、最长边→[-1,1]，与 OpticsMeshBuilder 同变换）。
public struct OpticsAxisState {
    public var capsules: [SIMD4<Float>]

    public init(capsules: [SIMD4<Float>]) {
        self.capsules = capsules
    }

    public var capsuleCount: Int { capsules.count / 3 }

    // 256：宿主 gizmo 的刀口描边一条盖面边一根胶囊（×32 圈可达 ~160 根）；
    // shader 按 axisParams.x 提前 break，上限抬高不增运行时开销
    public static let maxCapsules = 256
}

/// 渲染输入（值类型，宿主侧在 update 时整体比对）
public struct OpticsRenderState {
    public var planes: [SIMD4<Float>]
    public var ior: Float
    public var dispersion: Float
    /// 已停用：shader 从不读 u.absorptionAxis*，该槽位已复用为白模通道
    /// （见 makeUniforms / optics_lab.metal 注释）。保留字段只为构造点兼容
    public var absorptionAxisX: SIMD3<Float>
    public var absorptionAxisY: SIMD3<Float>
    public var absorptionAxisZ: SIMD3<Float>
    /// 三主轴 × 8 个 SIMD4（31 波长 400–700nm/10nm，末位补 0）；
    /// 为 nil 时光谱吸收按 0 处理（近无色）。
    public var spectrumAxes: [[SIMD4<Float>]]?
    /// 照明光源：0 D65 / 1 A 2856K / 2 烛光 1900K / 3 E
    public var illuminant: Int
    public var environment: Int
    public var background: SIMD3<Float>
    public var exposure: Float
    public var environmentRotation: Float
    public var maxBounces: Int
    public var camera: OrbitCamera
    /// 偏光实验室参数；nil = 普通路径追踪
    public var polar: OpticsPolarRenderState?
    /// 解析球面：xyz 球心 / w 半径（归一化空间）。非 nil 时忽略 `planes`。
    /// 半空间表示每个面只有一个法向，平面逼近永远是刻面球 —— 光滑球体只能靠解析求交。
    public var sphere: SIMD4<Float>? = nil
    /// 三角网格路径（凹面支持，OpticsMesh.swift 构建）。非 nil 时走
    /// optics_lab.metal 的网格分支，忽略 `planes` 与 `sphere`；
    /// 可与 `polar` 叠加（偏光 3D 对网格做同样的 Jones 演算）。
    public var mesh: OpticsMeshState? = nil
    /// 晶胞模式（原子/化学键/晶胞框，optics_cell_fragment 解析求交）。
    /// 与 mesh/polar 互斥——三套几何分支共享相机与背景，但一次只用一套。
    public var cell: OpticsCellState? = nil
    /// 光轴指示层（紫色虚线，optics_axis_fragment alpha 叠加）。
    /// 仅网格模式有意义；nil = 不叠加。
    public var axes: OpticsAxisState? = nil
    /// 磨砂遮罩强度 0...1（仅网格路径生效）。经 bodyColorIor.x 这个
    /// 历史空槽位进 shader（见 makeUniforms 注释），0 = 通透默认
    public var frost: Float = 0
    /// 白模（教学晶体模型）：不透光漫反射，替代折射路径追踪（仅网格路径生效）。
    /// 与 frost 互斥使用——白模无体色，磨砂加深没有意义
    public var whiteModel: Bool = false
    /// 白模反照率（当前主题色下的"白"）。经停用的 absorptionAxisX.xyz 进 shader
    public var whiteAlbedo: SIMD3<Float> = SIMD3<Float>(0.93, 0.95, 0.97)
    /// 光轴指示层被晶体遮挡：true 时紫色虚线光轴对网格做逐像素 BVH 遮挡
    /// （进入晶体内部即被挡住不发光），与白模遮挡同源但互相独立——实验室透明宝石
    /// 想让光轴"藏进晶体里"时用它，而不必把整颗染成白模。打包进 panParams.w。
    public var occludeAxes: Bool = false

    public init(planes: [SIMD4<Float>],
                ior: Float,
                dispersion: Float,
                absorptionAxisX: SIMD3<Float>,
                absorptionAxisY: SIMD3<Float>,
                absorptionAxisZ: SIMD3<Float>,
                spectrumAxes: [[SIMD4<Float>]]?,
                illuminant: Int,
                environment: Int,
                background: SIMD3<Float>,
                exposure: Float,
                environmentRotation: Float,
                maxBounces: Int,
                camera: OrbitCamera,
                polar: OpticsPolarRenderState?,
                sphere: SIMD4<Float>? = nil,
                mesh: OpticsMeshState? = nil,
                cell: OpticsCellState? = nil,
                axes: OpticsAxisState? = nil,
                frost: Float = 0,
                whiteModel: Bool = false,
                whiteAlbedo: SIMD3<Float> = SIMD3<Float>(0.93, 0.95, 0.97),
                occludeAxes: Bool = false) {
        self.planes = planes
        self.ior = ior
        self.dispersion = dispersion
        self.absorptionAxisX = absorptionAxisX
        self.absorptionAxisY = absorptionAxisY
        self.absorptionAxisZ = absorptionAxisZ
        self.spectrumAxes = spectrumAxes
        self.illuminant = illuminant
        self.environment = environment
        self.background = background
        self.exposure = exposure
        self.environmentRotation = environmentRotation
        self.maxBounces = maxBounces
        self.camera = camera
        self.polar = polar
        self.sphere = sphere
        self.mesh = mesh
        self.cell = cell
        self.axes = axes
        self.frost = frost
        self.whiteModel = whiteModel
        self.whiteAlbedo = whiteAlbedo
        self.occludeAxes = occludeAxes
    }
}

// MARK: - Renderer

public final class OpticsRenderer: NSObject, MTKViewDelegate {
    public static let maxPlanes = 224
    /// spectralTable：5 照明体 × 8 + 3 主轴 × 8 = 64 个 SIMD4（与 optics_lab.metal 布局一致）
    public static let spectralFloats = 64 * 4
    /// 旋光色散 buffer(4)：6 个 float2 对打包为 3 个 SIMD4（分配 4 行余量）
    public static let rotationFloats = 4 * 4

    public let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelineState: MTLRenderPipelineState
    private let conoPipelineState: MTLRenderPipelineState
    private let conoGemAccumPipelineState: MTLRenderPipelineState
    private let conoGemDisplayPipelineState: MTLRenderPipelineState
    private let planeBuffer: MTLBuffer
    /// 每帧 uniforms 三缓冲环：真机 GPU 路径追踪单帧 >16ms，若共用单缓冲，
    /// CPU 覆写时 GPU 仍在读（帧号/载物台角/光轴角错位），会导致累积闪烁、
    /// 乒乓顺序错乱与参数调节延迟。配合 inFlightSemaphore 保证最多 3 帧在途。
    private let uniformBuffers: [MTLBuffer]
    private var inFlightIndex = 0
    private let inFlightSemaphore = DispatchSemaphore(value: OpticsRenderer.maxInFlight)
    private let spectralBuffer: MTLBuffer
    private let rotationBuffer: MTLBuffer
    /// 网格路径缓冲（凹面支持）：buffer(7) 开关参数、buffer(8) BVH 节点表、
    /// buffer(9) 三角形表。容量按 OpticsMeshBuilder 上限一次性分配；
    /// 网格内容随签名判重换入，凸体模式下 meshParams 恒为 0。
    private let meshParamsBuffer: MTLBuffer
    private let meshNodeBuffer: MTLBuffer
    private let meshTriBuffer: MTLBuffer
    /// 白模真晶棱表（buffer(15)）：每棱 2 行 (A.xyz,0)(B.xyz,0)，容量同上一次性分配。
    /// 棱集是三角形表的确定性函数，随网格同一签名判重换入（无需独立指纹）。
    private let meshEdgeBuffer: MTLBuffer
    private var lastMeshStamp: BufferStamp?
    /// 晶胞模式：optics_cell_fragment 管线与 buffer(10)(11)(12)。
    /// 与 mesh 缓冲同样的签名判重上传策略。
    private let cellPipelineState: MTLRenderPipelineState
    private let cellParamsBuffer: MTLBuffer
    private let cellAtomBuffer: MTLBuffer
    private let cellCapsuleBuffer: MTLBuffer
    private var lastCellStamp: BufferStamp?
    /// 晶轴指示层：alpha 混合管线的叠加 pass（buffer(13) 参数 / (14) 轴表）
    private let axisPipelineState: MTLRenderPipelineState
    private let axisParamsBuffer: MTLBuffer
    private let axisCapsuleBuffer: MTLBuffer
    private var lastAxisStamp: BufferStamp?
    /// 真实锥光渐进累积双纹理（乒乓；rgba16Float 线性 HDR）
    private var accumTextures: [MTLTexture] = []
    private var accumFrame = 0
    /// 最近一次写入的累积纹理索引（收敛后显示 pass 固定采样它）
    private var accumWriteIdx = 1
    private var lastSignature: String?

    /// 真实锥光每轮累积帧数。理想平行板方向像为确定性结果（无蒙特卡洛随机项），
    /// 单帧即最终图像；保留双 pass 仅为兼容模拟器的纹理读写限制。
    public static let conoGemAccumFrames = 1
    /// 累积纹理内部分辨率上限（最长边）。锥光方向像为平滑低频图案，
    /// 低分辨率累积 + 显示 pass 线性放大几乎无损，却把真机单帧成本降低数倍。
    public static let conoGemInternalCap: CGFloat = 480
    /// CPU 最多领先 GPU 的命令缓冲数（uniform ring 同大小）
    public static let maxInFlight = 3
    /// 「几何已关闭」的哨兵指纹（区别于 nil = 从未写过）
    private static let stampDisabled = BufferStamp(count0: -1, count1: -1, radius: -1)

    public private(set) var state: OpticsRenderState

    public static func make() -> OpticsRenderer? {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else { return nil }
        do {
            return try OpticsRenderer(device: device, queue: queue)
        } catch {
            assertionFailure("光学管线初始化失败: \(error)")
            return nil
        }
    }

    public init(device: MTLDevice, queue: MTLCommandQueue) throws {
        self.device = device
        self.commandQueue = queue
        let initialCamera = OrbitCamera()
        state = OpticsRenderState(
            planes: [], ior: 2.417, dispersion: 0.044,
            absorptionAxisX: SIMD3<Float>(0, 0, 0),
            absorptionAxisY: SIMD3<Float>(0, 0, 0),
            absorptionAxisZ: SIMD3<Float>(0, 0, 0),
            spectrumAxes: nil, illuminant: 0,
            environment: 0, background: SIMD3<Float>(0xEF / 255, 0xF1 / 255, 0xF1 / 255),
            exposure: 0, environmentRotation: 18, maxBounces: 6,
            camera: initialCamera,
            polar: nil
        )

        // Swift Package 分发：.metal 是包资源，经 makeLibrary(source:) 编译。
        // （App 内嵌时代的 device.makeDefaultLibrary() 只能读主 bundle，不可用。）
        guard let lib = try? OpticsShaderLibrary.makeLibrary(for: device),
              let vertex = lib.makeFunction(name: "optics_vertex"),
              let fragment = lib.makeFunction(name: "optics_fragment"),
              let conoFragment = lib.makeFunction(name: "optics_cono_fragment"),
              let conoGemAccumFragment = lib.makeFunction(name: "optics_cono_gem_accum_fragment"),
              let conoGemDisplayFragment = lib.makeFunction(name: "optics_cono_gem_display_fragment"),
              let cellFragment = lib.makeFunction(name: "optics_cell_fragment"),
              let axisFragment = lib.makeFunction(name: "optics_axis_fragment") else {
            throw NSError(domain: "OpticsRenderer", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "未找到 optics_lab 着色器函数"])
        }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertex
        descriptor.fragmentFunction = fragment
        descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        pipelineState = try device.makeRenderPipelineState(descriptor: descriptor)
        let cellDescriptor = MTLRenderPipelineDescriptor()
        cellDescriptor.vertexFunction = vertex
        cellDescriptor.fragmentFunction = cellFragment
        cellDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        cellPipelineState = try device.makeRenderPipelineState(descriptor: cellDescriptor)
        // 晶轴指示层：alpha 混合（叠加 pass 的 load/draw 语义，见 optics_axis_fragment）
        let axisDescriptor = MTLRenderPipelineDescriptor()
        axisDescriptor.vertexFunction = vertex
        axisDescriptor.fragmentFunction = axisFragment
        axisDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        axisDescriptor.colorAttachments[0].isBlendingEnabled = true
        axisDescriptor.colorAttachments[0].rgbBlendOperation = .add
        axisDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        axisDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        axisDescriptor.colorAttachments[0].alphaBlendOperation = .add
        axisDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
        axisDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        axisPipelineState = try device.makeRenderPipelineState(descriptor: axisDescriptor)
        let conoDescriptor = MTLRenderPipelineDescriptor()
        conoDescriptor.vertexFunction = vertex
        conoDescriptor.fragmentFunction = conoFragment
        conoDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        conoPipelineState = try device.makeRenderPipelineState(descriptor: conoDescriptor)
        // 真实锥光：两个 render pass——pass1 累积纹理（rgba16Float，乒乓读另一张），
        // pass2 采样累积结果显示到 drawable（模拟器不支持单 pass MRT+读回）
        let gemAccumDescriptor = MTLRenderPipelineDescriptor()
        gemAccumDescriptor.vertexFunction = vertex
        gemAccumDescriptor.fragmentFunction = conoGemAccumFragment
        gemAccumDescriptor.colorAttachments[0].pixelFormat = .rgba16Float
        conoGemAccumPipelineState = try device.makeRenderPipelineState(descriptor: gemAccumDescriptor)
        let gemDisplayDescriptor = MTLRenderPipelineDescriptor()
        gemDisplayDescriptor.vertexFunction = vertex
        gemDisplayDescriptor.fragmentFunction = conoGemDisplayFragment
        gemDisplayDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        conoGemDisplayPipelineState = try device.makeRenderPipelineState(descriptor: gemDisplayDescriptor)

        let planeCapacity = OpticsRenderer.maxPlanes * MemoryLayout<SIMD4<Float>>.stride
        let spectralCapacity = OpticsRenderer.spectralFloats * MemoryLayout<Float>.stride
        let rotationCapacity = OpticsRenderer.rotationFloats * MemoryLayout<Float>.stride
        let meshParamsCapacity = MemoryLayout<SIMD4<Float>>.stride
        let meshNodeCapacity = 3 * OpticsMeshBuilder.maxNodes * MemoryLayout<SIMD4<Float>>.stride
        let meshTriCapacity = 4 * OpticsMeshBuilder.maxTriangles * MemoryLayout<SIMD4<Float>>.stride
        let meshEdgeCapacity = 2 * OpticsMeshBuilder.maxCreaseEdges * MemoryLayout<SIMD4<Float>>.stride
        let cellParamsCapacity = MemoryLayout<SIMD4<Float>>.stride
        let cellAtomCapacity = 2 * OpticsCellState.maxAtoms * MemoryLayout<SIMD4<Float>>.stride
        let cellCapsuleCapacity = 3 * OpticsCellState.maxCapsules * MemoryLayout<SIMD4<Float>>.stride
        let axisParamsCapacity = MemoryLayout<SIMD4<Float>>.stride
        let axisCapsuleCapacity = 3 * OpticsAxisState.maxCapsules * MemoryLayout<SIMD4<Float>>.stride
        guard let planeBuffer = device.makeBuffer(length: planeCapacity,
                                                  options: .storageModeShared),
              let spectralBuffer = device.makeBuffer(length: spectralCapacity,
                                                     options: .storageModeShared),
              let rotationBuffer = device.makeBuffer(length: rotationCapacity,
                                                     options: .storageModeShared),
              let meshParamsBuffer = device.makeBuffer(length: meshParamsCapacity,
                                                       options: .storageModeShared),
              let meshNodeBuffer = device.makeBuffer(length: meshNodeCapacity,
                                                     options: .storageModeShared),
              let meshTriBuffer = device.makeBuffer(length: meshTriCapacity,
                                                    options: .storageModeShared),
              let meshEdgeBuffer = device.makeBuffer(length: meshEdgeCapacity,
                                                     options: .storageModeShared),
              let cellParamsBuffer = device.makeBuffer(length: cellParamsCapacity,
                                                       options: .storageModeShared),
              let cellAtomBuffer = device.makeBuffer(length: cellAtomCapacity,
                                                     options: .storageModeShared),
              let cellCapsuleBuffer = device.makeBuffer(length: cellCapsuleCapacity,
                                                        options: .storageModeShared),
              let axisParamsBuffer = device.makeBuffer(length: axisParamsCapacity,
                                                       options: .storageModeShared),
              let axisCapsuleBuffer = device.makeBuffer(length: axisCapsuleCapacity,
                                                        options: .storageModeShared) else {
            throw NSError(domain: "OpticsRenderer", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "GPU 缓冲分配失败"])
        }
        let uniformStride = MemoryLayout<OpticsUniforms>.stride
        let uniformBuffers: [MTLBuffer] = (0..<OpticsRenderer.maxInFlight).compactMap { _ in
            device.makeBuffer(length: uniformStride, options: .storageModeShared)
        }
        guard uniformBuffers.count == OpticsRenderer.maxInFlight else {
            throw NSError(domain: "OpticsRenderer", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "uniform ring 缓冲分配失败"])
        }
        self.planeBuffer = planeBuffer
        self.uniformBuffers = uniformBuffers
        self.spectralBuffer = spectralBuffer
        self.rotationBuffer = rotationBuffer
        self.meshParamsBuffer = meshParamsBuffer
        self.meshNodeBuffer = meshNodeBuffer
        self.meshTriBuffer = meshTriBuffer
        self.meshEdgeBuffer = meshEdgeBuffer
        self.cellParamsBuffer = cellParamsBuffer
        self.cellAtomBuffer = cellAtomBuffer
        self.cellCapsuleBuffer = cellCapsuleBuffer
        self.axisParamsBuffer = axisParamsBuffer
        self.axisCapsuleBuffer = axisCapsuleBuffer
        // 照明体 SPD 为常量：初始化时写入一次（5 表 × 31 bin，各占 32 Float 行）
        OpticsRenderer.fillSpectralIlluminants(dst: spectralBuffer.contents())
        super.init()
        // 布局自检：与 optics_lab.metal 的 Uniforms 逐字段对应，改任何一边都要同时改这里。
        // 构成（字节）：基础 11×float4 = 176 → int4 flags 16 = 192 → float2 resolution 8 = 200
        // → 补齐到 208 → 偏光 5×float4 = 80 = 288 → 解析球面 1×float4 = 304
        // → 三偏光片 polarLab 1×float4 = 320 → 宝石姿态 gemTumble 1×float4 = 336。
        // 这个断言是唯一能在启动期拦住「Swift 与 MSL 字段错位」的东西 ——
        // 错位不会编译失败，只会让画面静默用错参数。
        // ⚠️ 每次扩 Uniforms 都必须同步这个常量，而且**同时改三处**：
        //    本文件、optics_lab.metal 的 Uniforms、宿主测试的同名断言。
        assert(uniformStride == 336,
               "OpticsUniforms 布局与 MSL 不一致：\(uniformStride)（应为 336）")
    }

    public func attach(to view: MTKView) {
        view.device = device
        view.colorPixelFormat = .bgra8Unorm
        view.framebufferOnly = true
        view.isPaused = true
        view.enableSetNeedsDisplay = false
        view.delegate = self
        view.clearColor = MTLClearColor(red: 0.94, green: 0.95, blue: 0.95, alpha: 1)
        view.sampleCount = 1
    }

    /// 与 Web 端一致的有效渲染倍率：min(screenScale, 1.5, 1100 / 最长边)
    public static func effectiveScale(for bounds: CGSize, screenScale: CGFloat) -> CGFloat {
        let longest = max(bounds.width, bounds.height)
        guard longest > 0 else { return 1 }
        return min(screenScale, 1.5, 1100 / longest)
    }

    public func update(_ newState: OpticsRenderState) {
        state = newState
    }

    /// 卸载前排干在途帧。completed handler 是 GPU 完成后的异步回调——若渲染器
    /// 带着在途帧 deinit，信号量 dispose 就与回调赛跑（实测：晶形切红外销毁
    /// 视图时 dispatch_semaphore_dispose SIGTRAP）。调用前必须已停帧
    /// （delegate 置 nil），把 maxInFlight 个许可全部等回来后再释放渲染器。
    public func drainInFlight() {
        var got = 0
        for _ in 0..<OpticsRenderer.maxInFlight {
            // 正常情况下每帧完成即归还许可，这里只是把它等走；超时是防御
            // （真机路径追踪单帧数百 ms，3 帧在途约 1s 内必然排干）
            if inFlightSemaphore.wait(timeout: .now() + 5) != .success { break }
            got += 1
        }
        // 等走的许可必须如数归还：deinit 之后不会再有任何 signal（delegate
        // 已停、帧已完成），dispose 时 value 必须回到初值——实测这个版本的
        // libdispatch 对失衡的 semaphore dispose 一律 SIGTRAP
        for _ in 0..<got { inFlightSemaphore.signal() }
    }

    public func requestRender(in view: MTKView) {
        view.draw()
    }

    /// 按当前 bounds 校正 drawableSize（bounds 在 SwiftUI 首帧后才稳定）。
    public func syncDrawableSize(for view: MTKView) {
        // contentScaleFactor 是 UIView 的属性；macOS 走窗口 backingScaleFactor
        #if canImport(UIKit)
        let screenScale = view.contentScaleFactor
        #else
        let screenScale = view.window?.backingScaleFactor ?? 2
        #endif
        let scale = OpticsRenderer.effectiveScale(for: view.bounds.size,
                                                  screenScale: screenScale)
        let target = CGSize(width: max(2, view.bounds.width * scale),
                            height: max(2, view.bounds.height * scale))
        guard view.drawableSize != target else { return }
        view.drawableSize = target
        view.draw()
    }

    // MARK: - Uniform 组装（渲染器与 CPU 参考 diff harness 共用的唯一组装点）
    //
    // harness 的前提是 GPU 与 CPU 吃到逐字节相同的参数 —— 组装逻辑若有两份，
    // 测的就是「两份组装的差异」而不是「着色器数学 vs 参考数学」。所以组装只此一份，
    // 渲染器 draw(in:) 与测试侧都从这里取。

    /// 几何解析：解析球面优先于平面表。w > 0 时平面数必须为 0，否则球面分支虽然生效，
    /// flags.z 却仍在声明一批被忽略的平面（且浪费一次上传）。
    /// 网格/晶胞路径同样接管几何：任一非 nil 时平面表清空（flags.z = 0），
    /// 着色器按各自开关切分支，不会读 planeTable（偏光 3D + 网格时
    /// traceGemPolar 走 BVH 分支，同样不读；sphere 原样返回仅为取景参考）。
    ///
    /// ⚠️ 每帧只应调用一次：它会被 `makeUniforms` 与平面上传各用一次，
    /// 重复调用等于每帧多拷一份最多 224 个 SIMD4 的数组（见 draw(in:)）。
    public static func resolvedGeometry(_ state: OpticsRenderState) -> (planes: [SIMD4<Float>], sphere: SIMD4<Float>) {
        let sphere = state.sphere ?? SIMD4<Float>(0, 0, 0, 0)
        guard sphere.w <= 0, state.mesh == nil, state.cell == nil else {
            return ([], sphere)
        }
        // 未超上限时直接复用原数组（COW，只多一次 retain），不逐帧拷贝
        let planes = state.planes.count > maxPlanes
            ? Array(state.planes.prefix(maxPlanes)) : state.planes
        return (planes, sphere)
    }

    /// 组装 Uniforms（字段顺序/取值与 optics_lab.metal 的 Uniforms 一一对应；
    /// 末尾的 stride 断言是启动期对二进制布局的唯一防线）。
    ///
    /// `geometry`：draw(in:) 已经解析过一次几何时传进来复用，避免每帧第二次
    /// 解析（历史实现里这里和平面上传各解析一次，白扔一份数组拷贝）。
    public static func makeUniforms(state: OpticsRenderState, width: Float, height: Float,
                                    gemFrame: Int = 0,
                                    geometry: (planes: [SIMD4<Float>], sphere: SIMD4<Float>)? = nil)
    -> OpticsUniforms {
        let f = state.camera.frame
        let geo = geometry ?? resolvedGeometry(state)
        let polar = state.polar
        let polarParams = SIMD4<Float>(
            polar.map { $0.mode == 1 ? 1 : 0 } ?? 0,
            polar?.analyzerAngleRad ?? 0,
            polar?.stageRad ?? 0,
            polar?.plateThicknessMm ?? 0
        )
        let crystalA = SIMD4<Float>(
            polar.map { Float($0.crystalKind) } ?? 0,
            polar?.nO ?? 0,
            polar?.nE ?? 0,
            0
        )
        let crystalB = SIMD4<Float>(
            polar?.nA ?? 0,
            polar?.nB ?? 0,
            polar?.nG ?? 0,
            polar?.halfVrad ?? 0
        )
        let polarFlags = SIMD4<Float>(
            (polar?.sodium ?? false) ? 1 : 0,
            polar?.fovHalfAngleRad ?? 0,
            polar.map { Float(min($0.rotationSamples.count, 6)) } ?? 0,
            Float(gemFrame)
        )
        let crystalOrient = SIMD4<Float>(
            polar?.axisTiltRad ?? 0,
            polar?.axisAzimuthRad ?? 0,
            polar?.axisRollRad ?? 0,
            0
        )
        // 相机尺度：网格/晶胞路径按归一化半径自适应（OpticsFraming，与上游
        // opticsMeshFraming 同式），凸体/球面沿用琢型定标 0.34
        let baseScale: Float
        if let mesh = state.mesh {
            baseScale = OpticsFraming.scale(radius: mesh.radius)
        } else if let cell = state.cell {
            baseScale = OpticsFraming.scale(radius: cell.radius)
        } else {
            baseScale = OpticsCameraScale.defaultScale
        }
        return OpticsUniforms(
            cameraPositionScale: SIMD4<Float>(f.position, baseScale / state.camera.zoom),
            cameraForwardExposure: SIMD4<Float>(f.forward, state.exposure),
            cameraRightEnvRot: SIMD4<Float>(f.right, state.environmentRotation),
            cameraUpFocal: SIMD4<Float>(f.up, 0),
            // x 槽位复用为磨砂强度（体色升级光谱后 xyz 空置；CPU 参考 harness
            // 不读这个槽，frost=0 时与旧值 1 一样无副作用）
            bodyColorIor: SIMD4<Float>(state.frost, 0, 0, state.ior),
            observerDispersion: SIMD4<Float>(simd_normalize(f.position), state.dispersion),
            backgroundAbsorption: SIMD4<Float>(state.background, 0),
            // absorptionAxisX 在 shader 里已无读者（RGB 回退路径停用）——
            // 复用为白模通道：xyz = 反照率，w = 白模开关。见 optics_lab.metal 同注释
            absorptionAxisX: SIMD4<Float>(state.whiteModel ? state.whiteAlbedo : .zero,
                                          state.whiteModel ? 1 : 0),
            absorptionAxisY: SIMD4<Float>(state.absorptionAxisY, 0),
            absorptionAxisZ: SIMD4<Float>(state.absorptionAxisZ, 0),
            // w 复用为光轴遮挡开关：true 时紫色虚线光轴进入晶体内部即被网格挡住。
            // 与白模通道(absorptionAxisX.w)分离——透明宝石也可单独让光轴藏进晶体。
            panParams: SIMD4<Float>(state.camera.panX, state.camera.panY, 0,
                                    state.occludeAxes ? 1 : 0),
            flags: SIMD4<Int32>(Int32(state.environment), Int32(state.maxBounces),
                                Int32(geo.planes.count), Int32(clamping: state.illuminant)),
            resolution: SIMD2<Float>(width, height),
            polarParams: polarParams,
            crystalA: crystalA,
            crystalB: crystalB,
            polarFlags: polarFlags,
            crystalOrient: crystalOrient,
            sphereParams: geo.sphere,
            polarLab: SIMD4<Float>(
                polar?.polarizerAngleRad ?? 0,
                polar?.midPolarizerAngleRad ?? 0,
                (polar?.midPolarizerOn ?? false) ? 1 : 0,
                // w = 宝石自转角。漏写（恒 0）时水平拖后宝石几何会转、光轴紫虚线
                // 与锥光图不动 —— 不报错，只是「标记脱宝石」。
                polar?.gemSpinRad ?? 0
            ),
            // 宝石姿态：非偏光模式必须写**单位四元数**而不是 0 —— shader 无条件
            // 拿它做世界⇄宝石系的旋转，全 0 会让宝石塌成一个点（画面全黑）。
            gemTumble: polar?.gemTumble ?? SIMD4<Float>(0, 0, 0, 1)
        )
    }

    /// 照明体 SPD → spectralTable[0..40) 行（5 表 × 8 SIMD4 行，各 31 bin 末位补 0）。
    public static func fillSpectralIlluminants(dst: UnsafeMutableRawPointer) {
        dst.withMemoryRebound(to: Float.self, capacity: spectralFloats) { dst in
            for (table, spd) in OpticsIlluminantSpectrum.tables.enumerated() {
                for b in 0..<31 { dst[table * 32 + b] = spd[b] }
            }
        }
    }

    /// 光谱吸收三主轴 → spectralTable[40..64) 行（每轴 8 个 SIMD4；nil 轴写 0 = 近无色）。
    public static func fillSpectralAxes(dst: UnsafeMutableRawPointer, state: OpticsRenderState) {
        dst.withMemoryRebound(to: Float.self, capacity: spectralFloats) { dst in
            let axisBase = 5 * 32   // 160 个 Float
            let packed = state.spectrumAxes
            for axis in 0..<3 {
                let rows = (axis < (packed?.count ?? 0)) ? packed![axis] : nil
                for q in 0..<8 {
                    let v = (rows.map { $0[q] }) ?? SIMD4<Float>(repeating: 0)
                    let o = axisBase + axis * 32 + q * 4
                    dst[o] = v.x; dst[o + 1] = v.y; dst[o + 2] = v.z; dst[o + 3] = v.w
                }
            }
        }
    }

    /// 网格几何换入（buffer 7/8/9）。判重：meshParams（含网格开关）只在
    /// 换网格时写入——若每帧覆写这 16 字节，CPU 会与在途 GPU 命令缓冲竞写
    /// 同一缓冲，模式切换瞬间可能读到撕裂的 (mode, nodeCount) 组合。
    /// 指纹除计数外带首末行列值：不同晶形的节点/三角形数完全可能相同。
    private func updateMeshBuffers() {
        let stride = MemoryLayout<SIMD4<Float>>.stride
        guard let mesh = state.mesh,
              mesh.nodeCount > 0, mesh.triangleCount > 0,
              mesh.nodeCount <= OpticsMeshBuilder.maxNodes,
              mesh.triangleCount <= OpticsMeshBuilder.maxTriangles else {
            if lastMeshStamp != Self.stampDisabled {
                lastMeshStamp = Self.stampDisabled
                var off = SIMD4<Float>(0, 0, 0, 0)
                meshParamsBuffer.contents().copyMemory(from: &off, byteCount: stride)
            }
            return
        }
        let stamp = BufferStamp(
            count0: mesh.nodeCount, count1: mesh.triangleCount, radius: mesh.radius,
            v0: mesh.nodes.first ?? .zero, v1: mesh.nodes.last ?? .zero,
            v2: mesh.triangles.first ?? .zero, v3: mesh.triangles.last ?? .zero)
        guard stamp != lastMeshStamp else { return }
        lastMeshStamp = stamp
        mesh.nodes.withUnsafeBufferPointer { src in
            meshNodeBuffer.contents().copyMemory(
                from: src.baseAddress!, byteCount: mesh.nodes.count * stride)
        }
        mesh.triangles.withUnsafeBufferPointer { src in
            meshTriBuffer.contents().copyMemory(
                from: src.baseAddress!, byteCount: mesh.triangles.count * stride)
        }
        // 真晶棱表（白模黑色实线的几何凭据；容量截断防御——超限宁可缺棱）
        let edgeRows = mesh.creaseEdges.prefix(OpticsMeshBuilder.maxCreaseEdges * 2)
        edgeRows.withUnsafeBufferPointer { src in
            if let base = src.baseAddress {
                meshEdgeBuffer.contents().copyMemory(
                    from: base, byteCount: src.count * stride)
            }
        }
        // meshParams = (网格开关, BVH 节点数, 归一化半径, 晶棱数)。
        // z 半径供 shader 画偏光载物台圆盘（尺寸/高度随宝石自适应）；
        // w 晶棱数供白模画黑色实线棱（其余路径不读）。
        var params = SIMD4<Float>(1, Float(mesh.nodeCount), mesh.radius,
                                  Float(mesh.creaseEdgeCount))
        meshParamsBuffer.contents().copyMemory(from: &params, byteCount: stride)
    }

    /// 晶胞几何换入（buffer 10/11/12）。判重策略与网格相同：
    /// cellParams（含模式开关）只在换几何时写入，避免与在途 GPU 竞写。
    /// 超容量直接整体关闭——宁可空屏也不画半颗晶胞误导读者。
    private func updateCellBuffers() {
        let stride = MemoryLayout<SIMD4<Float>>.stride
        guard let cell = state.cell,
              cell.atomCount > 0,
              cell.atomCount <= OpticsCellState.maxAtoms,
              cell.capsuleCount <= OpticsCellState.maxCapsules else {
            if lastCellStamp != Self.stampDisabled {
                lastCellStamp = Self.stampDisabled
                var off = SIMD4<Float>(0, 0, 0, 0)
                cellParamsBuffer.contents().copyMemory(from: &off, byteCount: stride)
            }
            return
        }
        let stamp = BufferStamp(
            count0: cell.atomCount, count1: cell.capsuleCount, radius: cell.radius,
            v0: cell.atoms.first ?? .zero, v1: cell.capsules.last ?? .zero)
        guard stamp != lastCellStamp else { return }
        lastCellStamp = stamp
        cell.atoms.withUnsafeBufferPointer { src in
            cellAtomBuffer.contents().copyMemory(
                from: src.baseAddress!, byteCount: cell.atoms.count * stride)
        }
        cell.capsules.withUnsafeBufferPointer { src in
            cellCapsuleBuffer.contents().copyMemory(
                from: src.baseAddress!, byteCount: cell.capsules.count * stride)
        }
        var params = SIMD4<Float>(Float(cell.atomCount), Float(cell.capsuleCount), 0, 0)
        cellParamsBuffer.contents().copyMemory(from: &params, byteCount: stride)
    }

    /// 晶轴指示层换入（buffer 13/14）。判重与网格/晶胞相同：
    /// axisParams 只在换几何时写入，避免与在途 GPU 竞写。
    private func updateAxisBuffers() {
        let stride = MemoryLayout<SIMD4<Float>>.stride
        guard let axes = state.axes,
              axes.capsuleCount > 0,
              axes.capsuleCount <= OpticsAxisState.maxCapsules else {
            if lastAxisStamp != Self.stampDisabled {
                lastAxisStamp = Self.stampDisabled
                var off = SIMD4<Float>(0, 0, 0, 0)
                axisParamsBuffer.contents().copyMemory(from: &off, byteCount: stride)
            }
            return
        }
        let stamp = BufferStamp(
            count0: axes.capsuleCount, count1: 0, radius: 0,
            v0: axes.capsules.first ?? .zero, v1: axes.capsules.last ?? .zero)
        guard stamp != lastAxisStamp else { return }
        lastAxisStamp = stamp
        axes.capsules.withUnsafeBufferPointer { src in
            axisCapsuleBuffer.contents().copyMemory(
                from: src.baseAddress!, byteCount: axes.capsules.count * stride)
        }
        // (胶囊数, 全局不透明度, 0, 0)
        var params = SIMD4<Float>(Float(axes.capsuleCount), 1, 0, 0)
        axisParamsBuffer.contents().copyMemory(from: &params, byteCount: stride)
    }

    // MARK: MTKViewDelegate

    public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // 尺寸变化已在 draw(in:) 中处理；此处不主动绘制，避免嵌套调用。
    }

    public func draw(in view: MTKView) {
        // 非阻塞节流：真机路径追踪单帧远超 16ms，3 帧在途时直接跳过本次回调
        // （MTKView 下一 tick 再来），把累积速率限制到 GPU 实际帧率，
        // 且绝不阻塞主线程——拖滑块/转载物台始终跟手。
        guard inFlightSemaphore.wait(timeout: .now()) == .success else { return }
        // 真实锥光自行组织两个 render pass（不使用 MTKView 的默认 pass），
        // 故此处只取 drawable/commandBuffer，encoder 按模式延后创建。
        guard let drawable = view.currentDrawable,
              let buffer = commandQueue.makeCommandBuffer() else {
            inFlightSemaphore.signal()
            return
        }
        buffer.addCompletedHandler { [weak self] _ in self?.inFlightSemaphore.signal() }
        let uniformBuffer = uniformBuffers[inFlightIndex % OpticsRenderer.maxInFlight]
        inFlightIndex = (inFlightIndex + 1) % OpticsRenderer.maxInFlight

        // 几何解析（球面优先/平面截断）每帧只做一次：平面上传与 makeUniforms
        // 共用同一份，flags.z 与上传的平面数必然同源 —— 见 resolvedGeometry。
        let geometry = OpticsRenderer.resolvedGeometry(state)
        let activePlanes = geometry.planes
        let planeCount = activePlanes.count
        if planeCount > 0 {
            activePlanes.withUnsafeBufferPointer { src in
                planeBuffer.contents().copyMemory(
                    from: src.baseAddress!,
                    byteCount: planeCount * MemoryLayout<SIMD4<Float>>.stride)
            }
        }

        let isConoscopic = state.polar?.mode == 2
        let isConoscopicGem = state.polar?.mode == 3
        // 参数签名只在真实锥光下有意义（其余模式不累积），非锥光帧不必付
        // 这十几次字符串插值的代价。
        if isConoscopicGem {
            let signature = OpticsRenderer.stateSignature(state: state,
                                                          drawableSize: view.drawableSize)
            if signature != lastSignature {
                lastSignature = signature
                accumFrame = 0
            }
        }
        // 收敛后冻结：显示-only 重绘不再推进帧号，避免新随机样本扰动静态画面
        let gemFrame = isConoscopicGem
            ? min(accumFrame + 1, OpticsRenderer.conoGemAccumFrames) : 0
        // uniforms 组装只此一份（makeUniforms），diff harness 用同一函数喂 CPU 参考 ——
        // 若渲染器与 harness 各自组装，测到的就是「两份组装的差异」而非着色器数学。
        var uniforms = OpticsRenderer.makeUniforms(
            state: state,
            width: Float(view.drawableSize.width),
            height: Float(view.drawableSize.height),
            gemFrame: gemFrame,
            geometry: geometry
        )
        uniformBuffer.contents().copyMemory(from: &uniforms,
                                            byteCount: MemoryLayout<OpticsUniforms>.stride)

        // 光谱吸收轴 → spectralTable[40..64) 行（照明体 5×8 行之后；每轴 8 个 SIMD4）
        OpticsRenderer.fillSpectralAxes(dst: spectralBuffer.contents(), state: state)

        // 旋光色散采样 → buffer(4)：每行 2 个 float2 对（wl, rate, wl, rate）
        rotationBuffer.contents().withMemoryRebound(to: Float.self,
                                                    capacity: OpticsRenderer.rotationFloats) { dst in
            let samples = state.polar?.rotationSamples.prefix(6) ?? []
            for i in 0..<6 {
                if i < samples.count {
                    dst[i * 2] = samples[i].x
                    dst[i * 2 + 1] = samples[i].y
                } else {
                    dst[i * 2] = 0
                    dst[i * 2 + 1] = 0
                }
            }
        }

        // 真实锥光：累积 pass（低分辨率乒乓纹理）+ 显示 pass（线性放大到 drawable）。
        // 收敛后跳过累积 pass，只重放显示 pass——画面彻底冻结，拖滑块等 UI 重绘
        // 不再注入新随机样本；两个 pass 必须分属独立 encoder（同 buffer 顺序执行）。
        if isConoscopicGem {
            ensureAccumTextures(size: view.drawableSize)
            let converged = accumFrame >= OpticsRenderer.conoGemAccumFrames

            if !converged {
                let readIdx = accumFrame % 2
                let writeIdx = 1 - readIdx
                accumWriteIdx = writeIdx

                // Pass 1：方向像路径追踪 + running average → accum[write]
                let accumPass = MTLRenderPassDescriptor()
                accumPass.colorAttachments[0].texture = accumTextures[writeIdx]
                accumPass.colorAttachments[0].loadAction = .dontCare
                accumPass.colorAttachments[0].storeAction = .store
                guard let accumEncoder = buffer.makeRenderCommandEncoder(descriptor: accumPass) else {
                    buffer.commit()
                    return
                }
                accumEncoder.setRenderPipelineState(conoGemAccumPipelineState)
                accumEncoder.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
                accumEncoder.setFragmentBuffer(uniformBuffer, offset: 0, index: 1)
                accumEncoder.setFragmentBuffer(planeBuffer, offset: 0, index: 2)
                accumEncoder.setFragmentBuffer(spectralBuffer, offset: 0, index: 3)
                accumEncoder.setFragmentBuffer(rotationBuffer, offset: 0, index: 4)
                accumEncoder.setFragmentTexture(accumTextures[readIdx], index: 1)
                accumEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
                accumEncoder.endEncoding()
                accumFrame += 1
            }

            // Pass 2：累积结果色调映射 → drawable（独立 render pass，线性放大）
            let displayPass = MTLRenderPassDescriptor()
            displayPass.colorAttachments[0].texture = drawable.texture
            displayPass.colorAttachments[0].loadAction = .dontCare
            displayPass.colorAttachments[0].storeAction = .store
            guard let displayEncoder = buffer.makeRenderCommandEncoder(descriptor: displayPass) else {
                buffer.commit()
                return
            }
            displayEncoder.setRenderPipelineState(conoGemDisplayPipelineState)
            displayEncoder.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
            displayEncoder.setFragmentBuffer(uniformBuffer, offset: 0, index: 1)
            displayEncoder.setFragmentTexture(accumTextures[accumWriteIdx], index: 1)
            displayEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            displayEncoder.endEncoding()
            buffer.present(drawable)
            buffer.commit()
            // 收敛期间连续刷新；收敛即暂停（后续 requestRender 仅触发显示 pass）
            view.isPaused = accumFrame >= OpticsRenderer.conoGemAccumFrames
            return
        }

        // 其余模式：MTKView 默认 pass（drawable 全屏三角形）
        // 晶胞模式：optics_cell_fragment 解析求交（原子/键/晶胞框），
        // 与路径追踪分支互斥（state.polar 与 state.cell 不会同时非 nil）
        let isCell = state.cell != nil && state.polar == nil
        guard let descriptor = view.currentRenderPassDescriptor,
              let encoder = buffer.makeRenderCommandEncoder(descriptor: descriptor) else {
            buffer.commit()
            return
        }
        encoder.setRenderPipelineState(isCell ? cellPipelineState
                                       : isConoscopic ? conoPipelineState : pipelineState)
        encoder.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
        encoder.setFragmentBuffer(uniformBuffer, offset: 0, index: 1)
        encoder.setFragmentBuffer(planeBuffer, offset: 0, index: 2)
        encoder.setFragmentBuffer(spectralBuffer, offset: 0, index: 3)
        encoder.setFragmentBuffer(rotationBuffer, offset: 0, index: 4)
        // 网格路径缓冲（凹面支持）：凸体模式下 meshParams 恒 0，着色器走原半空间分支
        updateMeshBuffers()
        encoder.setFragmentBuffer(meshParamsBuffer, offset: 0, index: 7)
        encoder.setFragmentBuffer(meshNodeBuffer, offset: 0, index: 8)
        encoder.setFragmentBuffer(meshTriBuffer, offset: 0, index: 9)
        encoder.setFragmentBuffer(meshEdgeBuffer, offset: 0, index: 15)
        // 晶胞路径缓冲：非晶胞模式下 cellParams 恒 0
        updateCellBuffers()
        encoder.setFragmentBuffer(cellParamsBuffer, offset: 0, index: 10)
        encoder.setFragmentBuffer(cellAtomBuffer, offset: 0, index: 11)
        encoder.setFragmentBuffer(cellCapsuleBuffer, offset: 0, index: 12)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        // 晶轴指示层：主结果之上 load 叠加（alpha 混合发光细线）。
        // 白模态着色器要对着网格做逐像素遮挡（轴被晶体挡住不发光），
        // 所以网格三缓冲也必须绑进这个 pass；透光态不读它们。
        if state.axes?.capsuleCount ?? 0 > 0 && !isCell {
            updateAxisBuffers()
            let overlayPass = MTLRenderPassDescriptor()
            overlayPass.colorAttachments[0].texture = drawable.texture
            overlayPass.colorAttachments[0].loadAction = .load
            overlayPass.colorAttachments[0].storeAction = .store
            if let overlayEncoder = buffer.makeRenderCommandEncoder(descriptor: overlayPass) {
                overlayEncoder.setRenderPipelineState(axisPipelineState)
                overlayEncoder.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
                overlayEncoder.setFragmentBuffer(uniformBuffer, offset: 0, index: 1)
                overlayEncoder.setFragmentBuffer(axisParamsBuffer, offset: 0, index: 13)
                overlayEncoder.setFragmentBuffer(axisCapsuleBuffer, offset: 0, index: 14)
                overlayEncoder.setFragmentBuffer(meshParamsBuffer, offset: 0, index: 7)
                overlayEncoder.setFragmentBuffer(meshNodeBuffer, offset: 0, index: 8)
                overlayEncoder.setFragmentBuffer(meshTriBuffer, offset: 0, index: 9)
                overlayEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
                overlayEncoder.endEncoding()
            }
        }
        buffer.present(drawable)
        buffer.commit()
    }

    /// 分配/重建真实锥光累积纹理对。
    /// 内部分辨率按 conoGemInternalCap 等比缩小（宽高比与 drawable 完全一致，
    /// shader 内圆形遮罩仅依赖宽高比，故 u.resolution 传 drawable 尺寸也无偏）。
    private func ensureAccumTextures(size drawable: CGSize) {
        let scale = min(1, OpticsRenderer.conoGemInternalCap
                        / max(drawable.width, drawable.height))
        let w = max(2, Int(drawable.width * scale))
        let h = max(2, Int(drawable.height * scale))
        if accumTextures.count == 2,
           accumTextures[0].width == w, accumTextures[0].height == h { return }
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: w, height: h, mipmapped: false)
        desc.usage = [.renderTarget, .shaderRead]
        desc.storageMode = .private
        accumTextures = [0, 1].compactMap { _ in device.makeTexture(descriptor: desc) }
        accumFrame = 0
        accumWriteIdx = 1
        lastSignature = nil
    }

    /// 真实锥光累积重置签名（材质/观察参数/几何/尺寸任一变化即重建）
    private static func stateSignature(state: OpticsRenderState, drawableSize: CGSize) -> String {
        let p = state.polar
        var parts = [
            "m\(p?.mode ?? 0)",
            "b\(p?.analyzerAngleRad ?? 0)",
            "pl\(p?.polarizerAngleRad ?? 0),\(p?.midPolarizerAngleRad ?? 0),\(p?.midPolarizerOn ?? false)",
            // 样品总方位角 = 台角 + 宝石自转（specimenSpin）。两者必须**都**进签名：
            // 只写 stageRad 时，手指水平拨宝石在正交 3D 里照常转（那条路径每帧重绘），
            // 但真实锥光的累积纹理判定为「没变」→ 沿用上一张冻住的图，
            // 画面上就是「拖了宝石，追光图纹丝不动」——不报错，也查不到。
            "s\(p?.stageRad ?? 0),\(p?.gemSpinRad ?? 0)",
            // 宝石姿态（四元数）：本轮起真实锥光吃它（indicatrixFrame 的内层 T）。
            // 它变了而签名没变 = 竖直拖拽在追光下毫无反应，且同样静默。
            "tumble\(p?.gemTumble ?? SIMD4<Float>(0, 0, 0, 1))",
            "t\(p?.plateThicknessMm ?? 0)",
            "k\(p?.crystalKind ?? 0)",
            "o\(p?.nO ?? 0),\(p?.nE ?? 0),\(p?.nA ?? 0),\(p?.nB ?? 0),\(p?.nG ?? 0)",
            "v\(p?.halfVrad ?? 0)",
            "na\(p?.sodium ?? false)",
            "tilt\(p?.axisTiltRad ?? 0),\(p?.axisAzimuthRad ?? 0),\(p?.axisRollRad ?? 0)",
            "ill\(state.illuminant)",
            "exp\(state.exposure)",
            // 真实锥光观察轴沿相机视线：拖拽视角必须触发重渲染
            "cam\(state.camera.yaw),\(state.camera.elevation)",
            "sz\(Int(drawableSize.width))x\(Int(drawableSize.height))",
            "pc\(state.planes.count)",
            // 球体：平面数恒为 0，签名必须带上球心/半径，否则换球不重置累积
            "sp\(state.sphere?.x ?? 0),\(state.sphere?.y ?? 0),\(state.sphere?.z ?? 0),\(state.sphere?.w ?? 0)"
        ]
        if let first = state.planes.first, let last = state.planes.last {
            parts.append("p0\(first.x),\(first.y),\(first.z),\(first.w)")
            parts.append("pN\(last.x),\(last.y),\(last.z),\(last.w)")
        }
        return parts.joined(separator: "|")
    }
}

/// 命名空间，避免与本地变量重名
public enum OpticsCameraScale {
    public static let defaultScale: Float = 0.34
}

/// 几何换入判重用的一次性指纹。
///
/// 原先每帧拼一个 String（插值 SIMD4 会走描述构造，是反射级花销），拖拽时
/// 等于每帧三次字符串构造 + 一次拼接。改成纯值比较后零分配、零格式化。
/// 语义与旧签名完全一致：两个计数 + 半径 + 最多四个采样行值。
private struct BufferStamp: Equatable {
    var count0: Int
    var count1: Int
    var radius: Float
    // 采样行值（元组不参与 Equatable 合成，只能摊平成四个字段）
    var v0: SIMD4<Float>
    var v1: SIMD4<Float>
    var v2: SIMD4<Float>
    var v3: SIMD4<Float>

    init(count0: Int, count1: Int, radius: Float,
         v0: SIMD4<Float> = .zero, v1: SIMD4<Float> = .zero,
         v2: SIMD4<Float> = .zero, v3: SIMD4<Float> = .zero) {
        self.count0 = count0
        self.count1 = count1
        self.radius = radius
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
    }
}

// MARK: - 真实照明体相对光谱功率（31 bin，400–700nm/10nm）
//
//  D65：CIE 015:2004 昼光照明体 10nm 表重采样，按 Σȳ(λ)S(λ) = Σȳ(λ) 归一；
//  A：Planck 辐射体 2856K（c2 = 1.4387769e7 nm·K，560nm 锚点归一形式）；
//  烛光：同式 1900K 黑体近似（油灯/烛焰色温的宝石学惯例值）；
//  E：各波长等功率参比。归一化与构建脚本、Metal CIE 积分白场完全一致
//  （完全色适应下中性物体在任意光源下均为白）。
public enum OpticsIlluminantSpectrum {
    /// 顺序与照明体枚举 rawValue / Metal spectralTable[0..4) 对应
    public static let tables: [[Float]] = [
        // D65 6504K
        [0.8372, 0.9251, 0.9443, 0.8766, 1.0606, 1.1829, 1.1910, 1.1617,
         1.1718, 1.1000, 1.1061, 1.0899, 1.0596, 1.0889, 1.0555, 1.0515,
         1.0111, 0.9736, 0.9686, 0.8968, 0.9099, 0.9059, 0.8867, 0.8422,
         0.8463, 0.8088, 0.8109, 0.8321, 0.7917, 0.7047, 0.7239],
        // CIE A 2856K
        [0.1458, 0.1752, 0.2081, 0.2446, 0.2845, 0.3279, 0.3748, 0.4249,
         0.4781, 0.5343, 0.5932, 0.6547, 0.7184, 0.7841, 0.8516, 0.9206,
         0.9908, 1.0619, 1.1338, 1.2060, 1.2784, 1.3507, 1.4228, 1.4942,
         1.5649, 1.6347, 1.7034, 1.7708, 1.8367, 1.9011, 1.9638],
        // 烛光 1900K
        [0.0211, 0.0296, 0.0407, 0.0551, 0.0733, 0.0960, 0.1240, 0.1580,
         0.1990, 0.2477, 0.3050, 0.3717, 0.4488, 0.5371, 0.6373, 0.7503,
         0.8768, 1.0174, 1.1727, 1.3433, 1.5297, 1.7320, 1.9508, 2.1860,
         2.4379, 2.7065, 2.9917, 3.2934, 3.6112, 3.9450, 4.2943],
        // E 等能白
        [Float](repeating: 1, count: 31),
        // 钠光 D 线 589.3nm：31 bin 表中落在 590nm bin（index 19）的单色线谱。
        // 峰值 14.11 使 Σȳ(λ)S(λ) 与其余照明体等亮度归一（≈10.68），
        // 正交偏光片段的钠光分支据此把明视觉亮度映射为 589.3nm 固定黄色度。
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 14.11,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
}

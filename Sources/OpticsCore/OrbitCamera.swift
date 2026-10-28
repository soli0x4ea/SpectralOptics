import simd
import CoreGraphics

// MARK: - 轨道相机（与 OpenGemCutting viewportOrbit.js 完全一致）
//
// 自 GemologyNotes 的 CutModels.swift 原样提取（该文件其余部分为业务数据，
// 不属于引擎）。yaw/elevation/zoom/pan 的语义是渲染取景的唯一事实源，
// OpticsRenderState.camera 与 OrbitGestureController 共用它。

public struct OrbitCamera {
    public var yaw: Float = -0.62
    public var elevation: Float = 0.42
    public var zoom: Float = 1
    public var panX: Float = 0
    public var panY: Float = 0

    public static let distance: Float = 4.4

    public init(yaw: Float = -0.62, elevation: Float = 0.42,
                zoom: Float = 1, panX: Float = 0, panY: Float = 0) {
        self.yaw = yaw
        self.elevation = elevation
        self.zoom = zoom
        self.panX = panX
        self.panY = panY
    }

    /// 拖拽旋转（水平/竖直均为弧度，右/下为正）；允许越过极点并反向补偿
    public mutating func orbit(horizontal: Float, vertical: Float) {
        let facing: Float = cos(elevation) < 0 ? -1 : 1
        yaw -= horizontal * facing
        elevation += vertical
    }

    public mutating func zoom(by factor: Float) {
        zoom = min(2.4, max(0.55, zoom * factor))
    }

    /// 捏合缩放：**方向语义只此一处**。
    ///
    /// 手指张开（UIPinchGestureRecognizer.scale 增大）= 模型变大。
    /// zoom 进 uniforms 时是 `baseScale / zoom`（OpticsRenderer.makeUniforms），
    /// 而屏幕上模型尺寸 ∝ zoom —— 所以张开必须让 zoom **变大**。
    /// 历史实现写成 `lastPinch / g.scale`（张开反而缩小），三处视口一起反了。
    public mutating func applyPinch(from previous: CGFloat, to current: CGFloat) {
        guard previous > 0, current > 0 else { return }
        zoom(by: Float(current / previous))
    }

    public mutating func reset() {
        self = OrbitCamera()
    }

    public struct Frame {
        public let position: SIMD3<Float>
        public let forward: SIMD3<Float>
        public let right: SIMD3<Float>
        public let up: SIMD3<Float>
    }

    public var frame: Frame {
        let h = cos(elevation)
        let position = SIMD3<Float>(
            sin(yaw) * h * OrbitCamera.distance,
            -cos(yaw) * h * OrbitCamera.distance,
            sin(elevation) * OrbitCamera.distance
        )
        let forward = simd_normalize(-position)
        let right = SIMD3<Float>(cos(yaw), sin(yaw), 0)
        let up = simd_normalize(simd_cross(right, forward))
        return Frame(position: position, forward: forward, right: right, up: up)
    }
}

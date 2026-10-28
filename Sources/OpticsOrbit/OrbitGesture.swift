#if canImport(UIKit)
// 本 product 仅 iOS / Mac Catalyst：整个文件在 macOS 构建下为空
// （SwiftPM 不支持按 target 声明平台，标准做法是 canImport 门控）。
import UIKit
import MetalKit
import simd

// MARK: - 轨道相机手势（多个 Metal 视口共用）
//
// 存在理由：多个视口各自持有 `camera` 副本 + `lastPan / lastPinch /
// isInteracting`，同一段状态机抄 N 遍必然分叉——捏合方向就是这么漏的：
// 全是 `lastPinch / g.scale`（手指张开 → zoom 变小 → 模型缩小），反了一整轮
// 没人发现。现在手势状态机与方向语义都只此一份，各视口只做接线。

/// 视口类：bounds 稳定后回调（用于同步 MTKView 的 drawableSize）。
public final class OrbitMTKView: MTKView {
    public var onLayout: ((CGSize) -> Void)?
    /// 挂窗后补一帧。本渲染模式（isPaused=true + 显式 draw()）的首帧 draw 发生在
    /// makeUIView/updateUIView 阶段——那时视图还没进窗口，Catalyst 上 CAMetalLayer
    /// 那次呈现不进合成器（iOS 会把最后一帧带进挂窗后的合成），表现为
    /// 「3D 区一片空，点一下鼠标（触发重画）模型才刷出来」。挂窗后必须主动补画。
    public var onWindowAttached: (() -> Void)?

    public override func layoutSubviews() {
        super.layoutSubviews()
        guard bounds.width > 1, bounds.height > 1 else { return }
        onLayout?(bounds.size)
    }

    public override func didMoveToWindow() {
        super.didMoveToWindow()
        // window == nil 是拆除方向的回调，不处理；重复挂窗多画一帧，无害
        if window != nil { onWindowAttached?() }
    }
}

/// 单指拖拽的作用对象。各视口共用同一套手势状态机，语义按视口分叉。
public enum OrbitDragTarget {
    /// 默认：绕场景飞（改相机 yaw/elevation，物体在世界系不动）
    case camera
    /// 偏光实验室（正交偏光 / 真实锥光 / 理想锥光三个视图档）：相机锁定，
    /// 手指直接扳物体。语义解释（哪条轴对应哪个角）由宿主在 onObjectPan
    /// 里定义，状态机只负责把「拖了多少度」递出去。
    case object
}

/// 记录「触摸/鼠标按下瞬间」位置的 pan。
///
/// `UIPanGestureRecognizer` 要移动约 10pt 才进入 `.began`，此刻 `g.location`
/// 已是移动后的点。切磨手柄只有 ~9pt，用移动后的点做命中会出现「明明按在
/// 旋钮上、拖起来却转了相机」。web 版在 pointerdown 的**按下点**判定
/// gizmoDrag，与此对齐：命中一律用 `pressLocation`，拿不到才退回当前点。
public final class OrbitPanGestureRecognizer: UIPanGestureRecognizer {
    public private(set) var pressLocation: CGPoint?

    public override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent) {
        if let touch = touches.first {
            pressLocation = touch.location(in: view)
        }
        super.touchesBegan(touches, with: event)
    }

    public override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent) {
        super.touchesEnded(touches, with: event)
        pressLocation = nil
    }

    public override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent) {
        super.touchesCancelled(touches, with: event)
        pressLocation = nil
    }
}

/// 轨道相机的手势状态机：单指旋转、双指平移、捏合缩放、双击复位；
/// Catalyst 上另有鼠标滚轮缩放（UIPanGestureRecognizer.allowedScrollTypesMask）。
///
/// 相机副本的同步纪律（历史 bug 的来源，改这里前先读）：
///  - 手势按**增量**累积，拖拽期间必须以副本为准；
///  - 但非手势期间副本必须与宿主侧的绑定值对齐（`syncCamera`），
///    否则下一次拖拽会从旧的（默认）视角接着算，视角整段跳走。
public final class OrbitGestureController: NSObject, UIGestureRecognizerDelegate {
    /// 旋转灵敏度（弧度 / 点），与 Web 端 viewportOrbit.js 一致
    public static let orbitPerPoint: Float = 0.008
    /// 双指平移灵敏度（归一化单位 / 点）
    public static let panPerPoint: Float = 0.002
    /// 物体拖拽灵敏度（度 / 点）。取 ~1°/pt：偏光台在屏幕上直径约 120pt，
    /// 手指拖过一颗宝石的宽度正好把它转过一圈 —— 与「抓住宝石边缘推」同感。
    /// 沿用相机轨道的 0.008 rad/pt（≈0.46°/pt）会显得拖半天转不动。
    public static let objectDegPerPoint: Float = 0.9
    /// 滚轮缩放灵敏度：zoom 因子 = exp(-dy·k)，滚轮向上（dy<0）拉近
    public static let wheelZoomPerPoint: Double = 0.008
    /// 缩放按钮每按一次的因子
    public static let buttonZoomFactor: Float = 1.25

    /// 手势累积用的相机副本
    public private(set) var camera: OrbitCamera
    /// 是否启用双指拖拽平移（量子实验室只做旋转）
    public let allowsTwoFingerPan: Bool

    /// 单指拖拽写谁。`.object` 时相机 yaw/elevation **完全不写**（该模式要的
    /// 就是「拖拽不再等于转相机」），捏合变焦与双击复位照旧生效。
    public var dragTarget: OrbitDragTarget = .camera

    /// 每一次增量后的回调
    public var onChange: ((OrbitCamera) -> Void)?
    /// 物体拖拽增量：单位**度**（屏幕向右为正、向下为正）。仅 `dragTarget == .object` 时触发。
    /// 用度而不是点：语义在这里定一次（见 objectDegPerPoint），视图侧只做角度累加。
    public var onObjectPan: ((_ dxDeg: Float, _ dyDeg: Float) -> Void)?
    /// 双击复位动作；nil 时走 `OrbitCamera.reset()`。
    /// 偏光实验室里被覆盖成「把样品放回台面正中、相机不动」（相机是该系统的
    /// 仪器，本来就锁在共轴观察位上），由调用方给。
    public var onReset: ((inout OrbitCamera) -> Void)?

    // MARK: 覆盖层（切磨 gizmo）手势托管
    //
    // 存在理由：切磨工作台的手柄层（角度弧桥 / 分度环 / 深度杆）曾经是盖在
    // MTKView 上的一个 SwiftUI Canvas，自带 DragGesture。即便用
    // `.simultaneousGesture(isEnabled:hovering)`，在 Mac Catalyst 上覆盖层
    // 仍会参与命中测试、吃掉滚轮与拖拽 —— 表现为「加一层切割涂层后鼠标转不动
    // 也缩不放宝石」。web 版（GemViewport.jsx `attachViewportInteractions`）
    // 只有**一个 canvas、一个 pointerdown 入口**：先判 gizmo，命中则整段拖拽
    // 交给 gizmo，否则落回相机轨道；滚轮恒为缩放。这里把同一条规则搬进唯一的
    // UIKit 手势入口，覆盖层因此可以彻底 `allowsHitTesting(false)`。

    /// 一次覆盖层拖拽会话：命中手柄后，控制器把本手势后续的移动/结束回吐给它。
    /// 会话对象在起手时捕获屏幕布局（弧/环/杆），与 web 版 `gizmoDrag` 同构。
    public struct OverlaySession {
        public var onMove: (CGPoint) -> Void
        public var onEnd: () -> Void

        public init(onMove: @escaping (CGPoint) -> Void, onEnd: @escaping () -> Void) {
            self.onMove = onMove
            self.onEnd = onEnd
        }
    }

    /// 视口覆盖层命中入口。单指 / 鼠标**按下**（不含滚轮）时先调它：
    /// 返回会话 → 这一整段手势交给覆盖层，相机不转、不平移；
    /// 返回 nil → 走原有相机轨道。滚轮缩放（numberOfTouches == 0）与捏合
    /// 始终不经过这里，因此手柄上方照样能缩放。其余视口不赋值，行为不变。
    public var overlayHitTest: ((CGPoint) -> OverlaySession?)?

    /// 当前进行中的覆盖层拖拽；nil = 本手势归相机。
    private var overlaySession: OverlaySession?

    private weak var view: UIView?
    private var isInteracting = false
    private var lastPan: CGPoint = .zero
    private var lastPinch: CGFloat = 1

    public init(camera: OrbitCamera = OrbitCamera(), allowsTwoFingerPan: Bool = true) {
        self.camera = camera
        self.allowsTwoFingerPan = allowsTwoFingerPan
        super.init()
    }

    /// 装手势到视口。`view` 只用来取平移量，弱引用。
    public func attach(to view: UIView) {
        self.view = view
        view.isMultipleTouchEnabled = true
        let pan = OrbitPanGestureRecognizer(target: self, action: #selector(pan(_:)))
        // 双指平移需要一个 recognizer 同时接住两指；单指旋转的视口不设上限
        // （保持历史行为：多指拖拽仍按旋转处理）
        if allowsTwoFingerPan { pan.maximumNumberOfTouches = 2 }
        // 滚轮/触控板滚动也走这同一个 recognizer（Catalyst 通道）。
        // **只有一个 pan**：单独再挂一个 wheel pan 会和触摸 pan 抢事件——
        // 鼠标拖拽也是 pan，两个 recognizer 竞争时拖拽可能落进 wheel 分支
        // 变成缩放（已踩）。分叉靠 handlePan 里的 numberOfTouches：滚动事件
        // 没有实体手指（==0），触摸拖拽 >=1。
        pan.allowedScrollTypesMask = [.discrete, .continuous]
        pan.delegate = self
        view.addGestureRecognizer(pan)
        let pinch = UIPinchGestureRecognizer(target: self, action: #selector(pinch(_:)))
        pinch.delegate = self
        view.addGestureRecognizer(pinch)
        let doubleTap = UITapGestureRecognizer(target: self, action: #selector(doubleTap(_:)))
        doubleTap.numberOfTapsRequired = 2
        doubleTap.delegate = self
        view.addGestureRecognizer(doubleTap)
        installZoomControls(in: view)
    }

    /// 缩放按钮（以及将来 gizmo）的编程入口：改相机 → 走同一条 onChange 通知链
    public func zoom(by factor: Float) {
        camera.zoom(by: factor)
        onChange?(camera)
    }

    /// 点在缩放按钮上的触摸不进轨道手势——否则连点两下 + 会误触双击复位。
    public func gestureRecognizer(_ g: UIGestureRecognizer, shouldReceive touch: UITouch) -> Bool {
        var v = touch.view
        while let current = v {
            if current is UIControl { return false }
            v = current.superview
        }
        return true
    }

    // MARK: 缩放按钮浮层（UIKit 造，多视口共用一份）

    /// 视口右下角竖排 +/- 胶囊。UIKit 而非 SwiftUI：OrbitGestureController
    /// 在 makeUIView 阶段 attach，SwiftUI body 拿不到 coordinator，而这套
    /// 浮层与主题无关（systemChromeMaterial 自适应明暗），放这里最省接线。
    private func installZoomControls(in view: UIView) {
        let stack = UIStackView(arrangedSubviews: [
            makeZoomButton(systemImage: "plus.magnifyingglass") { [weak self] in
                self?.zoom(by: Self.buttonZoomFactor)
            },
            makeZoomButton(systemImage: "minus.magnifyingglass") { [weak self] in
                self?.zoom(by: 1 / Self.buttonZoomFactor)
            },
        ])
        stack.axis = .vertical
        stack.spacing = 0

        let capsule = UIVisualEffectView(effect: UIBlurEffect(style: .systemChromeMaterial))
        capsule.layer.cornerRadius = 10
        capsule.layer.masksToBounds = true
        capsule.translatesAutoresizingMaskIntoConstraints = false
        capsule.contentView.addSubview(stack)
        stack.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            stack.topAnchor.constraint(equalTo: capsule.contentView.topAnchor, constant: 3),
            stack.bottomAnchor.constraint(equalTo: capsule.contentView.bottomAnchor, constant: -3),
            stack.leadingAnchor.constraint(equalTo: capsule.contentView.leadingAnchor, constant: 3),
            stack.trailingAnchor.constraint(equalTo: capsule.contentView.trailingAnchor, constant: -3),
        ])
        view.addSubview(capsule)
        NSLayoutConstraint.activate([
            capsule.trailingAnchor.constraint(
                equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -12),
            capsule.bottomAnchor.constraint(
                equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -12),
        ])
    }

    private func makeZoomButton(systemImage: String,
                                action: @escaping () -> Void) -> UIButton {
        var config = UIButton.Configuration.plain()
        config.image = UIImage(systemName: systemImage)
        config.preferredSymbolConfigurationForImage =
            UIImage.SymbolConfiguration(pointSize: 15, weight: .medium)
        let button = UIButton(type: .system)
        button.configuration = config
        button.translatesAutoresizingMaskIntoConstraints = false
        button.widthAnchor.constraint(equalToConstant: 34).isActive = true
        button.heightAnchor.constraint(equalToConstant: 34).isActive = true
        button.addAction(UIAction { _ in action() }, for: .touchUpInside)
        return button
    }

    /// 宿主侧 → 手势副本。拖拽期间拒绝回灌，否则增量累积会被绑定值反复拽回起点。
    public func syncCamera(_ c: OrbitCamera) {
        guard !isInteracting else { return }
        camera = c
    }

    /// 直接复位（复位令牌用；不触发回调）
    public func resetCamera() { camera.reset() }

    // MARK: 手势入口（@objc 只做转发，逻辑在下面）

    @objc private func pan(_ g: UIPanGestureRecognizer) { handlePan(g, in: view) }
    @objc private func pinch(_ g: UIPinchGestureRecognizer) { handlePinch(g) }
    @objc private func doubleTap(_ g: UITapGestureRecognizer) { handleDoubleTap() }

    public func handlePan(_ g: UIPanGestureRecognizer, in view: UIView?) {
        switch g.state {
        case .began:
            lastPan = .zero
            isInteracting = true
            overlaySession = nil
            // 仅实体指针（单指 / 鼠标左键）参与 gizmo 命中；滚轮 / 触控板滚动
            // numberOfTouches == 0，直接落进下面的缩放分支（对齐 web onWheel）。
            // 用真实按下点（OrbitPanGestureRecognizer.pressLocation）而非 pan
            // 识别后（已挪 ~10pt）的当前点，保证按在小手柄上一定抓得到。
            if g.numberOfTouches >= 1, let view {
                let pressPoint = (g as? OrbitPanGestureRecognizer)?.pressLocation
                    ?? g.location(in: view)
                overlaySession = overlayHitTest?(pressPoint)
            }
        case .changed:
            // 拖拽途中落下第二指 = 用户改意图为双指平移：释放覆盖层，交还相机。
            // 把 lastPan 对齐到当前位移，避免切换瞬间视角跳一下。
            if overlaySession != nil, g.numberOfTouches >= 2, let view {
                overlaySession?.onEnd()
                overlaySession = nil
                lastPan = g.translation(in: view)
            }
            // 覆盖层拖拽：回吐视口坐标，相机完全不动。
            if let session = overlaySession {
                session.onMove(g.location(in: view))
                return
            }
            guard let view else { return }
            // 滚轮/触控板滚动：没有实体手指（numberOfTouches == 0）。
            // 必须先于触摸分支判断——鼠标拖拽也是 pan，落错分支就变缩放（已踩）。
            if g.numberOfTouches == 0 {
                let dy = Float(g.translation(in: view).y)
                guard dy != 0 else { return }
                g.setTranslation(.zero, in: view)
                // 每个事件限幅：动量阶段的大 dy 不至于一次跳过整档缩放
                let factor = min(1.5, max(1 / 1.5,
                    Float(exp(-Double(dy) * Self.wheelZoomPerPoint))))
                camera.zoom(by: factor)
                onChange?(camera)
                return
            }
            let t = g.translation(in: view)
            let dx = Float(t.x - lastPan.x)
            let dy = Float(t.y - lastPan.y)
            lastPan = t
            if dragTarget == .object {
                // 物体模式下**所有** pan 都扳物体（含多指拖拽）：此时相机是锁的，
                // 把多指拖拽吞掉会让「两根手指随便推一下」变成什么都没发生。
                onObjectPan?(dx * Self.objectDegPerPoint, dy * Self.objectDegPerPoint)
            } else if allowsTwoFingerPan, g.numberOfTouches >= 2 {
                camera.panX += dx * Self.panPerPoint
                camera.panY -= dy * Self.panPerPoint
            } else {
                camera.orbit(horizontal: dx * Self.orbitPerPoint,
                             vertical: dy * Self.orbitPerPoint)
            }
            onChange?(camera)
        case .ended, .cancelled, .failed:
            overlaySession?.onEnd()
            overlaySession = nil
            isInteracting = false
        default:
            break
        }
    }

    public func handlePinch(_ g: UIPinchGestureRecognizer) {
        switch g.state {
        case .began:
            // lastPinch 必须在开始时对齐本次手势的起始 scale，否则会沿用
            // 上一次捏合结束时的值，第一次增量直接跳一大截。
            lastPinch = g.scale
            isInteracting = true
        case .changed:
            camera.applyPinch(from: lastPinch, to: g.scale)
            lastPinch = g.scale
            onChange?(camera)
        case .ended, .cancelled, .failed:
            isInteracting = false
        default:
            break
        }
    }

    public func handleDoubleTap() {
        if let onReset {
            var c = camera
            onReset(&c)
            camera = c
        } else {
            camera.reset()
        }
        onChange?(camera)
    }
}

#else
import Foundation
#endif

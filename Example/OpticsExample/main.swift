//
//  main.swift
//  OpticsExample
//
//  SpectralOptics / OpticsCore 的 macOS 最小示例：
//  用 OpticsMeshBuilder 构建一颗八面体（凹凸通吃的网格路径），
//  以钻石参数（n=2.417，色散 0.044）做 31 波长真光谱路径追踪。
//
//  交互：拖拽旋转，滚轮缩放（语义与 OpticsOrbit.OrbitGestureController 一致，
//  但示例只用 AppKit 事件，保持 OpticsCore 的 macOS 可用性演示）。
//
//  来源与许可：本示例为原创代码（MIT）。包含源自 OpenGemCutting（MIT）衍生
//  部分的引擎，其来源与许可边界见仓库 THIRD_PARTY_NOTICES.md 与
//  docs/compliance/PROVENANCE.md。
//
//  运行：swift run OpticsExample
//

import AppKit
import MetalKit
import simd
import OpticsCore

// MARK: - 视图（AppKit 轨道交互）

final class DemoOrbitView: MTKView {
    /// 与 OpticsOrbit.OrbitGestureController 相同的灵敏度常量
    private static let orbitPerPoint: Float = 0.008
    private static let wheelZoomPerPoint: Double = 0.008

    var camera = OrbitCamera()
    weak var renderer: OpticsRenderer?
    var onCameraChange: (() -> Void)?

    override func mouseDragged(with event: NSEvent) {
        camera.orbit(horizontal: Float(event.deltaX) * Self.orbitPerPoint,
                     vertical: Float(event.deltaY) * Self.orbitPerPoint)
        onCameraChange?()
    }

    override func scrollWheel(with event: NSEvent) {
        let dy = event.hasPreciseScrollingDeltas ? event.scrollingDeltaY : event.deltaY * 10
        guard dy != 0 else { return }
        camera.zoom(by: Float(exp(-Double(dy) * Self.wheelZoomPerPoint)))
        onCameraChange?()
    }

    override func layout() {
        super.layout()
        renderer?.syncDrawableSize(for: self)
    }
}

// MARK: - 几何：八面体（6 顶点 / 8 面，晶形多面体的最小样本）

func makeOctahedron() -> OpticsMeshState? {
    let vertices: [SIMD3<Double>] = [
        SIMD3( 1, 0, 0), SIMD3(-1, 0, 0),   // x±
        SIMD3(0,  1, 0), SIMD3(0, -1, 0),   // y±
        SIMD3(0, 0,  1), SIMD3(0, 0, -1),   // z±
    ]
    var faces: [[Int]] = []
    var normals: [SIMD3<Double>?] = []
    for sx in [1.0, -1.0] {
        for sy in [1.0, -1.0] {
            for sz in [1.0, -1.0] {
                // 每面取三个轴上对应符号的顶点；参考法向 = (sx, sy, sz) 归一化
                faces.append([sx > 0 ? 0 : 1, sy > 0 ? 2 : 3, sz > 0 ? 4 : 5])
                normals.append(simd_normalize(SIMD3<Double>(sx, sy, sz)))
            }
        }
    }
    return OpticsMeshBuilder.build(
        OpticsMeshBuilder.Input(vertices: vertices, faces: faces, faceNormals: normals))
}

// MARK: - 应用

final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow?
    var renderer: OpticsRenderer?

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard let renderer = OpticsRenderer.make() else {
            fputs("OpticsRenderer 初始化失败（无 Metal 设备或着色器编译失败）\n", stderr)
            exit(1)
        }
        self.renderer = renderer
        guard let mesh = makeOctahedron() else {
            fputs("八面体网格构建失败\n", stderr)
            exit(1)
        }

        let view = DemoOrbitView(frame: NSRect(x: 0, y: 0, width: 720, height: 640),
                                 device: renderer.device)
        renderer.attach(to: view)
        // 演示用连续重绘；生产环境按 isPaused=true + requestRender 按需绘制
        view.isPaused = false
        view.renderer = renderer

        let state = OpticsRenderState(
            planes: [],
            ior: 2.417, dispersion: 0.044,            // 钻石
            absorptionAxisX: .zero, absorptionAxisY: .zero, absorptionAxisZ: .zero,
            spectrumAxes: nil,                         // 无光谱吸收 = 近无色
            illuminant: 0,                             // D65
            environment: 0,
            background: SIMD3<Float>(0xEF / 255, 0xF1 / 255, 0xF1 / 255),
            exposure: 0,
            environmentRotation: 18,
            maxBounces: 6,
            camera: view.camera,
            polar: nil,                                // 普通路径追踪（偏光传 OpticsPolarRenderState）
            mesh: mesh                                 // 网格路径（凹面支持）
        )
        renderer.update(state)
        renderer.syncDrawableSize(for: view)

        view.onCameraChange = { [weak view, weak renderer] in
            guard let view, let renderer else { return }
            var s = renderer.state
            s.camera = view.camera
            renderer.update(s)
            renderer.requestRender(in: view)
        }

        let window = NSWindow(
            contentRect: view.frame,
            styleMask: [.titled, .closable, .resizable],
            backing: .buffered, defer: false)
        window.title = "SpectralOptics — 八面体光谱路径追踪（拖拽旋转 / 滚轮缩放）"
        window.contentView = view
        window.center()
        window.makeKeyAndOrderFront(nil)
        self.window = window
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationSupportsSecureRestorableState(_ app: NSApplication) -> Bool { true }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()

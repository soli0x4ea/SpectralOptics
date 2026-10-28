import XCTest
import simd
import Metal
import OpticsCore

// MARK: - Uniform 二进制布局 / 照明体表 / 渲染器初始化 的闸门
//
// 渲染器 init 内的 `assert(stride == 336)` 是启动期唯一防线，但 assert 在
// -O 构建下消失；这条测试靶把它补成**常驻**断言，并逐字段校验字节偏移 ——
// Swift 与 MSL 字段错位不会编译失败，只会让画面静默用错参数。

final class OpticsUniformsLayoutTests: XCTestCase {

    /// 总 stride：336 字节（构成见 OpticsRenderer.init 内注释）
    func testUniformsStrideIs336() {
        XCTAssertEqual(MemoryLayout<OpticsUniforms>.stride, 336,
                       "OpticsUniforms 布局与 optics_lab.metal 的 Uniforms 不一致 —— "
                       + "改任何一侧必须同时改另一侧与本测试")
    }

    /// 逐字段字节偏移：用哨兵值填充后按字节重解释，检查每个 SIMD4 落在
    /// 与 MSL 布局一一对应的位置（基础 208 + 偏光 80 + 球面 16 + 偏光片 16 + 姿态 16）。
    func testUniformsFieldByteOffsets() {
        func f(_ v: Float) -> SIMD4<Float> { SIMD4(v, v + 0.25, v + 0.5, v + 0.75) }
        var u = OpticsUniforms(
            cameraPositionScale: f(0),
            cameraForwardExposure: f(4),
            cameraRightEnvRot: f(8),
            cameraUpFocal: f(12),
            bodyColorIor: f(16),
            observerDispersion: f(20),
            backgroundAbsorption: f(24),
            absorptionAxisX: f(28),
            absorptionAxisY: f(32),
            absorptionAxisZ: f(36),
            panParams: f(40),
            flags: SIMD4<Int32>(101, 102, 103, 104),
            resolution: SIMD2<Float>(55, 66),
            polarParams: f(60),
            crystalA: f(64),
            crystalB: f(68),
            polarFlags: f(72),
            crystalOrient: f(76),
            sphereParams: f(80),
            polarLab: f(84),
            gemTumble: f(88))

        let floats: [Float] = withUnsafeBytes(of: &u) {
            Array($0.bindMemory(to: Float.self))
        }
        let int32s: [Int32] = withUnsafeBytes(of: &u) {
            Array($0.bindMemory(to: Int32.self))
        }

        // float4 期望值检查：bias = 字节偏移 / 4
        func expectF4(_ bias: Int, _ v: Float, _ name: String,
                      file: StaticString = #filePath, line: UInt = #line) {
            for k in 0..<4 {
                XCTAssertEqual(floats[bias + k], v + Float(k) * 0.25, accuracy: 1e-6,
                               "\(name) 第 \(k) 分量偏移错误（应为字节 \(bias * 4 + k * 4)）",
                               file: file, line: line)
            }
        }
        // 基础段 0..208（每 16 字节一个 float4）
        for (i, name) in ["cameraPositionScale", "cameraForwardExposure", "cameraRightEnvRot",
                          "cameraUpFocal", "bodyColorIor", "observerDispersion",
                          "backgroundAbsorption", "absorptionAxisX", "absorptionAxisY",
                          "absorptionAxisZ", "panParams"].enumerated() {
            expectF4(i * 4, Float(i * 4), name)
        }
        // flags @176（int4）
        for k in 0..<4 {
            XCTAssertEqual(int32s[44 + k], Int32(101 + k),
                           "flags 第 \(k) 分量偏移错误（应为字节 \(176 + k * 4)）")
        }
        // resolution @192（float2）
        XCTAssertEqual(floats[48], 55, accuracy: 1e-6)
        XCTAssertEqual(floats[49], 66, accuracy: 1e-6)
        // 偏光段 208..288 / 球面 288 / 偏光片 304 / 姿态 320
        expectF4(52, 60, "polarParams")
        expectF4(56, 64, "crystalA")
        expectF4(60, 68, "crystalB")
        expectF4(64, 72, "polarFlags")
        expectF4(68, 76, "crystalOrient")
        expectF4(72, 80, "sphereParams")
        expectF4(76, 84, "polarLab")
        expectF4(80, 88, "gemTumble")
    }

    /// 照明体 SPD 表：5 张 × 31 bin、全部有限、钠光单峰在 590nm bin。
    func testIlluminantTables() {
        XCTAssertEqual(OpticsIlluminantSpectrum.tables.count, 5)
        for (t, spd) in OpticsIlluminantSpectrum.tables.enumerated() {
            XCTAssertEqual(spd.count, 31, "表 \(t) 应有 31 bin")
            for v in spd where !v.isFinite || v < 0 {
                XCTFail("表 \(t) 出现非法值 \(v)")
            }
        }
        // 钠光（表 4）：单峰在 bin 19（590nm bin，D 线 589.3nm）
        let sodium = OpticsIlluminantSpectrum.tables[4]
        for (b, v) in sodium.enumerated() {
            XCTAssertEqual(v > 0, b == 19, "钠光表 bin \(b) 应为 \(b == 19 ? "峰" : "零")")
        }
        // E 等能白：全 1
        XCTAssertEqual(OpticsIlluminantSpectrum.tables[3].allSatisfy { $0 == 1 }, true)
    }

    /// 渲染器冒烟：真实 GPU 上运行时编译着色器 + 建全部管线 + 分配缓冲。
    /// 缺 Metal 设备时**失败而非跳过**（渲染正确性闸门的标准）。
    func testRendererInitOnRealDevice() {
        XCTAssertNotNil(OpticsRenderer.make(),
                        "无 Metal 设备或着色器编译失败：渲染器必须可用（不允许静默跳过）")
    }

    /// resolvedGeometry：解析球面 / 网格 / 晶胞接管几何时平面表必须清空。
    func testResolvedGeometryPrecedence() {
        func state(sphere: SIMD4<Float>? = nil,
                   mesh: OpticsMeshState? = nil,
                   cell: OpticsCellState? = nil) -> OpticsRenderState {
            OpticsRenderState(
                planes: [SIMD4(0, 0, 1, 0.5), SIMD4(0, 1, 0, 0.3)],
                ior: 1.6, dispersion: 0,
                absorptionAxisX: .zero, absorptionAxisY: .zero, absorptionAxisZ: .zero,
                spectrumAxes: nil, illuminant: 0, environment: 0,
                background: .zero, exposure: 0, environmentRotation: 0, maxBounces: 6,
                camera: OrbitCamera(), polar: nil,
                sphere: sphere, mesh: mesh, cell: cell)
        }
        // 纯平面：原样通过
        let plain = OpticsRenderer.resolvedGeometry(state())
        XCTAssertEqual(plain.planes.count, 2)
        // 球面接管：平面清空、球面原样返回
        let sphere = OpticsRenderer.resolvedGeometry(state(sphere: SIMD4(0, 0, 0, 1)))
        XCTAssertEqual(sphere.planes.count, 0)
        XCTAssertEqual(sphere.sphere.w, 1, accuracy: 1e-6)
        // 网格 / 晶胞接管：平面清空
        let meshFrame = OpticsMeshBuilder.Normalization(center: .zero, scale: 1, radius: 1)
        let mesh = OpticsRenderer.resolvedGeometry(state(mesh: OpticsMeshState(
            nodes: [SIMD4(repeating: 0), SIMD4(repeating: 0), SIMD4(1, 0, 0, 0)],
            triangles: [SIMD4(repeating: 0), SIMD4(repeating: 0),
                        SIMD4(repeating: 0), SIMD4(repeating: 0)],
            radius: 1, normalization: meshFrame)))
        XCTAssertEqual(mesh.planes.count, 0)
        let cell = OpticsRenderer.resolvedGeometry(state(cell: OpticsCellState(
            atoms: [SIMD4(repeating: 0), SIMD4(repeating: 1)], capsules: [], radius: 1)))
        XCTAssertEqual(cell.planes.count, 0)
    }
}

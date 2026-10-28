import XCTest
import Metal
import simd
import OpticsCore

// MARK: - 光学引擎 · Metal vs CPU 参考 diff 验收 harness
//
//  对每组固定输入（固定琢型/材质/相机/环境），用与渲染器完全相同的 uniforms 组装点
//  分别驱动 ① Metal 离屏渲染（真实管线）和 ② OpticsCPUReference（CPU 逐行镜像），
//  逐像素比对 8-bit 输出：任一 RGB 通道绝对差 > 8 计入失配。
//
//  方法学来自 OpenGemCutting 的 optics-webgpu-reproduction（yuyou-dev）：
//   - 固定输入，不采样随机数（本路径追踪无蒙特卡洛项，两边都是确定性的）；
//   - 缺 Metal / 缺着色器函数时**失败而非跳过** —— 静默跳过会让 harness
//     在没有测任何东西的机器上报「通过」；
//   - 正向断言（两边一致）必须配反面断言（故意污染 CPU 输入，断言 harness 必须爆红），
//     否则无法回答「这条断言真的咬得住吗」。
//
//  覆盖：普通模式（凸多面体平面表 + 解析球面）。偏光实验室路径未覆盖。
//  琢型几何取自 Tests/OpticsCoreTests/Fixtures/cuts（源自 OpenGemCutting
//  MIT 时代数据蒸馏，见 THIRD_PARTY_NOTICES.md）。
final class OpticsDiffHarnessTests: XCTestCase {

    /// 64×64 足以覆盖背景/轮廓/内部反射多类像素，CPU 参考全图 <1s
    private let texWidth = 64
    private let texHeight = 64
    /// 8/255 ≈ 3%：GPU fast-math 与 CPU 的浮点差经 ACES+gamma 放大后仍在 ±1 以内，
    /// 阈值 8 留出整个数量级的余量 —— 超过它就不是精度问题，是数学错了。
    private let channelThreshold = 8

    // MARK: GPU 侧（离屏渲染）

    private func renderGPU(state: OpticsRenderState) throws -> [UInt8] {
        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("无 Metal 设备：diff harness 无法运行（不允许静默跳过）")
            return []
        }
        // 包分发形态：着色器经 OpticsShaderLibrary 从 Bundle.module 源码编译
        // （与渲染器同一条加载路径，顺带覆盖运行时编译的正确性）
        guard let library = try? OpticsShaderLibrary.makeLibrary(for: device),
              let vertex = library.makeFunction(name: "optics_vertex"),
              let fragment = library.makeFunction(name: "optics_fragment") else {
            XCTFail("包内着色器编译失败或未找到 optics_vertex/optics_fragment")
            return []
        }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertex
        descriptor.fragmentFunction = fragment
        descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        let pipeline = try device.makeRenderPipelineState(descriptor: descriptor)

        // 几何/光谱与渲染器同源：resolvedGeometry / makeUniforms / fillSpectral* 只此一份
        let (planes, sphere) = OpticsRenderer.resolvedGeometry(state)
        var uniforms = OpticsRenderer.makeUniforms(
            state: state, width: Float(texWidth), height: Float(texHeight))
        // makeUniforms 内部走 state.sphere 重新解析，与上面解构结果一致；此处仅防手抖
        assert(uniforms.sphereParams == sphere && uniforms.flags.z == Int32(planes.count))

        let spectralRows = spectralRows(state: state)

        guard let planeBuffer = device.makeBuffer(length: OpticsRenderer.maxPlanes * 16,
                                                  options: .storageModeShared),
              let spectralBuffer = device.makeBuffer(
                length: OpticsRenderer.spectralFloats * MemoryLayout<Float>.stride,
                options: .storageModeShared),
              let rotationBuffer = device.makeBuffer(length: 64, options: .storageModeShared),
              let uniformBuffer = device.makeBuffer(
                length: MemoryLayout<OpticsUniforms>.stride, options: .storageModeShared) else {
            XCTFail("GPU 缓冲分配失败")
            return []
        }
        if !planes.isEmpty {
            planes.withUnsafeBufferPointer { src in
                planeBuffer.contents().copyMemory(
                    from: src.baseAddress!, byteCount: planes.count * 16)
            }
        }
        spectralRows.withUnsafeBufferPointer { src in
            spectralBuffer.contents().copyMemory(
                from: src.baseAddress!, byteCount: spectralRows.count * 16)
        }
        uniformBuffer.contents().copyMemory(from: &uniforms,
                                            byteCount: MemoryLayout<OpticsUniforms>.stride)

        let texDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: texWidth, height: texHeight, mipmapped: false)
        texDesc.usage = [.renderTarget, .shaderRead]
        guard let texture = device.makeTexture(descriptor: texDesc),
              let queue = device.makeCommandQueue(),
              let buffer = queue.makeCommandBuffer() else {
            XCTFail("离屏纹理/命令队列创建失败")
            return []
        }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = texture
        pass.colorAttachments[0].loadAction = .dontCare
        pass.colorAttachments[0].storeAction = .store
        guard let encoder = buffer.makeRenderCommandEncoder(descriptor: pass) else {
            XCTFail("离屏 render encoder 创建失败")
            return []
        }
        encoder.setRenderPipelineState(pipeline)
        encoder.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
        encoder.setFragmentBuffer(uniformBuffer, offset: 0, index: 1)
        encoder.setFragmentBuffer(planeBuffer, offset: 0, index: 2)
        encoder.setFragmentBuffer(spectralBuffer, offset: 0, index: 3)
        encoder.setFragmentBuffer(rotationBuffer, offset: 0, index: 4)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        buffer.commit()
        buffer.waitUntilCompleted()
        if let error = buffer.error {
            XCTFail("Metal 提交失败（command buffer error 不该被吞掉）: \(error)")
            return []
        }

        // 读回：blit → 共享 buffer。64px × 4B = 256B/行，天然满足对齐要求。
        // 必须用**新的** command buffer：首个 buffer 已 commit，不能再追加 encoder。
        let bytesPerRow = texWidth * 4
        guard let readback = device.makeBuffer(length: bytesPerRow * texHeight,
                                               options: .storageModeShared),
              let blitBuffer = queue.makeCommandBuffer(),
              let blit = blitBuffer.makeBlitCommandEncoder() else {
            XCTFail("读回缓冲/blit 创建失败")
            return []
        }
        blit.copy(from: texture, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: texWidth, height: texHeight, depth: 1),
                  to: readback, destinationOffset: 0, destinationBytesPerRow: bytesPerRow,
                  destinationBytesPerImage: bytesPerRow * texHeight)
        blit.endEncoding()
        blitBuffer.commit()
        blitBuffer.waitUntilCompleted()
        if let error = blitBuffer.error {
            XCTFail("blit 读回失败: \(error)")
            return []
        }
        let raw = readback.contents().bindMemory(to: UInt8.self, capacity: bytesPerRow * texHeight)
        return Array(UnsafeBufferPointer(start: raw, count: bytesPerRow * texHeight))
    }

    // MARK: CPU 侧 + 比对

    private func cpuPixels(state: OpticsRenderState, mutate: ((inout OpticsCPUReference.Scene) -> Void)? = nil) -> [SIMD3<Float>] {
        let (planes, _) = OpticsRenderer.resolvedGeometry(state)
        var scene = OpticsCPUReference.Scene(
            uniforms: OpticsRenderer.makeUniforms(
                state: state, width: Float(texWidth), height: Float(texHeight)),
            planes: planes,
            spectral: spectralRows(state: state))
        mutate?(&scene)
        var pixels = [SIMD3<Float>]()
        pixels.reserveCapacity(texWidth * texHeight)
        for r in 0..<texHeight {
            for c in 0..<texWidth {
                // 读回 buffer 行 0 是纹理顶部（NDC y=+1）；顶点 uv.y = ndc*0.5+0.5，
                // 故 uv.y = 1 − (r+0.5)/h。搞反的话背景渐变会整幅颠倒，harness 立刻爆红。
                let uv = SIMD2<Float>((Float(c) + 0.5) / Float(texWidth),
                                      1.0 - (Float(r) + 0.5) / Float(texHeight))
                pixels.append(OpticsCPUReference.shade(scene, uv: uv))
            }
        }
        return pixels
    }

    private struct DiffReport {
        var mismatchCount = 0
        var total = 0
        var maxChannelError = 0
        var meanAbsoluteChannelError: Float = 0
        var worstSamples: [String] = []
    }

    private func compare(cpu: [SIMD3<Float>], gpu: [UInt8]) -> DiffReport {
        var report = DiffReport(total: texWidth * texHeight)
        var totalError = 0
        for r in 0..<texHeight {
            for c in 0..<texWidth {
                let index = r * texWidth + c
                let base = index * 4
                // bgra8Unorm 的字节序是 B,G,R,A
                let gpuRGB = [Int(gpu[base + 2]), Int(gpu[base + 1]), Int(gpu[base])]
                // SIMD3 在该工具链上没有 map（仅有 2 分量 swizzle），逐分量量化
                let cpuPixel = cpu[index]
                let cpuRGB = [Self.quantize(cpuPixel.x), Self.quantize(cpuPixel.y),
                              Self.quantize(cpuPixel.z)]
                var maxDiff = 0
                for ch in 0..<3 {
                    let d = abs(cpuRGB[ch] - gpuRGB[ch])
                    maxDiff = max(maxDiff, d)
                    totalError += d
                }
                report.maxChannelError = max(report.maxChannelError, maxDiff)
                if maxDiff > channelThreshold {
                    report.mismatchCount += 1
                    if report.worstSamples.count < 5 {
                        report.worstSamples.append(
                            "(\(c),\(r)) cpu=\(cpuRGB) gpu=\(gpuRGB) maxΔ=\(maxDiff)")
                    }
                }
            }
        }
        report.meanAbsoluteChannelError = Float(totalError) / Float(report.total * 3)
        return report
    }

    /// 正向断言：两边一致。失配信息带最差样本坐标，方便定位到着色器的哪一段。
    private func assertMatches(_ state: OpticsRenderState,
                               file: StaticString = #filePath, line: UInt = #line) throws {
        let gpu = try renderGPU(state: state)
        let cpu = cpuPixels(state: state)
        let report = compare(cpu: cpu, gpu: gpu)
        XCTAssertEqual(report.mismatchCount, 0,
                       "Metal 与 CPU 参考失配 \(report.mismatchCount)/\(report.total) px，"
                       + "maxΔ=\(report.maxChannelError)，meanΔ=\(report.meanAbsoluteChannelError)；"
                       + "最差样本 \(report.worstSamples)",
                       file: file, line: line)
    }

    // MARK: 输入构造

    private func baseState(planes: [SIMD4<Float>] = [],
                           sphere: SIMD4<Float>? = nil) -> OpticsRenderState {
        OpticsRenderState(
            planes: planes, ior: 2.417, dispersion: 0.044,
            absorptionAxisX: SIMD3<Float>(0, 0, 0),
            absorptionAxisY: SIMD3<Float>(0, 0, 0),
            absorptionAxisZ: SIMD3<Float>(0, 0, 0),
            spectrumAxes: nil, illuminant: 0,
            environment: 0, background: SIMD3<Float>(0xEF / 255, 0xF1 / 255, 0xF1 / 255),
            exposure: 0, environmentRotation: 18, maxBounces: 6,
            camera: OrbitCamera(), polar: nil, sphere: sphere)
    }

    /// 从测试夹具读真实琢型几何（源自 OpenGemCutting MIT 时代数据，见 NOTICE）
    private func loadCutPlanes(_ id: String) throws -> [SIMD4<Float>] {
        struct CutGeometry: Decodable { let planes: [[Double]] }
        guard let url = Bundle.module.url(forResource: id, withExtension: "json",
                                          subdirectory: "Fixtures/cuts"),
              let geo = try? JSONDecoder().decode(CutGeometry.self,
                                                  from: Data(contentsOf: url)) else {
            XCTFail("测试夹具中未找到琢型几何 \(id).json")
            return []
        }
        return geo.planes.map {
            SIMD4<Float>(Float($0[0]), Float($0[1]), Float($0[2]), Float($0[3]))
        }
    }

    /// 与 spectralBuffer 逐字节一致的 64 行表：照明体 5×8 行 + 三主轴 3×8 行
    private func spectralRows(state: OpticsRenderState) -> [SIMD4<Float>] {
        var rows = [SIMD4<Float>](repeating: SIMD4<Float>(repeating: 0), count: 64)
        for (t, spd) in OpticsIlluminantSpectrum.tables.enumerated() {
            for b in 0..<31 { rows[t * 8 + b / 4][b % 4] = spd[b] }
        }
        let packed = state.spectrumAxes
        for axis in 0..<3 {
            for q in 0..<8 {
                rows[40 + axis * 8 + q] = (axis < (packed?.count ?? 0))
                    ? packed![axis][q] : SIMD4<Float>(repeating: 0)
            }
        }
        return rows
    }

    private static func quantize(_ ch: Float) -> Int {
        Int((simd_clamp(ch, 0, 1) * 255).rounded())
    }

    /// 人造多色性光谱：X 轴偏红吸收（透蓝绿）、Y 轴偏蓝吸收（透橙红）、Z 轴轻微中性
    private func craftedSpectrumAxes() -> [[SIMD4<Float>]] {
        (0..<3).map { axis in
            (0..<8).map { q -> SIMD4<Float> in
                var comps = [Float](repeating: 0, count: 4)
                for i in 0..<4 {
                    let bin = q * 4 + i
                    guard bin < 31 else { continue }
                    switch axis {
                    case 0: comps[i] = 0.6 * Float(bin) / 30
                    case 1: comps[i] = 0.6 * (1 - Float(bin) / 30)
                    default: comps[i] = 0.15
                    }
                }
                return SIMD4<Float>(comps[0], comps[1], comps[2], comps[3])
            }
        }
    }

    // MARK: 正向用例

    func testRoundBrilliantMatchesCPUReference() throws {
        // 经典圆形明亮式（53 平面）：主路径 = 平面表 + 三族色散 + D65
        var state = baseState(planes: try loadCutPlanes("100488-pc-07-159-rbc-281"))
        state.ior = 2.417
        state.dispersion = 0.044
        try assertMatches(state)
    }

    func testAnalyticSphereWithPanMatchesCPUReference() throws {
        // 解析球面分支 + 平移/缩放/烛光环境：覆盖 sphereParams 与 pan/zoom 路径
        var state = baseState(sphere: SIMD4<Float>(0, 0, 0, 1))
        state.ior = 1.772
        state.dispersion = 0.018
        state.camera.panX = 0.12
        state.camera.panY = -0.05
        state.camera.zoom = 1.4
        state.illuminant = 2
        state.environment = 1
        try assertMatches(state)
    }

    func testPleochroicSpectrumMatchesCPUReference() throws {
        // 多色性吸收张量 + hearts 环境（env==3 的 observerCard 重赋分支）
        var state = baseState(planes: try loadCutPlanes("100488-pc-07-159-rbc-281"))
        state.ior = 1.766
        state.dispersion = 0.018
        state.spectrumAxes = craftedSpectrumAxes()
        state.illuminant = 3
        state.environment = 3
        state.environmentRotation = 27
        try assertMatches(state)
    }

    func testSodiumMonochromaticMatchesCPUReference() throws {
        // 钠光分支：不做色适应，589.3nm 固定黄色度映射
        var state = baseState(planes: try loadCutPlanes("100058-pc-07-001c-square-emerald-1-4"))
        state.ior = 1.577
        state.dispersion = 0.014
        state.illuminant = 4
        state.maxBounces = 8
        try assertMatches(state)
    }

    // MARK: 反面断言

    /// harness 必须咬得住：只污染 CPU 侧输入（GPU 不变），色散 +0.02 会改变
    /// 三族 IOR 与全部内部光路，若比对仍报「一致」说明 harness 是哑的。
    /// 这是对「正向断言两边一致」的反面配对 —— 断言它必须落在别处。
    func testHarnessBitesWhenCPUInputCorrupted() throws {
        var state = baseState(planes: try loadCutPlanes("100488-pc-07-159-rbc-281"))
        state.ior = 2.417
        state.dispersion = 0.044
        let gpu = try renderGPU(state: state)
        let cpu = cpuPixels(state: state) { scene in
            scene.uniforms.observerDispersion.w += 0.02
        }
        let report = compare(cpu: cpu, gpu: gpu)
        XCTAssertGreaterThan(report.mismatchCount, report.total / 50,
                             "污染 CPU 色散后失配仅 \(report.mismatchCount)/\(report.total)，"
                             + "harness 阈值/流水线疑似失效（哑警报比没有警报更危险）")
    }
}

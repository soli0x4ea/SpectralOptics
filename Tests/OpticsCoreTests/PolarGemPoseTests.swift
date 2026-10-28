import XCTest
import simd
import OpticsCore

// MARK: - 偏光姿态（正交 3D「扳宝石」）的约定闸门
//
// 这一组测的是**约定**，不是观感：姿态分解 R(世界←宝石) = Rz(specimenRad)·T
//（specimenRad = 载物台角 + 宝石自转）一共有四种写法（搬轴/不搬轴 × 左乘/右乘），
// 其中三种也能把宝石转起来、也能转台、画面也「像那么回事」，只在某一个台角下
// 不跟手 —— 靠肉眼极难定位。所以每条正向断言都配一条反面断言（构造那个错实现，
// 断言它必须落在别处），否则无法回答「这条断言真的咬得住吗」。
//
// 移植说明：App 侧另有 testSpecimenSpinIsStagePlusSpin（验 PolariscopeSettings
// 的合成口径），那属于宿主视图层状态，不在本包 API 面内，不移植。

final class PolarGemPoseTests: XCTestCase {

    private let deg = Float.pi / 180

    /// 宝石自身轴（晶体 z）在世界系里的方向。
    /// 传**总方位角**（台角 + 自转），与视图/着色器同一口径。
    private func gemAxisWorld(_ tumble: simd_quatf, specimenDeg: Double) -> SIMD3<Float> {
        PolarGemPose.toWorld(PolarGemPose.stageAxis, tumble: tumble,
                             specimenRad: Float(specimenDeg) * deg)
    }

    /// **错误实现**：把世界轴直接当台面轴左乘（少一次 Rz(−stage) 搬轴）。
    /// 反面断言用它证明正向断言真的约束了「轴要搬进台面系」这件事。
    private func unTransportedTilt(_ tumble: simd_quatf, worldAxis: SIMD3<Float>,
                                   deltaRad: Float) -> simd_quatf {
        (simd_quatf(angle: deltaRad, axis: worldAxis) * tumble).normalized
    }

    /// 在 90° 与 217° 台角下做同一次竖直拖拽，宝石必须落在**同一个世界方向**：
    /// 手感要求「手指往下推 = 宝石北极朝屏幕下倒」，与台角无关。
    /// 沿世界 x 轴竖翻 90° 后，宝石轴应落在 −y（R_x(90°)·ẑ）。
    func testVerticalDragIsAppliedAboutWorldAxisRegardlessOfStage() {
        let expected = SIMD3<Float>(0, -1, 0)
        for stage in [0.0, 90.0, 217.0, 359.0] {
            let s = Float(stage) * deg
            let tumbled = PolarGemPose.tilt(PolarGemPose.identity,
                                            worldAxis: SIMD3<Float>(1, 0, 0),
                                            specimenRad: s, deltaRad: 90 * deg)
            let landed = gemAxisWorld(tumbled, specimenDeg: stage)
            XCTAssertEqual(simd_length(landed - expected), 0, accuracy: 1e-5,
                           "台角 \(stage)° 时竖拖 90° 落到了 \(landed)（应为 \(expected)）—— "
                           + "倾转轴没有搬进台面系，拖拽方向会随台角旋转")
        }

        // 反面：台角 90° 时少搬一次轴会落到 x 方向（与正解相差 90°）。
        // 台角 0 时两者恰好相同，所以反面断言必须在非零台角上做。
        let wrong = unTransportedTilt(PolarGemPose.identity,
                                      worldAxis: SIMD3<Float>(1, 0, 0), deltaRad: 90 * deg)
        let wrongLanded = gemAxisWorld(wrong, specimenDeg: 90)
        XCTAssertGreaterThan(simd_length(wrongLanded - expected), 0.5,
                             "不搬轴的错实现竟然也落在 \(wrongLanded) —— 反面断言失效，"
                             + "说明这条用例根本没在验「搬轴」")
    }

    /// 台角只绕**世界 z** 转：改台角后宝石轴应恰好是原方向绕 z 旋转台角，
    /// 倾角（相对台面法线的夹角）不变。
    func testStageTurnRotatesGemAboutWorldZOnly() {
        let tumble = PolarGemPose.from(tiltDeg: 30, azimuthDeg: 40)
        let base = gemAxisWorld(tumble, specimenDeg: 0)
        XCTAssertEqual(PolarGemPose.tiltDeg(tumble), 30, accuracy: 1e-4,
                       "夹具姿态的倾角读数应为 30°")
        for stage in [45.0, 180.0, 300.0] {
            let s = Float(stage) * deg
            let turned = gemAxisWorld(tumble, specimenDeg: stage)
            XCTAssertEqual(simd_length(turned - PolarGemPose.rotateZ(base, by: s)), 0,
                           accuracy: 1e-5,
                           "台角 \(stage)° 后宝石轴不是原方向绕世界 z 的旋转 —— "
                           + "台角被写进了姿态内层（宝石会绕自己那根被扳歪的轴公转）")
        }
    }

    /// 两个坐标系互为逆变换（光线进去、环境查询出来必须回到原方向）
    func testFrameRoundTrip() {
        let tumble = PolarGemPose.from(tiltDeg: 63, azimuthDeg: 121)
        for v in [SIMD3<Float>(1, 0, 0), SIMD3<Float>(0.3, -0.7, 0.2),
                  SIMD3<Float>(0, 0, 1)] {
            let there = PolarGemPose.toWorld(v, tumble: tumble, specimenRad: 0.7)
            let back = PolarGemPose.toGem(there, tumble: tumble, specimenRad: 0.7)
            XCTAssertEqual(simd_length(back - v), 0, accuracy: 1e-5,
                           "toGem∘toWorld ≠ 恒等（\(v) → \(back)）")
        }
    }

    /// 自转与台角是**同一根轴上的同一件事**，几何必须等价：
    /// 台角 s、自转 θ 与台角 s+θ、自转 0 必须落到同一个世界方向。
    ///
    /// 这条是「拆成两个记账」的合法性依据 —— 几何等价，才敢只把其中一个
    /// （台角）喂给盘面刻度。反面：若把自转实现成绕**别的轴**（比如绕宝石现状的
    /// 某个轴），本用例立刻红。
    func testSpinAndStageAreGeometricallyInterchangeable() {
        let tumble = PolarGemPose.from(tiltDeg: 35, azimuthDeg: 20)
        let stage = 25.0 * deg, spin = 40.0 * deg
        let v = SIMD3<Float>(0.3, -0.7, 0.2)   // 任意方向，避免落在对称轴上
        let split = PolarGemPose.toWorld(v, tumble: tumble, specimenRad: stage + spin)
        let folded = PolarGemPose.toWorld(v, tumble: tumble, specimenRad: stage)
        XCTAssertEqual(simd_length(PolarGemPose.rotateZ(folded, by: spin) - split), 0,
                       accuracy: 1e-5,
                       "台角+自转与合成后的总方位角落到了不同方向 —— 两者不再同轴同向，"
                       + "「拆开记账」的前提就没了")
    }

    /// 竖直拖搬轴用的是**总方位角**（台角 + 自转），不是纯台角。
    ///
    /// 场景：先水平拖把宝石拨到自转 40°（台角仍 0），再竖拖 90°。
    /// 正确实现（减总方位角）的世界落点 = R_x(90°)·[Rz(40°)·T·ẑ] ——
    /// 就是「在世界系里绕 x 轴翻 90°」，与手指动作直接对应。
    /// 反面（只减台角 0 = 不搬轴）：落成 Rz(40°)·R_x(90°)·T·ẑ，等于把自转当成
    /// 绕一根歪轴的公转 —— 症状是「先横拨再竖翻，翻面方向不跟手」。
    func testVerticalDragTransportsAxisByTotalAzimuth() {
        let tumble = PolarGemPose.from(tiltDeg: 20, azimuthDeg: 10)
        let spin = 40.0 * deg
        let axis = SIMD3<Float>(1, 0, 0)

        let good = PolarGemPose.tilt(tumble, worldAxis: axis,
                                     specimenRad: spin, deltaRad: 90 * deg)
        let goodWorld = PolarGemPose.toWorld(PolarGemPose.stageAxis, tumble: good,
                                             specimenRad: spin)
        let expected = simd_quatf(angle: 90 * deg, axis: axis)
            .act(PolarGemPose.toWorld(PolarGemPose.stageAxis, tumble: tumble,
                                      specimenRad: spin))
        XCTAssertEqual(simd_length(goodWorld - expected), 0, accuracy: 1e-5,
                       "竖拖落点 \(goodWorld) ≠ 世界系绕 x 翻 90° 的 \(expected) —— "
                       + "搬轴没有用总方位角（台角 + 自转）")

        // 反面：只减台角（此处台角 0，等于完全不搬轴）
        let bad = PolarGemPose.tilt(tumble, worldAxis: axis,
                                    specimenRad: 0, deltaRad: 90 * deg)
        let badWorld = PolarGemPose.toWorld(PolarGemPose.stageAxis, tumble: bad,
                                            specimenRad: spin)
        XCTAssertGreaterThan(simd_length(badWorld - expected), 0.3,
                             "只减台角的错实现竟然也落在 \(badWorld)（期望偏离 \(expected)）"
                             + " —— 反面断言失效，说明这条用例没在验「搬轴用总方位角」")
    }

    /// 台角推进是取模加法，归一到 [0,360)。
    func testStageAdvanceWrapsWithinFullTurn() {
        XCTAssertEqual(PolarGemPose.advanceStage(350, by: 20), 10, accuracy: 1e-9)
        XCTAssertEqual(PolarGemPose.advanceStage(0, by: -30), 330, accuracy: 1e-9)
        XCTAssertEqual(PolarGemPose.advanceStage(0, by: 720), 0, accuracy: 1e-9)
    }

    /// 水平拖（绕世界 y）与竖拖（绕世界 x）同一条纪律：**世界系落点与台角无关**。
    ///
    /// 沿世界 y 轴倾转 90° 后，宝石轴应落在 +x（R_y(90°)·ẑ）。
    /// 反面与竖拖用例同构：台角 90° 时少搬一次轴会落到别的方向。
    func testHorizontalDragIsAppliedAboutWorldYRegardlessOfStage() {
        let expected = SIMD3<Float>(1, 0, 0)
        for stage in [0.0, 90.0, 217.0, 359.0] {
            let s = Float(stage) * deg
            let tumbled = PolarGemPose.tilt(PolarGemPose.identity,
                                            worldAxis: PolarGemPose.tiltAxisY,
                                            specimenRad: s, deltaRad: 90 * deg)
            let landed = gemAxisWorld(tumbled, specimenDeg: stage)
            XCTAssertEqual(simd_length(landed - expected), 0, accuracy: 1e-5,
                           "台角 \(stage)° 时横拖 90° 落到了 \(landed)（应为 \(expected)）—— "
                           + "y 倾转轴没有搬进台面系，拖拽方向会随台角旋转")
        }

        // 反面：台角 90° 时少搬一次轴，落点必须偏离正解。
        // 台角 0 时两者恰好相同，所以反面断言必须在非零台角上做。
        let wrong = unTransportedTilt(PolarGemPose.identity,
                                      worldAxis: PolarGemPose.tiltAxisY, deltaRad: 90 * deg)
        let wrongLanded = gemAxisWorld(wrong, specimenDeg: 90)
        XCTAssertGreaterThan(simd_length(wrongLanded - expected), 0.5,
                             "不搬轴的错实现竟然也落在 \(wrongLanded) —— 反面断言失效，"
                             + "说明这条用例根本没在验「搬轴」")
    }

    /// 倾转分量读数必须把两根轴分开：纯 x 拖拽只动 tiltX、纯 y 拖拽只动 tiltY。
    ///
    /// 这是 UI 测试「横拖只动 tiltY、竖拖只动 tiltX」断言的数学前提 ——
    /// 分量读数本身串了（比如两轴用同一个叉积方向），UI 用例的横竖分离断言
    /// 就会在两个恒真式上空转。
    func testTiltComponentReadoutSplitsAxes() {
        let x30 = simd_quatf(angle: 30 * deg, axis: PolarGemPose.tiltAxisX)
        XCTAssertEqual(PolarGemPose.tiltComponentDeg(x30, worldAxis: PolarGemPose.tiltAxisX),
                       30, accuracy: 1e-4, "纯 x 倾转的 x 分量应为 30°")
        XCTAssertEqual(PolarGemPose.tiltComponentDeg(x30, worldAxis: PolarGemPose.tiltAxisY),
                       0, accuracy: 1e-4, "纯 x 倾转不该被读出 y 分量")

        let y40 = simd_quatf(angle: 40 * deg, axis: PolarGemPose.tiltAxisY)
        XCTAssertEqual(PolarGemPose.tiltComponentDeg(y40, worldAxis: PolarGemPose.tiltAxisY),
                       40, accuracy: 1e-4, "纯 y 倾转的 y 分量应为 40°")
        XCTAssertEqual(PolarGemPose.tiltComponentDeg(y40, worldAxis: PolarGemPose.tiltAxisX),
                       0, accuracy: 1e-4, "纯 y 倾转不该被读出 x 分量")
    }

    /// 倾角读数在整周翻转上单调覆盖 [0,180]：翻面找光轴出露点时不能卡在 90°。
    /// 反面：若读数写成「只取点积的 acos 而不取绝对值/不夹紧」，超过 90° 会回折。
    func testTiltReadoutCoversFullFlip() {
        for tilt in [0.0, 30.0, 90.0, 120.0, 179.0] {
            let tumble = PolarGemPose.from(tiltDeg: tilt, azimuthDeg: 0)
            XCTAssertEqual(PolarGemPose.tiltDeg(tumble), tilt, accuracy: 1e-3,
                           "倾角读数应能读到 \(tilt)°（翻过 90° 后不该回折）")
        }
    }

    /// 回正：光轴中线最短弧转到台面法线。正向 + 两个退化方向。
    func testAligningToStageAxis() {
        // 斜方向：T·v̂ = ẑ 且 v̂ 离 ẑ 越近转角越小（最短弧）
        let v = simd_normalize(SIMD3<Float>(0.3, -0.4, 0.6))
        let t = PolarGemPose.aligningToStageAxis(v)
        let landed = t.act(simd_normalize(v))
        XCTAssertEqual(simd_length(landed - SIMD3<Float>(0, 0, 1)), 0, accuracy: 1e-5,
                       "回正后光轴中线必须落在台面法线上")
        // 已对准 → identity；反平行 → 绕 x 翻 180°
        XCTAssertEqual(simd_length(PolarGemPose.aligningToStageAxis(
            SIMD3<Float>(0, 0, 1)).vector - SIMD4<Float>(0, 0, 0, 1)), 0, accuracy: 1e-6)
        let flip = PolarGemPose.aligningToStageAxis(SIMD3<Float>(0, 0, -1))
        XCTAssertEqual(PolarGemPose.tiltDeg(flip), 180, accuracy: 1e-4,
                       "反平行方向回正后倾角读数应为 180°")
    }

    // MARK: Uniform 接线（姿态值必须真的送到 shader）

    private func polarState(_ tumble: simd_quatf) -> OpticsPolarRenderState {
        OpticsPolarRenderState(
            mode: 1, analyzerAngleRad: .pi / 2, polarizerAngleRad: 0,
            midPolarizerAngleRad: 0, midPolarizerOn: false, stageRad: 0,
            plateThicknessMm: 1.2, crystalKind: 1, nO: 1.77, nE: 1.76,
            nA: 0, nB: 0, nG: 0, halfVrad: 0, sodium: false,
            fovHalfAngleRad: 0, rotationSamples: [],
            axisTiltRad: 0, axisAzimuthRad: 0, axisRollRad: 0,
            gemTumble: tumble.vector)
    }

    private func renderState(polar: OpticsPolarRenderState?) -> OpticsRenderState {
        OpticsRenderState(
            planes: [], ior: 1.77, dispersion: 0.018,
            absorptionAxisX: .zero, absorptionAxisY: .zero, absorptionAxisZ: .zero,
            spectrumAxes: nil, illuminant: 0, environment: 0,
            background: SIMD3<Float>(0.94, 0.95, 0.95),
            exposure: 0, environmentRotation: 18, maxBounces: 6,
            camera: OrbitCamera(), polar: polar)
    }

    func testMakeUniformsCarriesGemTumble() {
        let tumble = PolarGemPose.from(tiltDeg: 25, azimuthDeg: 0)
        let u = OpticsRenderer.makeUniforms(state: renderState(polar: polarState(tumble)),
                                            width: 64, height: 64)
        XCTAssertEqual(simd_length(u.gemTumble - tumble.vector), 0, accuracy: 1e-6,
                       "宝石姿态没有进 uniforms，shader 只能按正立算")
    }

    /// 反面配对：非偏光模式必须写**单位四元数**而不是 0。
    /// 全 0 会让 shader 的 quatRotate 把宝石整个塌进原点（画面全黑），
    /// 而且不报任何错 —— 所以这条要单独钉住。
    func testNonPolarModeWritesIdentityNotZero() {
        let u = OpticsRenderer.makeUniforms(state: renderState(polar: nil),
                                            width: 64, height: 64)
        XCTAssertEqual(u.gemTumble, SIMD4<Float>(0, 0, 0, 1),
                       "非偏光模式的 gemTumble 必须是单位四元数（写 0 会让宝石塌成一点）")
    }

    /// 宝石自转角必须真的进 `polarLab.w` —— 漏写时水平拖后宝石几何转了、
    /// 而光轴紫虚线与锥光图停在原方位（画面「像在转」，标记却脱钩，不报错）。
    func testMakeUniformsCarriesGemSpin() {
        var polar = polarState(PolarGemPose.identity)
        polar.stageRad = 0.4
        polar.gemSpinRad = 0.7
        let u = OpticsRenderer.makeUniforms(state: renderState(polar: polar),
                                            width: 64, height: 64)
        XCTAssertEqual(u.polarLab.w, 0.7, accuracy: 1e-6,
                       "宝石自转角没有进 polarLab.w，shader 的 specimenSpin 会漏掉自转")
        // 反面配对：台角与自转是两个槽，别写串
        XCTAssertEqual(u.polarParams.z, 0.4, accuracy: 1e-6,
                       "载物台角应仍走 polarParams.z（盘面刻度只认它）；"
                       + "写串了两者，转宝石就会带着台面一起转")
        // 非偏光模式必须为 0：不写会让背景/盘面拿到脏角
        let uNil = OpticsRenderer.makeUniforms(state: renderState(polar: nil),
                                               width: 64, height: 64)
        XCTAssertEqual(uNil.polarLab.w, 0, accuracy: 1e-9,
                       "非偏光模式的宝石自转角应为 0")
    }
}

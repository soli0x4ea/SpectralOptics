import simd

// MARK: - 偏光实验室 · 正交偏光 3D 的宝石姿态
//
// 存在理由：这套姿态约定必须同时在**四个地方**成立 ——
//   ① 拖拽时的姿态推进（本文件 tilt）
//   ② shader 世界系 ⇄ 宝石静止系的变换（optics_lab.metal 的 GemPose）
//   ③ 光轴紫虚线的胶囊方向（宿主的 buildOpticAxisCapsules）
//   ④ 单测
// 四处各写一遍 sin/cos 一定会分叉，而分叉的外貌是「光轴标记比晶体慢半拍」
// 这种没有任何编译或断言看得见的偏移。所以只此一份，且与 shader 逐字同式。

/// 宝石姿态分解（真值来源，改这里前先读）：
///
///     R(世界←宝石) = Rz(specimenRad) · T ,  specimenRad = stageRad + spinRad
///
/// - `T` = gemTumble：宝石**相对台面系**的姿态（单位四元数，identity = 正立在台上）。
///   拖拽只写 T：竖拖绕世界 x、横拖绕世界 y（两个都是台面内的倾转，
///   见 `tiltAxisX` / `tiltAxisY`）——不再有「水平拖 = 绕台面法线自转」。
/// - `specimenRad`：样品（宝石）绕**世界 z** 的总方位角 —— 由两个记账相加：
///     - `stageRad` **载物台角**：仪器的读数。滑杆控制；盘面刻度、金指针、
///       角标「台 x°」都跟它转；
///     - `spinRad` **宝石自转**：拖拽不写它（横拖已改为绕世界 y 倾转）。
///       这个量只剩 DEBUG 截图夹具会写，为的是能单独复现「宝石相对台面
///       转了一个方位角」的画面。
///   两者同轴同向、几何上可交换，所以宝石的世界系姿态只取决于它们的和；
///   拆成两个状态**纯粹是记账**。
///
/// 方位角在外层的理由：真台子是宝石随台公转，不是先扳歪宝石、再绕一根已经被
/// 扳歪的轴转台。若写成 Rz 在里层，转台时宝石的公转轴会变成倾斜后的「台面法线」——
/// 表现为「转了台，宝石翻着往外飘」，而读数、刻度等读出 UI 全都还是对的，
/// 只有画面不对。
public enum PolarGemPose {

    /// 样品/载物台的转轴 = 台面法线（世界系 z）
    public static let stageAxis = SIMD3<Float>(0, 0, 1)

    /// 竖直拖拽的倾转轴 = 世界 x 轴（沿台面法线顶视时，它就是屏幕右方）。
    ///
    /// 与 `stageAxis` 一样，是**物体姿态用的世界常量**，绝不取 `camera.frame.right`
    /// 那类相机相关量：物体姿态一旦挂在相机上，就随观察方式改变
    /// （「盘绕视线转」的坑，详见 optics_lab.metal 的 polarStageDisc 头注释）。
    /// 相机此刻锁在 yaw=0 顶视位，两者恰好相等 —— 正因如此，写错也看不出来。
    public static let tiltAxisX = SIMD3<Float>(1, 0, 0)

    /// 水平拖拽的倾转轴 = 世界 y 轴（沿台面法线顶视时，它就是屏幕上方）。
    ///
    /// 横拖从「绕台面法线自转」改成「绕屏幕竖直轴倾转」—— 两个拖拽
    /// 方向现在是一对互相垂直的台面内倾转轴（x/y），合起来是标准 trackball。
    /// 同样是物体姿态用的世界常量，不取相机量（理由同 `tiltAxisX`）。
    public static let tiltAxisY = SIMD3<Float>(0, 1, 0)

    /// 正立在台上（无倾转）
    public static let identity = simd_quatf(vector: SIMD4<Float>(0, 0, 0, 1))

    // MARK: 世界系 ⇄ 宝石静止系

    /// 绕世界 z 右手旋转（+x → +y）
    public static func rotateZ(_ v: SIMD3<Float>, by angle: Float) -> SIMD3<Float> {
        let c = cos(angle), s = sin(angle)
        return SIMD3<Float>(c * v.x - s * v.y, s * v.x + c * v.y, v.z)
    }

    /// 世界系方向 → 宝石静止系：先退样品方位角，再取倾转的逆。
    /// 光线、偏光片方位、环境查询都要走这里，否则会出现
    /// 「光率体转了、网格没转」的割裂（几何在宝石系、晶体光学在世界系）。
    ///
    /// `specimenRad` 传的是**总方位角**（台角 + 自转），不是纯台角 ——
    /// 少算自转时，水平拖过宝石之后光轴紫虚线会停在原方位（标记脱宝石）。
    public static func toGem(_ v: SIMD3<Float>, tumble: simd_quatf, specimenRad: Float) -> SIMD3<Float> {
        tumble.inverse.act(rotateZ(v, by: -specimenRad))
    }

    /// 宝石静止系方向 → 世界系（toGem 的逆）
    public static func toWorld(_ v: SIMD3<Float>, tumble: simd_quatf, specimenRad: Float) -> SIMD3<Float> {
        rotateZ(tumble.act(v), by: specimenRad)
    }

    // MARK: 拖拽推进

    /// 竖直拖拽：把宝石绕**世界系**的 `worldAxis`（相机右轴）倾转 `deltaRad`。
    ///
    /// 推导（为什么不能直接 `R·T`）：
    /// 手感要求这次倾转发生在世界系里 —— 手指往下推，宝石北极就朝屏幕下方倒，
    /// 与当前台角、自转角都无关。但状态量 `T` 是**台面系**的，两者差一个
    /// Rz(specimenRad = 台角 + 自转)：
    ///
    ///     Rz(s)·T_new = R_axis · Rz(s)·T_old
    ///  ⟹  T_new = R_{Rz(−s)·axis} · T_old
    ///
    /// 即「把世界轴搬进台面系再左乘」。省掉搬轴这一步（拿世界轴直接左乘 T）时，
    /// 拖拽方向会随台角旋转：台 0° 是竖直翻面，台 90° 变成绕屏幕横向翻。
    /// 它不报错、不崩，只在某一个台角下突然不跟手。
    ///
    /// `specimenRad` 必须是**总方位角**：宝石的现方位由台角与自转共同决定，
    /// 只减台角的话，水平拖过之后再做竖直拖，翻面方向会歪掉一个自转角。
    public static func tilt(_ tumble: simd_quatf, worldAxis: SIMD3<Float>,
                            specimenRad: Float, deltaRad: Float) -> simd_quatf {
        guard abs(deltaRad) > 1e-9 else { return tumble }
        let axis = rotateZ(worldAxis, by: -specimenRad)
        let len = simd_length(axis)
        guard len > 1e-6 else { return tumble }
        return (simd_quatf(angle: deltaRad, axis: axis / len) * tumble).normalized
    }

    /// 载物台角推进（取模 360）。只由面板滑杆调用 —— 拖拽不走这里。
    public static func advanceStage(_ stageDeg: Double, by deltaDeg: Double) -> Double {
        let v = (stageDeg + deltaDeg).truncatingRemainder(dividingBy: 360)
        return v < 0 ? v + 360 : v
    }

    // MARK: 读数

    /// 宝石轴（晶体自身 z）相对台面法线的夹角（度）—— 角标读数。
    ///
    /// 由四元数**导出**，不另存一份角度状态：存两份迟早分叉，而这类分叉的症状
    /// 是「角标写着 30° 但宝石看着没动」，同样没有任何断言看得见。
    public static func tiltDeg(_ tumble: simd_quatf) -> Double {
        let axis = tumble.act(stageAxis)
        return Double(acos(min(1, max(-1, axis.z)))) * 180 / .pi
    }

    /// 倾转在**某一根世界轴**（`tiltAxisX` / `tiltAxisY`）上的分量（度）——
    /// DEBUG 姿态读数用，让 UI 测试能把「横拖动了 y、竖拖动了 x」分开咬住。
    ///
    /// 推导：绕单位轴 a（a ⊥ ẑ）转 θ 把 ẑ 映到 ẑ·cosθ + (a×ẑ)·sinθ，故
    /// θ = atan2(v·(a×ẑ), v·ẑ)（v = 宝石轴的世界方向）。纯 x 拖拽只动 x 分量、
    /// 纯 y 拖拽只动 y 分量；复合姿态下两分量是欧拉式分解（不唯一），但
    /// 「单方向拖拽只动自己的分量」这条测试要的性质严格成立。
    public static func tiltComponentDeg(_ tumble: simd_quatf, worldAxis: SIMD3<Float>) -> Double {
        let v = tumble.act(stageAxis)
        let perp = simd_cross(worldAxis, stageAxis)
        return Double(atan2(simd_dot(v, perp), simd_dot(v, stageAxis))) * 180 / .pi
    }

    /// 造一个确定的姿态（DEBUG 截图夹具与单测用）：先绕台面 x 轴倾 `tiltDeg`，
    /// 再绕台面法线转 `azimuthDeg`。**与拖拽路径不同**（拖拽按世界轴增量推进），
    /// 这里只为了在夹具里复现一个已知姿态，不参与手感。
    public static func from(tiltDeg: Double, azimuthDeg: Double) -> simd_quatf {
        let t = Float(tiltDeg) * .pi / 180
        let a = Float(azimuthDeg) * .pi / 180
        return (simd_quatf(angle: a, axis: SIMD3<Float>(0, 0, 1))
                * simd_quatf(angle: t, axis: SIMD3<Float>(1, 0, 0))).normalized
    }

    // MARK: 回正（光轴中线 → 物镜轴）

    /// 造一个把宝石静止系方向 `v` 转到台面法线（世界 +z）的**最短弧**姿态：
    /// 返回的 T 满足 T·v̂ = ẑ。双击回正用——回正语义是「光轴中线对准物镜轴，
    /// 锥光图直接居中可观察」，不是「宝石正立」（正立是 identity 的旧语义，
    /// 只对 Bxa 恰好 ∥ 晶体 z 的品种成立）。
    ///
    /// 取最短弧的理由：把 Bxa 竖起来有无限多种姿态（绕 Bxa 的滚转自由），最短弧
    /// 让网格只做必要的翻转——回正后的宝石方位最接近回正前，且光轴面（xp）的
    /// 方位由这一旋转顺带确定（绕物镜轴的残余方位转台即可观察，无需指定）。
    ///
    /// 退化：v̂ ≈ +ẑ（已对准）→ identity；v̂ ≈ −ẑ（反平行）→ 绕 x 翻 180°
    ///（旋转轴不唯一，任取一根台面内轴；±Bxa 是同一条轴线，翻面后锥光等价）。
    public static func aligningToStageAxis(_ v: SIMD3<Float>) -> simd_quatf {
        let len = simd_length(v)
        guard len > 1e-6 else { return identity }
        let u = v / len
        let d = simd_dot(u, stageAxis)
        if d > 0.9999 { return identity }
        if d < -0.9999 {
            return simd_quatf(angle: .pi, axis: tiltAxisX).normalized
        }
        let axis = simd_cross(u, stageAxis)
        return simd_quatf(angle: acos(min(1, max(-1, d))), axis: simd_normalize(axis)).normalized
    }
}

import Foundation
import simd
import OpticsCore

/// Swift 的 SIMD4 只有 1/2 分量 swizzle，没有 Metal 的 `.xyz` ——
/// 用这个扩展对齐着色器写法，省得到处手写 SIMD3(x, y, z)。
/// fileprivate 限定在本文件：它只是移植辅助，不进包的 API 面。
private extension SIMD4 where Scalar == Float {
    var xyz: SIMD3<Float> { SIMD3(x, y, z) }
}

// MARK: - 光学引擎普通模式 · CPU 参考实现
//
//  optics_lab.metal 普通模式（非偏光）数学的逐行 CPU 镜像。为什么存在：
//  Metal 着色器没有断言可言，「符号错一位 / 步长用错 / Y 轴方向反了」这类错误
//  从不报错，只在画面某个角落现形。本文件是同一份数学的第二实现，
//  OpticsDiffHarnessTests 把两边的 8-bit 输出逐像素比对（阈值 8/255）。
//
//  同步纪律：修改 optics_lab.metal 的普通模式数学（traceGemSpectral /
//  environmentRadiance / sceneBackground / fragment 主流程）时必须同步改这里。
//  两份不同步时 harness 会立刻爆红 —— 这正是它存在的意义，不是麻烦。
//
//  kCMF / kXYZToRGB 与 Metal 侧常量同源（构建脚本同一张表），此处有意重复：
//  着色器常量无法被 Swift 侧 import，复制是唯一选择；改表要两边一起改。
//
//  范围：仅普通模式。偏光路径（Jones 锥光 / 密度矩阵）尚未覆盖。
enum OpticsCPUReference {

    static let epsilon: Float = 0.0012
    static let maxBouncesCap = 8
    static let specBins = 31

    // MARK: 常量表（与 optics_lab.metal 同源）

    static let kCMF: [SIMD3<Float>] = [
        SIMD3(0.014310, 0.000396, 0.067850), SIMD3(0.043510, 0.001210, 0.207400),
        SIMD3(0.134380, 0.004000, 0.645600), SIMD3(0.283900, 0.011600, 1.385600),
        SIMD3(0.348280, 0.023000, 1.747060), SIMD3(0.336200, 0.038000, 1.772110),
        SIMD3(0.290800, 0.060000, 1.669200), SIMD3(0.195360, 0.090980, 1.287640),
        SIMD3(0.095640, 0.139020, 0.812950), SIMD3(0.032010, 0.208020, 0.465180),
        SIMD3(0.004900, 0.323000, 0.272000), SIMD3(0.009300, 0.503000, 0.158200),
        SIMD3(0.063270, 0.710000, 0.078250), SIMD3(0.165500, 0.862000, 0.042160),
        SIMD3(0.290400, 0.954000, 0.020300), SIMD3(0.433450, 0.994950, 0.008750),
        SIMD3(0.594500, 0.995000, 0.003900), SIMD3(0.762100, 0.952000, 0.002100),
        SIMD3(0.916300, 0.870000, 0.001650), SIMD3(1.026300, 0.757000, 0.001100),
        SIMD3(1.062200, 0.631000, 0.000800), SIMD3(1.002600, 0.503000, 0.000340),
        SIMD3(0.854450, 0.381000, 0.000190), SIMD3(0.642400, 0.265000, 0.000050),
        SIMD3(0.447900, 0.175000, 0.000020), SIMD3(0.283500, 0.107000, 0.000000),
        SIMD3(0.164900, 0.061000, 0.000000), SIMD3(0.087400, 0.032000, 0.000000),
        SIMD3(0.046770, 0.017000, 0.000000), SIMD3(0.022700, 0.008210, 0.000000),
        SIMD3(0.011359, 0.004102, 0.000000)
    ]

    // XYZ(D65) → linear sRGB（与 Metal 侧同表）。
    // 两边都是「列构造」：三个参数各是矩阵的一列（即标准 sRGB 矩阵的一行被转置存入），
    // Metal 的 float3x3 与 Swift 的 simd_float3x3 乘法语义一致，M*v 可直接对齐。
    // 显式标注 simd_float3x3：simd_matrix 的重载推导会落到 double3x3。
    static let kXYZToRGB = simd_float3x3(
        SIMD3<Float>( 3.24096994, -0.96924364,  0.05563008),
        SIMD3<Float>(-1.53738318,  1.87596750, -0.20397696),
        SIMD3<Float>(-0.49861076,  0.04155506,  1.05697151)
    )

    // MARK: 场景输入

    /// 与 GPU 侧 buffer(1..3) 逐字节同源的输入。uniforms 必须来自
    /// OpticsRenderer.makeUniforms（唯一组装点），planes 已过 resolvedGeometry，
    /// spectral 与 spectralBuffer 的 64 个 SIMD4 行一致。
    struct Scene {
        var uniforms: OpticsUniforms
        var planes: [SIMD4<Float>]
        var spectral: [SIMD4<Float>]
    }

    // MARK: 入口

    /// 单像素着色，返回最终显示色（gamma 后，与 GPU 8-bit 输出同一定义域）。
    static func shade(_ scene: Scene, uv: SIMD2<Float>) -> SIMD3<Float> {
        let u = scene.uniforms
        precondition(Int(u.polarParams.x.rounded()) == 0,
                     "diff harness 只覆盖普通模式；偏光路径尚未有 CPU 参考")
        // 与 fragment 相同的焦平面映射（含宽高比、焦点偏移与平移）
        var centered = (uv - SIMD2<Float>(0.5, 0.5)) * 2.0
        centered.x *= u.resolution.x / max(u.resolution.y, 1)
        centered.x += u.cameraUpFocal.w   // focalOffset
        centered -= SIMD2<Float>(u.panParams.x, u.panParams.y)
        let rayOrigin = u.cameraPositionScale.xyz
        let rayDirection = simd_normalize(
            u.cameraForwardExposure.xyz
            + u.cameraRightEnvRot.xyz * centered.x * u.cameraPositionScale.w
            + u.cameraUpFocal.xyz * centered.y * u.cameraPositionScale.w
        )

        let ior = u.bodyColorIor.w
        let dispersion = u.observerDispersion.w
        let redIor = max(Float(1.001), ior - dispersion * 0.48)
        let blueIor = ior + dispersion * 0.52

        // 31 bin 出射光谱；三族波长共享几何、各自一个 IOR（B/G 色散）
        // 族划分：B 400–490（10 bin）/ G 500–570（8 bin）/ R 580–700（13 bin）
        var radiance = [Float](repeating: 0, count: specBins)
        var missed = false
        traceGemSpectral(scene, origin: rayOrigin, direction: rayDirection,
                         ior: blueIor, binBegin: 0, binCount: 10,
                         radiance: &radiance, missed: &missed)
        if missed {
            return finalize(scene, uv: uv, rayOrigin: rayOrigin,
                            rayDirection: rayDirection, radiance: nil)
        }
        traceGemSpectral(scene, origin: rayOrigin, direction: rayDirection,
                         ior: ior, binBegin: 10, binCount: 8,
                         radiance: &radiance, missed: &missed)
        traceGemSpectral(scene, origin: rayOrigin, direction: rayDirection,
                         ior: redIor, binBegin: 18, binCount: 13,
                         radiance: &radiance, missed: &missed)
        return finalize(scene, uv: uv, rayOrigin: rayOrigin,
                        rayDirection: rayDirection, radiance: radiance)
    }

    // MARK: CIE 积分 + 色调映射（fragment 尾段）

    static func finalize(_ scene: Scene, uv: SIMD2<Float>,
                         rayOrigin: SIMD3<Float>, rayDirection: SIMD3<Float>,
                         radiance: [Float]?) -> SIMD3<Float> {
        let u = scene.uniforms
        let color: SIMD3<Float>
        if let radiance {
            if u.flags.w == 4 {
                // 钠光单色照明：不做色适应，按 589.3nm CIE 色度（Y 归一）
                var y: Float = 0
                for b in 0..<specBins { y += kCMF[b].y * radiance[b] }
                color = kXYZToRGB * SIMD3<Float>(1.325, 1.0, 0.0026) * y
            } else {
                // CIE 1931 三刺激积分（照明体白场归一 = 完全色适应）
                let whiteBase = illumBase(u)
                var x: Float = 0, y: Float = 0, z: Float = 0
                var xw: Float = 0, yw: Float = 0, zw: Float = 0
                for b in 0..<specBins {
                    let s = specValue(scene.spectral, whiteBase, b)
                    x += kCMF[b].x * radiance[b]
                    y += kCMF[b].y * radiance[b]
                    z += kCMF[b].z * radiance[b]
                    xw += kCMF[b].x * s
                    yw += kCMF[b].y * s
                    zw += kCMF[b].z * s
                }
                let xyz = SIMD3<Float>(x / max(xw, 1e-6),
                                       y / max(yw, 1e-6),
                                       z / max(zw, 1e-6))
                color = kXYZToRGB * xyz
            }
        } else {
            color = sceneBackground(u, uv: uv, rayOrigin: rayOrigin,
                                    rayDirection: rayDirection)
        }
        var c = simd_max(color, SIMD3<Float>(repeating: 0))
        c *= exp2f(u.cameraForwardExposure.w)
        // ACES filmic 近似
        c = simd_clamp((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14),
                       SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 1))
        // simd 模块没有 SIMD3 版 pow，逐分量 powf（gamma 指数对三通道相同）
        let gamma: Float = 1.0 / 2.2
        c = SIMD3<Float>(powf(max(c.x, 0), gamma), powf(max(c.y, 0), gamma), powf(max(c.z, 0), gamma))
        return c
    }

    // MARK: 几何（intersectConvex / nextBoundary 的半空间 + 解析球面分支）

    struct ConvexHit {
        var hit: Bool
        var nearT: Float
        var farT: Float
        var nearNormal: SIMD3<Float>
    }

    static func intersectConvex(_ scene: Scene, _ origin: SIMD3<Float>,
                                _ direction: SIMD3<Float>) -> ConvexHit {
        let u = scene.uniforms
        var result = ConvexHit(hit: false, nearT: -1e5, farT: 1e5,
                               nearNormal: SIMD3<Float>(0, 0, 1))
        if u.sphereParams.w > 0 {
            // 解析球面 |o + t·d − c|² = r²（|d| = 1）。语义与半空间版一致。
            let oc = origin - u.sphereParams.xyz
            let halfB = simd_dot(direction, oc)
            let cc = simd_dot(oc, oc) - u.sphereParams.w * u.sphereParams.w
            let disc = halfB * halfB - cc
            if disc < 0 { return result }
            let root = sqrtf(disc)
            result.nearT = -halfB - root
            result.farT = -halfB + root
            result.nearNormal = simd_normalize(
                origin + result.nearT * direction - u.sphereParams.xyz)
            result.hit = result.farT > max(result.nearT, 0)
            return result
        }
        let count = Int(u.flags.z)
        for index in 0..<count {
            let plane = scene.planes[index]
            let denominator = simd_dot(plane.xyz, direction)
            let signedDistance = plane.w - simd_dot(plane.xyz, origin)
            if abs(denominator) < 1e-6 {
                if signedDistance < 0 { return ConvexHit(hit: false, nearT: result.nearT,
                                                         farT: result.farT, nearNormal: result.nearNormal) }
                continue
            }
            let distance = signedDistance / denominator
            if denominator < 0 && distance > result.nearT {
                result.nearT = distance
                result.nearNormal = plane.xyz
            } else if denominator > 0 {
                result.farT = min(result.farT, distance)
            }
            if result.nearT > result.farT {
                return ConvexHit(hit: false, nearT: result.nearT, farT: result.farT,
                                 nearNormal: result.nearNormal)
            }
        }
        result.hit = result.farT > max(result.nearT, 0)
        return result
    }

    static func nextBoundary(_ scene: Scene, _ origin: SIMD3<Float>,
                             _ direction: SIMD3<Float>,
                             _ hitT: inout Float, _ hitNormal: inout SIMD3<Float>) -> Bool {
        let u = scene.uniforms
        hitT = 1e5
        hitNormal = SIMD3<Float>(0, 0, 1)
        if u.sphereParams.w > 0 {
            // 体内射线：解析球面的出射点恒为远根
            let oc = origin - u.sphereParams.xyz
            let halfB = simd_dot(direction, oc)
            let cc = simd_dot(oc, oc) - u.sphereParams.w * u.sphereParams.w
            let disc = halfB * halfB - cc
            if disc < 0 { return false }
            hitT = -halfB + sqrtf(disc)
            if hitT <= epsilon { return false }
            hitNormal = simd_normalize(origin + hitT * direction - u.sphereParams.xyz)
            return hitT < 1e4
        }
        let count = Int(u.flags.z)
        for index in 0..<count {
            let plane = scene.planes[index]
            let denominator = simd_dot(plane.xyz, direction)
            if denominator <= 1e-6 { continue }
            let distance = (plane.w - simd_dot(plane.xyz, origin)) / denominator
            if distance > epsilon && distance < hitT {
                hitT = distance
                hitNormal = plane.xyz
            }
        }
        return hitT < 1e4
    }

    // MARK: 光学（Fresnel / 环境 / 光谱表）

    static func dielectricFresnel(_ cosineIncident: Float, _ n1: Float, _ n2: Float) -> Float {
        let cosI = simd_clamp(abs(cosineIncident), 0, 1)
        let ratio = n1 / n2
        let sinTSquared = ratio * ratio * max(0, 1 - cosI * cosI)
        if sinTSquared >= 1 { return 1 }
        let cosT = sqrtf(max(0, 1 - sinTSquared))
        let parallel = ((n2 * cosI) - (n1 * cosT)) / max(1e-6, (n2 * cosI) + (n1 * cosT))
        let perpendicular = ((n1 * cosI) - (n2 * cosT)) / max(1e-6, (n1 * cosI) + (n2 * cosT))
        return 0.5 * (parallel * parallel + perpendicular * perpendicular)
    }

    static func rotateEnvironment(_ u: OpticsUniforms, _ direction: SIMD3<Float>) -> SIMD3<Float> {
        let angle = u.cameraRightEnvRot.w * Float.pi / 180
        let c = cosf(angle), s = sinf(angle)
        return SIMD3<Float>(c * direction.x - s * direction.y,
                            s * direction.x + c * direction.y,
                            direction.z)
    }

    static func environmentRadiance(_ scene: Scene, _ rawDirection: SIMD3<Float>) -> SIMD3<Float> {
        let u = scene.uniforms
        let worldDirection = simd_normalize(rawDirection)
        // 偏光模式：下方载物台是经下偏光片的亮漫射光源（harness 普通模式不会走到）
        if Int(u.polarParams.x.rounded()) == 1 {
            let stage = 1.0 - smoothstep(-0.75, -0.35, worldDirection.z)
            return simd_mix(SIMD3<Float>(repeating: 0.004), SIMD3<Float>(repeating: 1.0),
                            SIMD3<Float>(repeating: stage))
        }
        let direction = rotateEnvironment(u, worldDirection)
        let horizon = smoothstep(-0.6, 0.85, direction.z)
        var base = simd_mix(SIMD3<Float>(0.12, 0.14, 0.16), SIMD3<Float>(0.92, 0.91, 0.88),
                            SIMD3<Float>(repeating: horizon))
        let env = Int(u.flags.x)
        if env == 1 { base = simd_mix(SIMD3<Float>(0.035, 0.045, 0.06), SIMD3<Float>(0.82, 0.76, 0.66),
                                      SIMD3<Float>(repeating: horizon)) }
        if env == 2 { base = simd_mix(SIMD3<Float>(0.012, 0.016, 0.022), SIMD3<Float>(0.42, 0.50, 0.60),
                                      SIMD3<Float>(repeating: horizon)) }

        let leftPanel = powf(max(0, simd_dot(direction, simd_normalize(SIMD3<Float>(-0.72, -0.36, 0.58)))), 42)
        let rightPanel = powf(max(0, simd_dot(direction, simd_normalize(SIMD3<Float>(0.78, 0.18, 0.52)))), 70)
        let crownPanel = powf(max(0, simd_dot(direction, simd_normalize(SIMD3<Float>(0.08, 0.18, 0.98)))), 110)
        let warmStrip = powf(max(0, simd_dot(direction, simd_normalize(SIMD3<Float>(-0.15, 0.96, 0.22)))), 95)
        let darkCard = powf(max(0, simd_dot(direction, simd_normalize(SIMD3<Float>(0.18, -0.92, 0.34)))), 4)
        let darkCardTwo = powf(max(0, simd_dot(direction, simd_normalize(SIMD3<Float>(-0.64, 0.68, 0.35)))), 5)

        let panels = leftPanel * SIMD3<Float>(5.6, 6.0, 6.6)
            + rightPanel * SIMD3<Float>(7.2, 6.4, 5.8)
            + crownPanel * SIMD3<Float>(4.0, 4.4, 5.2)
            + warmStrip * SIMD3<Float>(3.3, 1.9, 0.8)
        let darkStrength: Float = env == 2 ? 1.2 : 0.88
        var observerCard = smoothstep(0.91, 0.985,
                                      simd_dot(worldDirection, simd_normalize(u.observerDispersion.xyz)))
        var radiance = simd_max(SIMD3<Float>(repeating: 0.006),
                                base + panels - (darkCard + darkCardTwo) * darkStrength)

        if env == 3 {
            let azimuth = atan2f(worldDirection.y, worldDirection.x)
            let eightFold = 0.5 + 0.5 * cosf(8.0 * azimuth)
            let polar = simd_dot(worldDirection, simd_normalize(u.observerDispersion.xyz))
            let scopeRing = smoothstep(0.22, 0.72, polar) * (1.0 - smoothstep(0.88, 0.97, polar))
            let scopeSegments = simd_mix(0.12, 1.0, smoothstep(0.35, 0.78, eightFold))
            let scopeWhite = SIMD3<Float>(1.55, 1.48, 1.38) * simd_mix(0.34, 1.0, scopeSegments * scopeRing)
            let scopeRed = SIMD3<Float>(1.2, 0.055, 0.11) * (1.0 - scopeSegments) * scopeRing
            radiance = SIMD3<Float>(repeating: 0.035) + scopeWhite + scopeRed
            // env==3 用更宽的观察者卡阈值重赋 observerCard（易漏：常量变了，变量还在）
            observerCard = smoothstep(0.84, 0.975, polar)
        }

        let obsFactor: Float = env == 3 ? 0.035 : 0.16
        return simd_max(SIMD3<Float>(repeating: 0.004),
                        radiance * simd_mix(1.0, obsFactor, observerCard))
    }

    /// 环境按 Rec.709 亮度折叠为标量（照明体颜色由 SPD 严格给出）
    static func environmentLuminance(_ scene: Scene, _ direction: SIMD3<Float>) -> Float {
        simd_dot(environmentRadiance(scene, direction), SIMD3<Float>(0.2126, 0.7152, 0.0722))
    }

    static func sceneBackground(_ u: OpticsUniforms, uv: SIMD2<Float>,
                                rayOrigin: SIMD3<Float>, rayDirection: SIMD3<Float>) -> SIMD3<Float> {
        let vertical = smoothstep(0, 1, uv.y)
        var color = u.backgroundAbsorption.xyz * simd_mix(0.94, 1.035, vertical)
        if rayDirection.z < -0.0001 {
            let floorT = (-1.08 - rayOrigin.z) / rayDirection.z
            if floorT > 0 {
                let floorPoint = rayOrigin + rayDirection * floorT
                // Swift 的 SIMD3 同样没有 .xy swizzle，手动取前两维
                let floorXY = SIMD2<Float>(floorPoint.x, floorPoint.y)
                let shadow = expf(-2.0 * simd_dot(floorXY, floorXY))
                color *= 1.0 - shadow * 0.105
            }
        }
        let centered = uv - SIMD2<Float>(0.5, 0.5)
        color *= 1.0 - 0.055 * simd_dot(centered, centered)
        return color
    }

    // MARK: 光谱表（行布局与 spectralBuffer 逐字节一致）

    static func illumBase(_ u: OpticsUniforms) -> Int {
        let sel = min(max(Int(u.flags.w), 0), 4)
        return sel * 8
    }

    /// 读取 table 中 base 行的第 bin 个分量（每行 4 bin）
    static func specValue(_ table: [SIMD4<Float>], _ base: Int, _ bin: Int) -> Float {
        table[base + (bin >> 2)][bin & 3]
    }

    /// 晶体主轴 j（0=X/Np, 1=Y/Nm, 2=Z/Ng）的吸收系数。轴间步长是 8 行而非 1 ——
    /// 误用 40 + j 会读到轴 0 光谱错位 40/80nm 的数据（与 Metal 侧注释同源）。
    static func specAxisValue(_ table: [SIMD4<Float>], _ j: Int, _ bin: Int) -> Float {
        specValue(table, 40 + 8 * j, bin)
    }

    // MARK: 主路径追踪（traceGemSpectral 逐行镜像）

    /// 凸多面体/解析球面单族波长的透射路径 + 最多 8 次反射反弹。
    /// 同族波长共享几何路径与 Fresnel 分裂，逐波长仅 Beer–Lambert 吸收不同；
    /// 多色性：α_eff(λ) = ½Σ_j(1−d_j²)α_j(λ)。
    static func traceGemSpectral(_ scene: Scene, origin: SIMD3<Float>, direction: SIMD3<Float>,
                                 ior: Float, binBegin: Int, binCount: Int,
                                 radiance: inout [Float], missed: inout Bool) {
        let entry = intersectConvex(scene, origin, direction)
        if !entry.hit || entry.nearT < 0 {
            missed = true
            return
        }
        let u = scene.uniforms
        let iBase = illumBase(u)
        let entryPoint = origin + direction * entry.nearT
        let entryFresnel = dielectricFresnel(simd_dot(-direction, entry.nearNormal), 1.0, ior)
        let entryLuminance = environmentLuminance(scene, simd_reflect(direction, entry.nearNormal))
        var insideDirection = refract(direction, entry.nearNormal, 1.0 / ior)
        if simd_dot(insideDirection, insideDirection) < 1e-7 {
            for b in 0..<binCount {
                let bin = binBegin + b
                radiance[bin] = entryLuminance * specValue(scene.spectral, iBase, bin) * entryFresnel
            }
            missed = false
            return
        }

        var throughput = [Float](repeating: 0, count: specBins)
        for b in 0..<binCount {
            let bin = binBegin + b
            radiance[bin] = entryLuminance * specValue(scene.spectral, iBase, bin) * entryFresnel
            throughput[bin] = 1.0 - entryFresnel
        }
        var insideOrigin = entryPoint + insideDirection * epsilon
        let maxBounces = min(Int(u.flags.y), maxBouncesCap)
        for _ in 0..<maxBounces {
            var boundaryT: Float = 0
            var boundaryNormal = SIMD3<Float>(0, 0, 1)
            if !nextBoundary(scene, insideOrigin, insideDirection, &boundaryT, &boundaryNormal) { break }
            let boundaryPoint = insideOrigin + insideDirection * boundaryT
            // 该光段方向上的逐波长有效吸收（局部系对角张量；非偏振光两偏振各半）
            let w = SIMD3<Float>(repeating: 0.5) * (SIMD3<Float>(repeating: 1.0) - insideDirection * insideDirection)
            var maxThroughput: Float = 0
            for b in 0..<binCount {
                let bin = binBegin + b
                let segmentAbsorption =
                    w.x * specAxisValue(scene.spectral, 0, bin)
                  + w.y * specAxisValue(scene.spectral, 1, bin)
                  + w.z * specAxisValue(scene.spectral, 2, bin)
                throughput[bin] *= expf(-max(segmentAbsorption, 0) * boundaryT)
                maxThroughput = max(maxThroughput, throughput[bin])
            }
            let fresnel = dielectricFresnel(simd_dot(insideDirection, boundaryNormal), ior, 1.0)
            let exitDirection = refract(insideDirection, -boundaryNormal, ior)
            if simd_dot(exitDirection, exitDirection) > 1e-7 {
                let exitLuminance = environmentLuminance(scene, exitDirection)
                for b in 0..<binCount {
                    let bin = binBegin + b
                    radiance[bin] += throughput[bin] * (1.0 - fresnel)
                                * exitLuminance * specValue(scene.spectral, iBase, bin)
                    throughput[bin] *= fresnel
                }
            } else {
                for b in 0..<binCount {
                    throughput[binBegin + b] *= fresnel
                }
            }
            if maxThroughput * fresnel < 0.002 { break }
            insideDirection = simd_reflect(insideDirection, boundaryNormal)
            insideOrigin = boundaryPoint + insideDirection * epsilon
        }
        missed = false
    }

    // MARK: 工具（GLSL 语义对齐）

    /// GLSL/MSL refract 语义：全内反射返回零向量
    static func refract(_ i: SIMD3<Float>, _ n: SIMD3<Float>, _ eta: Float) -> SIMD3<Float> {
        let d = simd_dot(n, i)
        let k = 1.0 - eta * eta * (1.0 - d * d)
        if k < 0 { return SIMD3<Float>(repeating: 0) }
        return eta * i - (eta * d + sqrtf(k)) * n
    }

    static func smoothstep(_ edge0: Float, _ edge1: Float, _ x: Float) -> Float {
        let t = simd_clamp((x - edge0) / (edge1 - edge0), 0, 1)
        return t * t * (3 - 2 * t)
    }
}

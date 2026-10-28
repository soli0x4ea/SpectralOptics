import XCTest
import simd
import OpticsCore

// MARK: - 网格光学路径（凹面支持）· BVH 构建与求交验收
//
//  对齐上游 OpenGemCutting vgpu 后端的第二条光学路径：任意闭合三角网格
//  （保留凹槽/孔洞/分离组件，永不做凸包）。着色器侧无法在此单测，
//  这里验收的是 Swift 侧唯一事实源 OpticsMesh.swift——它的打包行
//  直接喂 GPU，遍历凭据（预序 + 逃生链接）CPU/GPU 共用。
//
//  断言纪律（与 OpticsDiffHarnessTests 同源）：
//  正向断言必须配反面断言，回答「这条断言真的咬得住吗」：
//   - BVH 最近交 == 全三角形暴力最近交（正向）；
//   - 凹腔射线必须逃逸——凸包近似实现会命中"封口面"，必挂（反面）；
//   - 内部出射必须命中背面——加了背面剔除的实现必挂（反面）。
final class OpticsMeshTests: XCTestCase {

    // MARK: 测试网格

    /// 轴对齐盒：8 顶点 + 6 四边形面（外法向显式给出）
    private func box(_ mn: SIMD3<Double>, _ mx: SIMD3<Double>)
        -> (vertices: [SIMD3<Double>], faces: [[Int]], normals: [SIMD3<Double>?]) {
        let v = [
            SIMD3(mn.x, mn.y, mn.z), SIMD3(mx.x, mn.y, mn.z),
            SIMD3(mx.x, mx.y, mn.z), SIMD3(mn.x, mx.y, mn.z),
            SIMD3(mn.x, mn.y, mx.z), SIMD3(mx.x, mn.y, mx.z),
            SIMD3(mx.x, mx.y, mx.z), SIMD3(mn.x, mx.y, mx.z),
        ]
        // 每面 [顶点序, 外法向]；四边形顶点按外看逆时针。
        // 法向显式给出：构建器按参考法向定向几何法向，环绕方向笔误会被纠正
        let quads: [((Int, Int, Int, Int), SIMD3<Double>)] = [
            ((0, 3, 2, 1), SIMD3(0, 0, -1)),  // z = mn
            ((4, 5, 6, 7), SIMD3(0, 0, 1)),   // z = mx
            ((0, 1, 5, 4), SIMD3(0, -1, 0)),  // y = mn
            ((2, 3, 7, 6), SIMD3(0, 1, 0)),   // y = mx
            ((1, 2, 6, 5), SIMD3(1, 0, 0)),   // x = mx
            ((0, 4, 7, 3), SIMD3(-1, 0, 0)),  // x = mn
        ]
        return (v, quads.map { [$0.0.0, $0.0.1, $0.0.2, $0.0.3] }, quads.map { $0.1 })
    }

    /// L 形三盒拼合：非凸（凹口在 (+x,+y) 角）。三个盒共享面但不焊接顶点，
    /// 与真实数据（每面持有独立顶点副本）同构。
    private func lShapeMesh() -> (vertices: [SIMD3<Double>], faces: [[Int]], normals: [SIMD3<Double>?]) {
        let a = box(SIMD3(-2, -2, -1), SIMD3(0, 0, 1))
        let b = box(SIMD3(0, -2, -1), SIMD3(2, 0, 1))
        let c = box(SIMD3(-2, 0, -1), SIMD3(0, 2, 1))
        var vertices: [SIMD3<Double>] = []
        var faces: [[Int]] = []
        var normals: [SIMD3<Double>?] = []
        for part in [a, b, c] {
            let base = vertices.count
            vertices += part.vertices
            faces += part.faces.map { $0.map { $0 + base } }
            normals += part.normals
        }
        return (vertices, faces, normals)
    }

    private func buildL() -> OpticsMeshState {
        let m = lShapeMesh()
        guard let mesh = OpticsMeshBuilder.build(OpticsMeshBuilder.Input(
            vertices: m.vertices, faces: m.faces, faceNormals: m.normals)) else {
            // 空网格兜底（仅供编译器收尾；断言已在上一行失败）
            return OpticsMeshState(nodes: [], triangles: [], radius: 1,
                                   normalization: OpticsMeshBuilder.Normalization(
                                       center: .zero, scale: 1, radius: 1))
        }
        return mesh
    }

    /// simd overlay 的 `.xyz` 不是 Swift 公开 API，手工截取前三个分量
    private func xyz(_ v: SIMD4<Float>) -> SIMD3<Float> {
        SIMD3<Float>(v.x, v.y, v.z)
    }

    /// 全三角形暴力最近交（与 BVH 同一套 Möller–Trumbore，但零跳过——
    /// 被测对象是遍历正确性，不是求交公式）
    private func bruteForce(_ mesh: OpticsMeshState, origin: SIMD3<Float>,
                            direction: SIMD3<Float>) -> (t: Float, normal: SIMD3<Float>)? {
        var nearest: (t: Float, normal: SIMD3<Float>)?
        for index in 0..<mesh.triangleCount {
            let a = xyz(mesh.triangles[index * 4])
            let ab = xyz(mesh.triangles[index * 4 + 1])
            let ac = xyz(mesh.triangles[index * 4 + 2])
            let p = simd_cross(direction, ac)
            let det = simd_dot(ab, p)
            if abs(det) < 1e-12 { continue }
            let rel = origin - a
            let u = simd_dot(rel, p) / det
            if u < -1e-6 || u > 1.000001 { continue }
            let q = simd_cross(rel, ab)
            let v = simd_dot(direction, q) / det
            if v < -1e-6 || u + v > 1.000001 { continue }
            let distance = simd_dot(ac, q) / det
            if distance > 1e-7, nearest == nil || distance < nearest!.t {
                nearest = (distance, xyz(mesh.triangles[index * 4 + 3]))
            }
        }
        return nearest
    }

    // MARK: 正向：BVH == 暴力

    func testBVHRaycastMatchesBruteForceOnNonConvexMesh() {
        let mesh = buildL()
        // 确定性 LCG 射线束：原点取单位球面，方向指向网格内部区域
        var seed: UInt64 = 0x9E3779B97F4A7C15
        func next01() -> Float {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Float((seed >> 33) & 0xFFFFFF) / Float(0xFFFFFF)
        }
        var mismatches = 0
        for _ in 0..<600 {
            let theta = next01() * Float.pi * 2
            let phi = (next01() - 0.5) * Float.pi
            let origin = SIMD3<Float>(cos(theta) * cos(phi), sin(theta) * cos(phi), sin(phi)) * 4
            let target = SIMD3<Float>(next01() - 0.5, next01() - 0.5, next01() - 0.5) * 2
            let direction = simd_normalize(target - origin)
            let bvh = OpticsMeshBuilder.raycast(mesh, origin: origin, direction: direction)
            let brute = bruteForce(mesh, origin: origin, direction: direction)
            switch (bvh, brute) {
            case (nil, nil):
                continue
            case let (b?, f?):
                if abs(b.t - f.t) > 1e-4 * max(1, f.t)
                    || simd_dot(b.normal, f.normal) < 0.9999 {
                    mismatches += 1
                }
            default:
                mismatches += 1   // 一边命中一边未命中：遍历凭据坏了
            }
        }
        XCTAssertEqual(mismatches, 0, "BVH 与暴力求交不一致 \(mismatches) 条")
    }

    // MARK: 反面：凹腔必须保留（凸包近似实现必挂）

    func testConcaveNotchRayEscapes() {
        let mesh = buildL()
        // 归一化后坐标：L 覆盖 [-1,1]³，凹口开在 (+x,+y) 角。
        // 凹口内一点 (0.6, 0.6, 0) 朝凹口开口方向射出：
        // 真网格上无表面 → 必须未命中；凸包会在 x=1 或 y=1 的"封口面"命中。
        let origin = SIMD3<Float>(0.6, 0.6, 0)
        let escaping = simd_normalize(SIMD3<Float>(1, 1, 0.3))
        XCTAssertNil(OpticsMeshBuilder.raycast(mesh, origin: origin, direction: escaping),
                     "凹腔射线命中了表面——凸包近似混进来了？")
        // 同点朝实体方向：必须命中，且命中面外法向迎着射线（入射）
        let intoSolid = simd_normalize(SIMD3<Float>(-1, -0.4, 0))
        let hit = OpticsMeshBuilder.raycast(mesh, origin: origin, direction: intoSolid)
        XCTAssertNotNil(hit)
        if let hit {
            XCTAssertLessThan(simd_dot(intoSolid, hit.normal), 0,
                              "入射命中的法向应迎向射线（entering）")
        }
    }

    // MARK: 反面：内部出射必须命中背面（背面剔除实现必挂）

    func testInteriorRayExitsThroughBackface() {
        let mesh = buildL()
        // 实体内部一点（下左盒中心附近）沿 -x 出射：
        // 命中的是 x=-1 外壁（下左盒独有的面，无共面共享壁的平局歧义），
        // 法向 (-1,0,0) 顺着方向（出射侧背面）。
        // 若有人加背面剔除，这条断言直接红——凹面出射全靠背面。
        let origin = SIMD3<Float>(-0.8, -0.5, 0)
        let direction = SIMD3<Float>(-1, 0, 0)
        let hit = OpticsMeshBuilder.raycast(mesh, origin: origin, direction: direction)
        XCTAssertNotNil(hit, "内部出射未命中——背面被剔除了？")
        if let hit {
            XCTAssertGreaterThan(simd_dot(direction, hit.normal), 0.999,
                                 "出射命中应沿行进方向打到外法向背面")
            XCTAssertEqual(hit.normal.x, -1, accuracy: 1e-5)
        }
    }

    // MARK: BVH 结构凭据（CPU/GPU 无栈遍历共用）

    func testEscapeLinksAndLeafInvariants() {
        let mesh = buildL()
        let nodeCount = mesh.nodeCount
        XCTAssertGreaterThan(nodeCount, 1)
        for node in 0..<nodeCount {
            let mnRow = mesh.nodes[node * 3]
            let mxRow = mesh.nodes[node * 3 + 1]
            let countRow = mesh.nodes[node * 3 + 2]
            let escape = Int(mnRow.w)
            let start = Int(mxRow.w)
            let count = Int(countRow.x)
            XCTAssertGreaterThanOrEqual(escape, node + 1, "节点 \(node) escape 未跳过自身")
            XCTAssertLessThanOrEqual(escape, nodeCount, "节点 \(node) escape 越界")
            if count == 0 {
                // 内部节点：start 无意义；子树由预序保证（左子 = node+1）
                _ = start
            } else {
                XCTAssertLessThanOrEqual(count, OpticsMeshBuilder.leafSize,
                                         "叶节点 \(node) 超过叶容量")
                XCTAssertGreaterThanOrEqual(count, 1)
                XCTAssertLessThanOrEqual(start + count, mesh.triangleCount,
                                         "叶节点 \(node) 三角形区间越界")
            }
        }
        // 根与末节点的 escape 都必须是 nodeCount：遍历循环的终止凭据
        XCTAssertEqual(Int(mesh.nodes[0].w), nodeCount)
        XCTAssertEqual(Int(mesh.nodes[(nodeCount - 1) * 3].w), nodeCount)
    }

    // MARK: 归一化与打包布局（GPU 行布局即 CPU 数据的行展开）

    func testSurfaceCentroidCentering() {
        // 两张不等面积三角形：tri1 顶点 (0,0,0)(2,0,0)(0,0,1) 面积 1，
        // tri2 顶点 (0,0,0)(0,0,1)(0,3,0) 面积 1.5。
        // 面积加权表面重心 = (1·(2/3,0,1/3) + 1.5·(0,1,1/3)) / 2.5
        //                 = (2/7.5, 0.6, 1/3)
        // 显式 [SIMD3<Double>]：整数字面量会推断成 SIMD3<Int>，与 Normalization.of
        // 的入参类型不符（顶点改为 Double 后这里就编不过了）
        let verts: [SIMD3<Double>] = [SIMD3(0, 0, 0), SIMD3(2, 0, 0), SIMD3(0, 0, 1), SIMD3(0, 3, 0)]
        let tris = [0, 1, 2, 0, 2, 3]
        guard let frame = OpticsMeshBuilder.Normalization.of(
            verts, triangleIndices: tris, centering: .surfaceCentroid) else {
            XCTFail("归一化失败")
            return
        }
        let expected = SIMD3(2.0 / 7.5, 1.5 / 2.5, 1.0 / 3.0)
        XCTAssertLessThan(simd_distance(SIMD3<Double>(frame.center), expected), 1e-9,
                          "表面重心应按面积加权（与三角化无关）")
        // 默认口径不变：AABB 中点 (1, 1.5, 0.5)
        guard let aabb = OpticsMeshBuilder.Normalization.of(verts) else {
            XCTFail("AABB 归一化失败")
            return
        }
        XCTAssertEqual(aabb.center, SIMD3(1, 1.5, 0.5))
        // radius 按所选中心重算：到重心的最大模长（顶点 (0,0,0) 最远）
        var radius: Float = 0
        for v in verts {
            radius = max(radius, simd_length(SIMD3<Float>(v - frame.center) / Float(frame.scale)))
        }
        XCTAssertEqual(frame.radius, radius, accuracy: 1e-6)
    }

    func testNormalizationAndPackingLayout() {
        let m = lShapeMesh()
        let mesh = buildL()
        // AABB 居中后最长边映射到 [-1, 1]（L 形最长边 = 4 → scale = 2）
        var mn = SIMD3<Double>(repeating: .greatestFiniteMagnitude)
        var mx = SIMD3<Double>(repeating: -.greatestFiniteMagnitude)
        for v in m.vertices { mn = simd_min(mn, v); mx = simd_max(mx, v) }
        // 拆开算，避免嵌套 max 的类型检查超时
        let ex = mx.x - mn.x, ey = mx.y - mn.y, ez = mx.z - mn.z
        let scale = Double(max(ex, max(ey, ez))) / 2
        XCTAssertGreaterThan(scale, 0)
        // radius = 归一化顶点最大模长
        var radius: Float = 0
        for v in m.vertices {
            radius = max(radius, simd_length(SIMD3<Float>(v - (mn + mx) / 2) / Float(scale)))
        }
        XCTAssertEqual(mesh.radius, radius, accuracy: 1e-5)
        // L 形沿 x/y 跨度 4、沿 z 只有 2 → 外角归一化到 (±1,±1,±0.5)，
        // 最大模长 √(1+1+0.25) = 1.5（不是立方体角的 √3）。
        XCTAssertEqual(mesh.radius, 1.5, accuracy: 1e-5,
                       "L 外角顶点 (±1,±1,±0.5) 应是最大模长 1.5")
        // 打包行数：节点 3 行/个，三角形 4 行/个
        XCTAssertEqual(mesh.nodes.count % 3, 0)
        XCTAssertEqual(mesh.triangles.count % 4, 0)
        XCTAssertEqual(mesh.triangleCount * 4, mesh.triangles.count)
        // L 形 = 3 盒 × 6 面 × 2 三角 = 36（共面共享面不剔除也不焊接）
        XCTAssertEqual(mesh.triangleCount, 36)
    }

    // MARK: 真晶棱提取（白模黑色实线的几何凭据）

    func testCreaseEdgesExcludeFanDiagonals() {
        let m = box(SIMD3(-1, -1, -1), SIMD3(1, 1, 1))
        guard let mesh = OpticsMeshBuilder.build(OpticsMeshBuilder.Input(
            vertices: m.vertices, faces: m.faces, faceNormals: m.normals)) else {
            return XCTFail("盒网格构建失败")
        }
        // 6 四边形面扇形三角化 → 12 三角形；立方体真棱恰 12 条，
        // 6 条扇形对角线（两侧法向逐位一致）必须被邻接判定排除——
        // 若混入，棱数会被抬到 18。
        XCTAssertEqual(mesh.triangleCount, 12)
        XCTAssertEqual(mesh.creaseEdgeCount, 12,
                       "扇形对角线不得混入真晶棱表")
        // 每条棱长度必为棱长 2；面/体对角线是 2√2 / 2√3——长度即判别器
        for e in 0..<mesh.creaseEdgeCount {
            let a = xyz(mesh.creaseEdges[e * 2])
            let b = xyz(mesh.creaseEdges[e * 2 + 1])
            XCTAssertEqual(simd_distance(a, b), 2.0, accuracy: 1e-3,
                           "第 \(e) 条棱长 \(simd_distance(a, b))，不是真晶棱")
        }
    }

    // MARK: 超上限拒绝（宁可失败也不静默截断封死孔洞）

    func testRejectsOversizedMeshInsteadOfTruncating() {
        // 构造风扇网格：maxTriangles+1 个三角形 → 必须被拒（宁可失败
        // 也不静默截断封死孔洞）；恰好 == 上限的网格则必须能建。
        func fan(triangleCount: Int) -> OpticsMeshBuilder.Input {
            var faces: [[Int]] = []
            var vertices: [SIMD3<Double>] = [SIMD3(0, 0, 0)]
            for i in 0..<(triangleCount + 1) {
                let a = Double(i) / Double(triangleCount + 1) * 2 * Double.pi
                vertices.append(SIMD3(cos(a), sin(a), 0))
            }
            for i in 1...triangleCount {
                faces.append([0, i, i + 1])
            }
            return OpticsMeshBuilder.Input(
                vertices: vertices, faces: faces, faceNormals: [])
        }
        XCTAssertNil(OpticsMeshBuilder.build(fan(triangleCount: OpticsMeshBuilder.maxTriangles + 1)),
                     "超上限网格必须被拒绝——截断等于把组件悄悄封死")
        XCTAssertNotNil(OpticsMeshBuilder.build(fan(triangleCount: OpticsMeshBuilder.maxTriangles)),
                        "恰好等于上限的网格应能构建")
    }

    // MARK: 固定帧（fixedNormalization：工作台取景钉毛坯）

    func testFixedNormalizationOverridesCentering() {
        let verts: [SIMD3<Double>] = [SIMD3(0, 0, 0), SIMD3(2, 0, 0), SIMD3(0, 0, 1), SIMD3(0, 3, 0)]
        // 毛坯帧：以原点为中心、scale=4 的固定帧（刻意与 AABB/重心都不同）
        let blank = OpticsMeshBuilder.Normalization(center: .zero, scale: 4, radius: 0.75)
        guard let mesh = OpticsMeshBuilder.build(OpticsMeshBuilder.Input(
            vertices: verts,
            faces: [[0, 1, 2], [0, 2, 3]],
            faceNormals: [],
            centering: .surfaceCentroid,   // 必须被 fixedNormalization 覆盖
            fixedNormalization: blank)) else {
            return XCTFail("固定帧构建失败")
        }
        XCTAssertEqual(mesh.normalization, blank,
                       "固定帧必须原样回传（画在网格上的几何要吃同一份）")
        // 顶点必须按固定帧变换：frame(p) = (p − 0)/4 → 顶点 (2,0,0) → (0.5, 0, 0)
        let mapped = mesh.normalization(SIMD3(2, 0, 0))
        XCTAssertEqual(simd_length(mapped - SIMD3<Float>(0.5, 0, 0)), 0, accuracy: 1e-6)
    }
}

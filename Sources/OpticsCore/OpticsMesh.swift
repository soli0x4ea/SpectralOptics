import Foundation
import simd

// MARK: - 网格光学几何（凹面支持）
//
// 对齐上游 OpenGemCutting 2026-09-21 vgpu 后端的第二条光学路径
// （meshRaycast.js / opticsGeometry.js / opticsShader.wgsl）：
// 凸多面体继续走半空间平面表；任意闭合三角网格（保留凹槽、孔洞与
// 分离组件，**永不做凸包**）走本文件构建的 BVH 路径。
//
// Swift 侧是这条路径的唯一事实源：归一化、BVH 构建、GPU 缓冲打包、
// CPU 参考求交都在这里。optics_lab.metal 的 intersectMeshBVH /
// traceMeshSpectral 与这里逐行同构——改任何一边都要同步另一边，
// 否则 GPU 画面与 CPU 断言测的就不再是同一份几何。

/// 渲染器的网格输入（值类型；宿主侧经 OpticsRenderState.mesh 传入）
public struct OpticsMeshState {
    /// BVH 节点表：每节点 3 个 SIMD4 ——
    /// (min.xyz, escape) (max.xyz, start) (count, 0, 0, 0)。
    /// 与上游 packOpticsMeshTextures 的节点行布局逐字段一致。
    public var nodes: [SIMD4<Float>]
    /// 三角形表：每三角形 4 个 SIMD4 —— (a,0) (ab,0) (ac,0) (normal,0)
    public var triangles: [SIMD4<Float>]
    /// 归一化后顶点最大半径（framing 用，上游 mesh.radius）
    public var radius: Float
    /// 实际使用的归一化帧（外部传入的固定帧或内部 surfaceCentroid 帧）。
    /// 刀口描边等「画在网格上的几何」必须吃这一份，避免与 mesh 不同源。
    public let normalization: OpticsMeshBuilder.Normalization
    /// 真晶棱表（白模黑色实线用）：每条棱 2 行 SIMD4 —— (A.xyz,0) (B.xyz,0)。
    /// 只含「两侧面法向夹角超阈」的二面角棱与边界棱；扇形三角化的对角线
    /// 两侧法向一致，被邻接判定排除。shader 逐像素做点-线段距离测得实线。
    public var creaseEdges: [SIMD4<Float>] = []

    public init(nodes: [SIMD4<Float>],
                triangles: [SIMD4<Float>],
                radius: Float,
                normalization: OpticsMeshBuilder.Normalization,
                creaseEdges: [SIMD4<Float>] = []) {
        self.nodes = nodes
        self.triangles = triangles
        self.radius = radius
        self.normalization = normalization
        self.creaseEdges = creaseEdges
    }

    public var nodeCount: Int { nodes.count / 3 }
    public var triangleCount: Int { triangles.count / 4 }
    /// 晶棱条数（creaseEdges 每 2 行一条）
    public var creaseEdgeCount: Int { creaseEdges.count / 2 }
}

/// 相机尺度公式（网格与晶胞两条解析路径共用）：
/// 半径 r 在距离 4.4 处的张角切线，按半视场折算，1.08 为上游的出画余量。
public enum OpticsFraming {
    public static func scale(radius: Float, halfField: Float = 0.5) -> Float {
        let r = max(radius, 0.01)
        let tangent = r / sqrt(max(4.4 * 4.4 - r * r, 0.01))
        return tangent / max(halfField, 0.05) * 1.08
    }
}

/// simd overlay 的 `.xyz` 组件访问器不是 Swift 公开 API（那是 MSL 的语法），
/// 手工截取前三个分量
private func xyz(_ v: SIMD4<Float>) -> SIMD3<Float> {
    SIMD3<Float>(v.x, v.y, v.z)
}

public enum OpticsMeshBuilder {
    /// GPU 缓冲容量上限（OpticsRenderer 的 mesh 缓冲按此分配）。
    /// 叶 ≤4 时节点总数 < 2×(maxTriangles/4)，4096 节点上限有充足余量。
    public static let maxTriangles = 4096
    public static let maxNodes = 4096
    /// 晶棱表上限（OpticsRenderer 的 edge 缓冲按此分配）。
    /// 凸多面体棱数 ≤ 3×顶点数−6，全库最大晶形远低于此；1024 是数倍余量。
    public static let maxCreaseEdges = 1024
    /// 叶节点最大三角形数（与上游 buildMeshBvh 一致）
    public static let leafSize = 4

    /// 网格归一化（网格 BVH、标注层、白模光轴的唯一事实源）：
    /// 最长边映射到 [-1,1]；radius = 归一化后顶点的最大模长。
    /// 任何"画在网格上的几何"（晶轴、Miller 锚点、光轴胶囊）都必须吃这一份
    /// 变换，杜绝两处各归一化一次带来的漂移。
    public struct Normalization: Equatable {
        public let center: SIMD3<Double>
        public let scale: Double
        public let radius: Float

        public init(center: SIMD3<Double>, scale: Double, radius: Float) {
            self.center = center
            self.scale = scale
            self.radius = radius
        }

        /// 数据系点 → 归一化系
        public func callAsFunction(_ p: SIMD3<Double>) -> SIMD3<Float> {
            SIMD3<Float>(p - center) / Float(scale)
        }

        /// 中心口径。
        /// - `aabb`：包围盒中点。盒状几何（晶胞的原子∪角点）的自然中心。
        /// - `surfaceCentroid`：表面积加权重心（面元质心按面积加权，与三角化
        ///   无关）。晶形多面体的**视觉质量中心**——三方/四方/六方的单锥组合
        ///   闭合后晶体整体偏离晶体学原点，AABB 居中会让轮廓的「质量」明显
        ///   偏离载台中心。对称晶形两种口径重合，零视觉变化。
        public enum Centering: Equatable {
            case aabb
            case surfaceCentroid
        }

        /// - Parameters:
        ///   - triangleIndices: 扁平三角形索引（表面积重心用）；nil 时退化为
        ///     顶点平均。scale 恒取 AABB 最长边（取景尺寸稳定，不随口径变）。
        public static func of(_ vertices: [SIMD3<Double>],
                              triangleIndices: [Int]? = nil,
                              centering: Centering = .aabb) -> Normalization? {
            guard !vertices.isEmpty else { return nil }
            var mn = SIMD3<Double>(repeating: .greatestFiniteMagnitude)
            var mx = SIMD3<Double>(repeating: -.greatestFiniteMagnitude)
            for v in vertices { mn = simd_min(mn, v); mx = simd_max(mx, v) }
            let aabbCenter = (mn + mx) / 2
            let scale = max(mx.x - mn.x, mx.y - mn.y, mx.z - mn.z, 1e-6) / 2
            let center: SIMD3<Double>
            switch centering {
            case .aabb:
                center = aabbCenter
            case .surfaceCentroid:
                if let tri = triangleIndices, tri.count >= 3 {
                    var acc = SIMD3<Double>.zero
                    var areaSum = 0.0
                    var i = 0
                    while i + 2 < tri.count {
                        let a = vertices[tri[i]], b = vertices[tri[i + 1]], c = vertices[tri[i + 2]]
                        let area = 0.5 * simd_length(simd_cross(b - a, c - a))
                        if area.isFinite, area > 0 {
                            acc += (a + b + c) * (area / 3)
                            areaSum += area
                        }
                        i += 3
                    }
                    center = areaSum > 0 ? acc / areaSum : aabbCenter
                } else {
                    center = vertices.reduce(SIMD3<Double>.zero, +) / Double(vertices.count)
                }
            }
            var radius: Float = 0
            for v in vertices {
                radius = max(radius, simd_length(SIMD3<Float>(v - center) / Float(scale)))
            }
            return Normalization(center: center, scale: scale, radius: radius)
        }
    }

    public struct Input {
        /// 原始顶点（任意尺度，构建时归一化）
        public var vertices: [SIMD3<Double>]
        /// 扇形三角化的面（每面一串顶点索引；纯三角形网格每面 3 个）
        public var faces: [[Int]]
        /// 每面参考外法向（nil = 按索引环绕方向的几何法向）
        public var faceNormals: [SIMD3<Double>?]
        /// 归一化中心口径。晶形多面体传 .surfaceCentroid（视觉质量中心）；
        /// 晶胞等盒状几何保持默认 .aabb。
        public var centering: Normalization.Centering = .aabb
        /// 外部强制的归一化帧（工作台编辑态传**毛坯**的固定帧）。
        /// 非 nil 时忽略 centering，直接用它变换顶点、用它的 radius 取景——
        /// 切磨过程中几何体每刀都变，但取景必须钉在毛坯上（对齐 web
        /// calculateSceneScale(ghostBounds)），否则分度环会随行业角忽大忽小。
        public var fixedNormalization: Normalization? = nil

        public init(vertices: [SIMD3<Double>],
                    faces: [[Int]],
                    faceNormals: [SIMD3<Double>?],
                    centering: Normalization.Centering = .aabb,
                    fixedNormalization: Normalization? = nil) {
            self.vertices = vertices
            self.faces = faces
            self.faceNormals = faceNormals
            self.centering = centering
            self.fixedNormalization = fixedNormalization
        }
    }

    /// 便利构造：扁平三角形索引（positions/indices 直接喂入）。
    /// 面参考法向取首顶点法向（晶形网格每面顶点持有独立法线副本）。
    public static func build(vertices: [SIMD3<Double>], triangleIndices: [Int],
                             vertexNormals: [SIMD3<Double>]? = nil,
                             centering: Normalization.Centering = .aabb,
                             fixedNormalization: Normalization? = nil) -> OpticsMeshState? {
        var faces: [[Int]] = []
        var normals: [SIMD3<Double>?] = []
        var i = 0
        while i + 2 < triangleIndices.count {
            let a = triangleIndices[i]
            faces.append([a, triangleIndices[i + 1], triangleIndices[i + 2]])
            normals.append(vertexNormals.flatMap { $0.indices.contains(a) ? $0[a] : nil })
            i += 3
        }
        return build(Input(vertices: vertices, faces: faces, faceNormals: normals,
                           centering: centering, fixedNormalization: fixedNormalization))
    }

    /// 扇形面 → 扁平三角形索引（表面积重心计算用；退化三角形面积权重为 0，无碍）
    private static func fanTriangles(_ face: [Int]) -> [Int] {
        guard face.count >= 3 else { return [] }
        var out: [Int] = []
        out.reserveCapacity((face.count - 2) * 3)
        for t in 1..<(face.count - 1) {
            out.append(contentsOf: [face[0], face[t], face[t + 1]])
        }
        return out
    }

    /// 归一化 + 建树 + 打包。三角形超上限返回 nil（调用方回退凸包或报错，
    /// 绝不静默截断——截断等于把孔洞/组件悄悄封死）。
    public static func build(_ input: Input) -> OpticsMeshState? {
        // —— 归一化：Normalization 是居中/最长边→[-1,1] 的唯一事实源 ——
        // 晶形多面体用表面积重心（视觉质量中心），build(Input) 自带的
        // 三角形清单在这里即可拿到（Input.faces 扇形展开）。
        // fixedNormalization 优先：工作台把取景钉在毛坯帧上（见 Input 注释）。
        guard let frame = input.fixedNormalization ?? Normalization.of(
            input.vertices,
            triangleIndices: input.centering == .surfaceCentroid
                ? input.faces.flatMap { fanTriangles($0) }
                : nil,
            centering: input.centering) else { return nil }
        let norm = input.vertices.map { frame($0) }
        let radius = frame.radius

        // —— 三角形（上游 buildMeshBvh）：扇形三角化 + 退化剔除 + 外法向定向 ——
        struct Tri {
            var a, ab, ac, normal: SIMD3<Float>
            var mn, mx: SIMD3<Float>
        }
        var tris: [Tri] = []
        for (f, face) in input.faces.enumerated() where face.count >= 3 {
            for k in 1..<(face.count - 1) {
                let ia = face[0], ib = face[k], ic = face[k + 1]
                guard norm.indices.contains(ia), norm.indices.contains(ib),
                      norm.indices.contains(ic) else { continue }
                let a = norm[ia], b = norm[ib], c = norm[ic]
                let ab = b - a, ac = c - a
                let g = simd_cross(ab, ac)
                let len = simd_length(g)
                // 扇形三角化可能带进共线边界顶点：其微小抵消残差不是表面，
                // float32 上传会放大成任意法向的薄片并造成光学自命中。
                // 以两边长度为基准约束抵消（而非面积阈值），保住真实的小/薄三角形。
                let roundingBound = 16 * Float.ulpOfOne * simd_length(ab) * simd_length(ac)
                if len <= roundingBound { continue }
                let geometric = g / len
                // 面法向参考决定外向符号；无参考时按索引环绕方向（假定外向）
                let reference = input.faceNormals.count > f ? input.faceNormals[f] : nil
                let outward: SIMD3<Double> = reference ?? SIMD3<Double>(geometric)
                let sign: Float = simd_dot(SIMD3<Float>(outward), geometric) < 0 ? -1 : 1
                tris.append(Tri(a: a, ab: ab, ac: ac, normal: sign * geometric,
                                mn: simd_min(a, simd_min(b, c)),
                                mx: simd_max(a, simd_max(b, c))))
            }
        }
        guard !tris.isEmpty, tris.count <= maxTriangles else { return nil }

        // —— 真晶棱提取（白模黑色实线的几何凭据）——
        // 键用量化位置而非顶点索引：晶形网格相邻面各持顶点副本（每面顶点
        // 由面-面求交独立算出），索引对不上。量化 4096 步 ≈ 5e-4（归一化域），
        // 同一面内的共享顶点逐位相同必落同键。
        // 判定：一条边聚到 2 个面法向 → 二面角 > ~1.8° 才算真棱（扇形对角线
        // 两侧法向逐位一致，dot=1 被排除）；只聚到 1 个面 → 边界棱（闭合晶形
        // 理论上没有，防御性保留——宁可多画一条不存在的棱，不可漏一条真棱）。
        struct EdgeKey: Hashable {
            var lo: SIMD3<Int32>
            var hi: SIMD3<Int32>
        }
        func quantize(_ v: SIMD3<Float>) -> SIMD3<Int32> {
            SIMD3<Int32>(Int32((v.x * 4096.0).rounded()),
                         Int32((v.y * 4096.0).rounded()),
                         Int32((v.z * 4096.0).rounded()))
        }
        var edgeFaces: [EdgeKey: [SIMD3<Float>]] = [:]
        var edgeEnds: [EdgeKey: (SIMD3<Float>, SIMD3<Float>)] = [:]
        edgeFaces.reserveCapacity(tris.count * 3 / 2)
        for t in tris {
            let a = t.a, b = t.a + t.ab, c = t.a + t.ac
            for (p, q) in [(a, b), (b, c), (c, a)] {
                let kp = quantize(p), kq = quantize(q)
                let key = kp.x < kq.x || (kp.x == kq.x && (kp.y < kq.y || (kp.y == kq.y && kp.z < kq.z)))
                    ? EdgeKey(lo: kp, hi: kq) : EdgeKey(lo: kq, hi: kp)
                var faces = edgeFaces[key] ?? []
                if faces.count < 2 { faces.append(t.normal) }
                edgeFaces[key] = faces
                if edgeEnds[key] == nil { edgeEnds[key] = (p, q) }
            }
        }
        var creaseRows: [SIMD4<Float>] = []
        creaseRows.reserveCapacity(min(edgeFaces.count, maxCreaseEdges) * 2)
        for (key, faces) in edgeFaces {
            guard let ends = edgeEnds[key] else { continue }
            if faces.count == 2, simd_dot(faces[0], faces[1]) > 0.9995 { continue }
            guard creaseRows.count + 2 <= maxCreaseEdges * 2 else { break }
            // 行里存量化前的原始端点——量化只做键，几何用全精度
            creaseRows.append(SIMD4<Float>(ends.0, 0))
            creaseRows.append(SIMD4<Float>(ends.1, 0))
        }

        // —— BVH（上游 buildMeshBvh）：中位切分 + 预序逃生链接（无栈遍历的凭据）——
        struct Node {
            var mn: SIMD3<Float>
            var mx: SIMD3<Float>
            var start: Int
            var count: Int
            var escape: Int
        }
        var nodes: [Node] = []
        func partition(_ start: Int, _ end: Int) {
            let index = nodes.count
            var nmn = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
            var nmx = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
            for t in start..<end {
                nmn = simd_min(nmn, tris[t].mn); nmx = simd_max(nmx, tris[t].mx)
            }
            nodes.append(Node(mn: nmn, mx: nmx, start: start, count: end - start, escape: 0))
            if end - start > OpticsMeshBuilder.leafSize {
                let extent = nmx - nmn
                let axis = extent.x > extent.y ? (extent.x > extent.z ? 0 : 2)
                                               : (extent.y > extent.z ? 1 : 2)
                let slice = tris[start..<end].sorted {
                    $0.mn[axis] + $0.mx[axis] < $1.mn[axis] + $1.mx[axis]
                }
                for (offset, t) in slice.enumerated() { tris[start + offset] = t }
                let middle = (start + end) / 2
                nodes[index].count = 0
                partition(start, middle)
                partition(middle, end)
            }
            // 逃生链接 = 本子树（含子孙）之后第一个节点下标：
            // 无栈遍历据此整段跳过未命中的子树，CPU/GPU 两侧共用同一凭据
            nodes[index].escape = nodes.count
        }
        partition(0, tris.count)

        // —— GPU 打包（上游 packOpticsMeshTextures 的 buffer 版）——
        var nodeRows: [SIMD4<Float>] = []
        nodeRows.reserveCapacity(nodes.count * 3)
        for n in nodes {
            nodeRows.append(SIMD4<Float>(n.mn, Float(n.escape)))
            nodeRows.append(SIMD4<Float>(n.mx, Float(n.start)))
            nodeRows.append(SIMD4<Float>(Float(n.count), 0, 0, 0))
        }
        var triRows: [SIMD4<Float>] = []
        triRows.reserveCapacity(tris.count * 4)
        for t in tris {
            triRows.append(SIMD4<Float>(t.a, 0))
            triRows.append(SIMD4<Float>(t.ab, 0))
            triRows.append(SIMD4<Float>(t.ac, 0))
            triRows.append(SIMD4<Float>(t.normal, 0))
        }
        return OpticsMeshState(nodes: nodeRows, triangles: triRows, radius: radius,
                               normalization: frame, creaseEdges: creaseRows)
    }

    // MARK: CPU 参考求交（与 optics_lab.metal 的 intersectMeshBVH 逐行同构）

    /// 无栈 BVH 遍历最近命中。方向需已归一化；不做背面剔除——
    /// 凹面光学出射与空腔需要每个边界的两侧。
    /// 返回 (距离, 外法向)；未命中返回 nil。
    public static func raycast(_ mesh: OpticsMeshState, origin: SIMD3<Float>,
                               direction: SIMD3<Float>,
                               minDistance: Float = 1e-7, maxDistance: Float = 1e19)
        -> (t: Float, normal: SIMD3<Float>)? {
        var nearest = maxDistance
        var result: (t: Float, normal: SIMD3<Float>)?
        var node = 0
        let nodeCount = mesh.nodeCount
        while node < nodeCount {
            let mnRow = mesh.nodes[node * 3]
            let mxRow = mesh.nodes[node * 3 + 1]
            let countRow = mesh.nodes[node * 3 + 2]
            if !boundsOverlap(origin: origin, direction: direction,
                              mn: xyz(mnRow), mx: xyz(mxRow), limit: nearest) {
                node = Int(mnRow.w)   // 逃生链接：整段跳过未命中子树
                continue
            }
            let start = Int(mxRow.w)
            let count = Int(countRow.x)
            for index in start..<(start + count) {
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
                if distance > minDistance && distance < nearest {
                    nearest = distance
                    result = (distance, xyz(mesh.triangles[index * 4 + 3]))
                }
            }
            node += 1
        }
        return result
    }

    /// 射线-包围盒（slab 法，与 WGSL meshBounds 一致；方向分量近零走轴区间判断）
    public static func boundsOverlap(origin: SIMD3<Float>, direction: SIMD3<Float>,
                                     mn: SIMD3<Float>, mx: SIMD3<Float>, limit: Float) -> Bool {
        var nearT: Float = 0
        var farT: Float = limit
        for axis in 0..<3 {
            let o = origin[axis], d = direction[axis]
            if abs(d) < 1e-12 {
                if o < mn[axis] || o > mx[axis] { return false }
            } else {
                let a = (mn[axis] - o) / d
                let b = (mx[axis] - o) / d
                nearT = max(nearT, min(a, b))
                farT = min(farT, max(a, b))
                if nearT > farT { return false }
            }
        }
        return true
    }
}

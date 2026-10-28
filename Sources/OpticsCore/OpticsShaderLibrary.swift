import Metal

/// 着色器库加载（包内唯一入口）。
///
/// App 内嵌时着色器可用 `device.makeDefaultLibrary()` 从主 bundle 拿；
/// 作为 Swift Package 分发后 `.metal` 源码是包资源（`Bundle.module`），
/// 必须用 `makeLibrary(source:)` 运行时编译。MSL 编译发生在渲染器初始化
/// 一次（毫秒级），后续帧零成本；如需零启动开销可换成预编译 `.metallib`。
public enum OpticsShaderLibrary {
    public enum ShaderError: Error, CustomStringConvertible {
        case resourceMissing
        case compilationFailed(String)

        public var description: String {
            switch self {
            case .resourceMissing:
                return "optics_lab.metal 不在包资源中"
            case .compilationFailed(let log):
                return "MSL 编译失败: \(log)"
            }
        }
    }

    /// 从包资源读取 optics_lab.metal 并为指定设备编译。
    public static func makeLibrary(for device: MTLDevice) throws -> MTLLibrary {
        guard let url = Bundle.module.url(forResource: "optics_lab", withExtension: "metal"),
              let source = try? String(contentsOf: url, encoding: .utf8) else {
            throw ShaderError.resourceMissing
        }
        do {
            return try device.makeLibrary(source: source, options: nil)
        } catch {
            throw ShaderError.compilationFailed("\(error)")
        }
    }
}

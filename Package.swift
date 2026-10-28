// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "SpectralOptics",
    platforms: [
        .iOS(.v15),
        .macOS(.v12),
    ],
    products: [
        // 纯渲染：OpticsRenderer + 网格/BVH + 姿态数学 + 照明体 SPD（无 UIKit）
        .library(name: "OpticsCore", targets: ["OpticsCore"]),
        // 可选视口：OrbitMTKView + 轨道手势状态机（UIKit，iOS/Catalyst）
        .library(name: "OpticsOrbit", targets: ["OpticsOrbit"]),
    ],
    targets: [
        .target(
            name: "OpticsCore",
            resources: [
                .copy("Resources/optics_lab.metal")
            ]
        ),
        .target(
            name: "OpticsOrbit",
            dependencies: ["OpticsCore"]
        ),
        .testTarget(
            name: "OpticsCoreTests",
            dependencies: ["OpticsCore"],
            resources: [
                .copy("Fixtures")
            ]
        ),
    ]
)

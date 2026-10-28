//
//  optics_lab.metal
//  SpectralOptics
//
//  宝石光学路径追踪着色器。由 OpenGemCutting 的 OpticsViewport.jsx
//  FRAGMENT_SHADER（WebGL2 GLSL 300 ES，MIT，翻译源 blob
//  fff810ce4e825df209882cf5f696c86dcd6f879d @ dbc623e）逐行翻译为
//  Metal Shading Language，并将 RGB 体色吸收升级为 31 波长
//  （400–700nm/10nm）真光谱路径追踪。
//  仅保留凸多面体 planes 模式（目录内全部琢型均为凸多面体；最密的球面逼近 128 平面）。
//  来源与许可：见仓库 THIRD_PARTY_NOTICES.md 与 docs/compliance/PROVENANCE.md。
//
//  物理模型：
//   - Fresnel 反射/透射分裂、全内反射、B–G 色散（蓝/绿/红三族波长共享几何）
//   - Beer–Lambert 光谱吸收：三主轴 × 31 波长对角张量（笔记可见光谱 CIE 锚定）
//   - 真实照明体 SPD（D65 昼光 / A 2856K 白炽灯 / 烛光 1900K / E 等能白）
//   - CIE 1931 2° 标准色度观察者积分 → XYZ → linear sRGB（白场色适应归一）
//   - 4 种程序化照明环境、ACES 近似色调映射 + gamma 2.2
//
//  坐标约定与 Web 版一致：z+ 为台面/冠部方向，实体归一化到 [-1, 1]，
//  相机距离 4.4，默认焦平面系数 0.34。
//
#include <metal_stdlib>
using namespace metal;

#define MAX_BOUNCES 8
#define MAX_PLANES 224
#define EPSILON 0.0012
#define SPEC_BINS 31

/// 全部标量/向量参数以 16 字节对齐向量打包，Swift 侧逐字段对应。
struct Uniforms {
    float4 cameraPositionScale;   // xyz: 相机位置, w: cameraScale
    float4 cameraForwardExposure; // xyz: forward, w: exposure (exp2)
    float4 cameraRightEnvRot;     // xyz: right,   w: environmentRotation(deg)
    float4 cameraUpFocal;         // xyz: up,      w: focalOffset
    float4 bodyColorIor;          // xyz unused, w: ior
    float4 observerDispersion;    // xyz: observerDirection, w: dispersion
    float4 backgroundAbsorption;  // xyz: background linear rgb, w: unused
    float4 absorptionAxisX;       // （RGB 回退路径已停用，保留以维持 208 字节布局）
    float4 absorptionAxisY;
    float4 absorptionAxisZ;
    float4 panParams;             // x: panX, y: panY, z: aspect(w/h), w: unused
    int4   flags;                 // x: environment 0...3, y: maxBounces, z: planeCount, w illuminant 0...4
    float2 resolution;
    // —— 偏光实验室（208 字节之后追加；普通模式 mode=0 时全部忽略）——
    float4 polarParams;           // x: renderMode 0普通/1正交偏光3D, y: 分析镜绝对方位(rad), z: 载物台旋转(rad), w: 锥光等效板厚(mm)
    float4 crystalA;              // x: kind 0均质/1一轴+/2一轴-/3二轴+/4二轴-, y: nO(或均质n), z: nE, w unused
    float4 crystalB;              // x: nA, y: nB, z: nG, w: 2V(rad)
    float4 polarFlags;            // x: 钠光(锥光)1/白光0, y: 锥光视场半角(rad), z: 旋光谱点数, w: 真实锥光累积帧序号
    float4 crystalOrient;         // x: 光轴(二轴晶=光轴中线Bxa)倾角(rad), y: 方位角(rad), z: 光轴面滚转(rad), w unused
    // —— 解析球面（自建实验型「光滑球体」）——
    // xyz: 球心, w: 半径；w > 0 时启用解析球面求交，此时 flags.z(平面数) 必须为 0。
    // 之所以用解析求交而不是平面逼近：半空间表示每个面只有一个法向，
    // 无论细分到多少面，看到的永远是刻面球（且平面循环 O(n) 更慢）。
    float4 sphereParams;
    // —— 三偏光片（与 Swift 侧 OpticsUniforms 末尾同步追加，二进制布局必须一致）——
    // x: 下偏光片(起偏镜)绝对方位(rad)，y: 中偏光片绝对方位(rad)，
    // z: 中偏光片启用(1/0)，w: 宝石自转角(rad)。分析镜绝对方位在 polarParams.y。
    // 三个方位都是实验室系绝对角（0° = +x）。中偏光片固定插在宝石/晶片与
    // 分析镜之间——插在任一主偏光片同侧会被紧邻的同侧偏光片吞并，没有物理意义。
    // w 是「宝石自己在台上转」的记账：与 polarParams.z（载物台角）同轴同向，
    // 样品方位取两者之和（specimenSpin），**盘面只取 polarParams.z** ——
    // 台面是台上的固定参照，转宝石不该连刻度一起转。
    float4 polarLab;
    // —— 宝石姿态（正交偏光 3D 的「手指扳宝石」自由度）——
    // 宝石在**台面系**里的姿态，单位四元数 (ix, iy, iz, r)，identity = (0,0,0,1) 正立在台上。
    // 世界←宝石 = Rz(specimenSpin) · gemTumble：样品方位角在最外层绕世界 z，
    // 倾转在内层（手指翻面）。推导与失败面貌见 PolarGemPose（Swift 侧同式）。
    // ⚠️ 非偏光模式必须写单位四元数，不能写 0 —— 本字段无条件参与成像。
    float4 gemTumble;
};

/// 样品（宝石 / 锥光平行晶片）绕台面法线的**总方位角** = 载物台角 + 宝石自转。
/// 除了可见的载物台圆盘（polarStageDisc 只取 polarParams.z），所有「样品在转」
/// 的地方都走这里：宝石姿态、光率体定向、锥光干涉图。
/// 漏掉 polarLab.w 的症状是「水平拖之后宝石转了、光轴紫虚线与干涉图不转」——
/// 编译与断言都不会响。
inline float specimenSpin(constant Uniforms &u) {
    return u.polarParams.z + u.polarLab.w;
}

/// buffer(3) spectralTable：80 个 float4 = 1280 字节
//  [0..40)  5 个照明体相对光谱功率 S(λ)，各 8 行（31 bin 400–700nm/10nm，末位补 0）
//           顺序：0 D65 / 1 A 2856K / 2 烛光 1900K / 3 E 等能白 / 4 钠光 D 线 589.3nm
//  [40..48) 晶体 X 主轴单位光程吸收 α(λ)
//  [48..56) 晶体 Y 主轴
//  [56..64) 晶体 Z 主轴（台面法线）
//  照明体选择：u.flags.w（0...4）
#define SPEC_ILLUM_STRIDE 8
#define SPEC_AXIS_X_BASE 40
#define SPEC_AXIS_Y_BASE 48
#define SPEC_AXIS_Z_BASE 56
#define SPEC_TABLE_ROWS 80

/// 当前选中照明体在 spectralTable 中的起始行
int illumBase(constant Uniforms &u) {
    int sel = clamp(u.flags.w, 0, 4);
    return sel * SPEC_ILLUM_STRIDE;
}

// 偏光渲染模式
#define POLAR_MODE_OFF       0
#define POLAR_MODE_CROSSED3D 1
#define SODIUM_ILLUM_INDEX   4

// CIE 1931 2° 标准色度观察者配色函数（400–700nm/10nm，与构建脚本同表）
constant float3 kCMF[SPEC_BINS] = {
    float3(0.014310, 0.000396, 0.067850),
    float3(0.043510, 0.001210, 0.207400),
    float3(0.134380, 0.004000, 0.645600),
    float3(0.283900, 0.011600, 1.385600),
    float3(0.348280, 0.023000, 1.747060),
    float3(0.336200, 0.038000, 1.772110),
    float3(0.290800, 0.060000, 1.669200),
    float3(0.195360, 0.090980, 1.287640),
    float3(0.095640, 0.139020, 0.812950),
    float3(0.032010, 0.208020, 0.465180),
    float3(0.004900, 0.323000, 0.272000),
    float3(0.009300, 0.503000, 0.158200),
    float3(0.063270, 0.710000, 0.078250),
    float3(0.165500, 0.862000, 0.042160),
    float3(0.290400, 0.954000, 0.020300),
    float3(0.433450, 0.994950, 0.008750),
    float3(0.594500, 0.995000, 0.003900),
    float3(0.762100, 0.952000, 0.002100),
    float3(0.916300, 0.870000, 0.001650),
    float3(1.026300, 0.757000, 0.001100),
    float3(1.062200, 0.631000, 0.000800),
    float3(1.002600, 0.503000, 0.000340),
    float3(0.854450, 0.381000, 0.000190),
    float3(0.642400, 0.265000, 0.000050),
    float3(0.447900, 0.175000, 0.000020),
    float3(0.283500, 0.107000, 0.000000),
    float3(0.164900, 0.061000, 0.000000),
    float3(0.087400, 0.032000, 0.000000),
    float3(0.046770, 0.017000, 0.000000),
    float3(0.022700, 0.008210, 0.000000),
    float3(0.011359, 0.004102, 0.000000)
};

// XYZ(D65) → linear sRGB（Metal 矩阵按列构造）
constant float3x3 kXYZToRGB = float3x3(
    float3( 3.24096994, -0.96924364,  0.05563008),
    float3(-1.53738318,  1.87596750, -0.20397696),
    float3(-0.49861076,  0.04155506,  1.05697151)
);

// D65 昼光相对 SPD（锥光白光分支专用，绕过 illumBase 光源选择；与构建脚本同表）
constant float kD65White[31] = {
    0.8372,0.9251,0.9443,0.8766,1.0606,1.1829,1.191,1.1617,1.1718,1.0,
    1.1061,1.0899,1.0596,1.0889,1.0555,1.0515,1.0111,0.9736,0.9686,0.8968,
    0.9099,0.9059,0.8867,0.8422,0.8463,0.8088,0.8109,0.8321,0.7917,0.7047,0.7239
};

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex VertexOut optics_vertex(uint vid [[vertex_id]]) {
    // 与 Web 版全屏三角形相同：[-1,-1],[3,-1],[-1,3]
    float2 p = float2((vid == 1) ? 3.0 : -1.0,
                      (vid == 2) ? 3.0 : -1.0);
    VertexOut out;
    out.position = float4(p, 0.0, 1.0);
    out.uv = p * 0.5 + 0.5; // 屏幕左下角 (0,0)
    return out;
}

/// Sutherland–Hodgman 风格的凸半空间求交：near/far 距离与入射面法向。
bool intersectConvex(constant Uniforms &u,
                     constant float4 *planeTable,
                     float3 origin, float3 direction,
                     thread float &nearT, thread float &farT,
                     thread float3 &nearNormal) {
    nearT = -1e5;
    farT = 1e5;
    nearNormal = float3(0.0, 0.0, 1.0);
    int count = u.flags.z;
    if (u.sphereParams.w > 0.0) {
        // 解析球面 |o + t·d − c|² = r²（|d| = 1）。
        // 语义与半空间版一致：nearT 为入点、farT 为出点、nearNormal 为入点外法向；
        // 射线起点在体内时 nearT < 0，调用方自行拒绝（与平面版同）。
        float3 oc = origin - u.sphereParams.xyz;
        float halfB = dot(direction, oc);
        float cc = dot(oc, oc) - u.sphereParams.w * u.sphereParams.w;
        float disc = halfB * halfB - cc;
        if (disc < 0.0) return false;
        float root = sqrt(disc);
        nearT = -halfB - root;
        farT = -halfB + root;
        nearNormal = normalize(origin + nearT * direction - u.sphereParams.xyz);
        return farT > max(nearT, 0.0);
    }
    for (int index = 0; index < MAX_PLANES; index += 1) {
        if (index >= count) break;
        float4 plane = planeTable[index];
        float denominator = dot(plane.xyz, direction);
        float signedDistance = plane.w - dot(plane.xyz, origin);
        if (fabs(denominator) < 1e-6) {
            if (signedDistance < 0.0) return false;
            continue;
        }
        float distance = signedDistance / denominator;
        if (denominator < 0.0 && distance > nearT) {
            nearT = distance;
            nearNormal = plane.xyz;
        } else if (denominator > 0.0) {
            farT = min(farT, distance);
        }
        if (nearT > farT) return false;
    }
    return farT > max(nearT, 0.0);
}

/// 沿射线在实体内部寻找下一个边界（仅计朝向射向前方的平面）。
bool nextBoundary(constant Uniforms &u,
                  constant float4 *planeTable,
                  float3 origin, float3 direction,
                  thread float &hitT, thread float3 &hitNormal) {
    hitT = 1e5;
    hitNormal = float3(0.0, 0.0, 1.0);
    int count = u.flags.z;
    if (u.sphereParams.w > 0.0) {
        // 体内射线：解析球面的出射点恒为远根（起点在体内 ⇒ 常数为负 ⇒ 远根必为正）
        float3 oc = origin - u.sphereParams.xyz;
        float halfB = dot(direction, oc);
        float cc = dot(oc, oc) - u.sphereParams.w * u.sphereParams.w;
        float disc = halfB * halfB - cc;
        if (disc < 0.0) return false;
        hitT = -halfB + sqrt(disc);
        if (hitT <= EPSILON) return false;
        hitNormal = normalize(origin + hitT * direction - u.sphereParams.xyz);
        return hitT < 1e4;
    }
    for (int index = 0; index < MAX_PLANES; index += 1) {
        if (index >= count) break;
        float4 plane = planeTable[index];
        float denominator = dot(plane.xyz, direction);
        if (denominator <= 1e-6) continue;
        float distance = (plane.w - dot(plane.xyz, origin)) / denominator;
        if (distance > EPSILON && distance < hitT) {
            hitT = distance;
            hitNormal = plane.xyz;
        }
    }
    return hitT < 1e4;
}

float dielectricFresnel(float cosineIncident, float n1, float n2) {
    float cosI = clamp(fabs(cosineIncident), 0.0, 1.0);
    float ratio = n1 / n2;
    float sinTSquared = ratio * ratio * max(0.0, 1.0 - cosI * cosI);
    if (sinTSquared >= 1.0) return 1.0;
    float cosT = sqrt(max(0.0, 1.0 - sinTSquared));
    float parallel = ((n2 * cosI) - (n1 * cosT)) / max(1e-6, (n2 * cosI) + (n1 * cosT));
    float perpendicular = ((n1 * cosI) - (n2 * cosT)) / max(1e-6, (n1 * cosI) + (n2 * cosT));
    return 0.5 * (parallel * parallel + perpendicular * perpendicular);
}

/// s/p 振幅 Fresnel：返回实振幅反射系数 rs/rp、能量透射振幅 sqrt(T)（ts/tp），
/// 全内反射时 |r|=1 且 s/p 相位跃变 ps/pp（Fresnel 菱体偏振旋转的来源）。
void fresnelSP(float n1, float n2, float cosI,
               thread float &rs, thread float &rp,
               thread float &ts, thread float &tp,
               thread float &ps, thread float &pp) {
    cosI = clamp(cosI, 0.0, 1.0);
    float ratio = n1 / n2;
    float sinI2 = max(0.0, 1.0 - cosI * cosI);
    float sinT2 = ratio * ratio * sinI2;
    if (sinT2 >= 1.0) {
        float a = sqrt(max(0.0, sinI2 - ratio * ratio));
        rs = 1.0; rp = 1.0; ts = 0.0; tp = 0.0;
        ps = -2.0 * atan(a / max(cosI, 1e-6));
        pp = -2.0 * atan(a / max(ratio * ratio * cosI, 1e-6));
        return;
    }
    float cosT = sqrt(max(0.0, 1.0 - sinT2));
    float sR = (n1 * cosI - n2 * cosT) / max(1e-6, n1 * cosI + n2 * cosT);
    float pR = (n2 * cosI - n1 * cosT) / max(1e-6, n2 * cosI + n1 * cosT);
    rs = sR; rp = pR;
    ts = sqrt(max(0.0, 1.0 - sR * sR));
    tp = sqrt(max(0.0, 1.0 - pR * pR));
    ps = 0.0; pp = 0.0;
}

/// 复数 z=r·(cos p + i sin p)
void cmul(float r0, float i0, float r1, float p1,
          thread float &orr, thread float &ori) {
    float c = cos(p1), s = sin(p1);
    float x = r1 * c, y = r1 * s;
    orr = r0 * x - i0 * y;
    ori = r0 * y + i0 * x;
}

float3 rotateEnvironment(constant Uniforms &u, float3 direction) {
    float angle = u.cameraRightEnvRot.w * 3.14159265359 / 180.0;
    float c = cos(angle);
    float s = sin(angle);
    return float3(c * direction.x - s * direction.y,
                  s * direction.x + c * direction.y,
                  direction.z);
}

float3 environmentRadiance(constant Uniforms &u, float3 rawDirection) {
    float3 worldDirection = normalize(rawDirection);
    // 偏光模式：下方载物台是经下偏光片的亮漫射光源，上方镜筒/遮光罩近黑。
    // 亮区严格限制在明显朝下的方向，腰围掠射（z≈0）归黑镜筒，避免轮廓眩光。
    if (int(round(u.polarParams.x)) == POLAR_MODE_CROSSED3D) {
        float stage = 1.0 - smoothstep(-0.75, -0.35, worldDirection.z);
        return mix(float3(0.004), float3(1.0), stage);
    }
    float3 direction = rotateEnvironment(u, worldDirection);
    float horizon = smoothstep(-0.6, 0.85, direction.z);
    float3 base = mix(float3(0.12, 0.14, 0.16), float3(0.92, 0.91, 0.88), horizon);
    int env = u.flags.x;
    if (env == 1) base = mix(float3(0.035, 0.045, 0.06), float3(0.82, 0.76, 0.66), horizon);
    if (env == 2) base = mix(float3(0.012, 0.016, 0.022), float3(0.42, 0.50, 0.60), horizon);

    float leftPanel = pow(max(0.0, dot(direction, normalize(float3(-0.72, -0.36, 0.58)))), 42.0);
    float rightPanel = pow(max(0.0, dot(direction, normalize(float3(0.78, 0.18, 0.52)))), 70.0);
    float crownPanel = pow(max(0.0, dot(direction, normalize(float3(0.08, 0.18, 0.98)))), 110.0);
    float warmStrip = pow(max(0.0, dot(direction, normalize(float3(-0.15, 0.96, 0.22)))), 95.0);
    float darkCard = pow(max(0.0, dot(direction, normalize(float3(0.18, -0.92, 0.34)))), 4.0);
    float darkCardTwo = pow(max(0.0, dot(direction, normalize(float3(-0.64, 0.68, 0.35)))), 5.0);

    float3 panels = leftPanel * float3(5.6, 6.0, 6.6)
        + rightPanel * float3(7.2, 6.4, 5.8)
        + crownPanel * float3(4.0, 4.4, 5.2)
        + warmStrip * float3(3.3, 1.9, 0.8);
    float darkStrength = env == 2 ? 1.2 : 0.88;
    float observerCard = smoothstep(0.91, 0.985, dot(worldDirection, normalize(u.observerDispersion.xyz)));
    float3 radiance = max(float3(0.006), base + panels - (darkCard + darkCardTwo) * darkStrength);

    if (env == 3) {
        float azimuth = atan2(worldDirection.y, worldDirection.x);
        float eightFold = 0.5 + 0.5 * cos(8.0 * azimuth);
        float polar = dot(worldDirection, normalize(u.observerDispersion.xyz));
        float scopeRing = smoothstep(0.22, 0.72, polar) * (1.0 - smoothstep(0.88, 0.97, polar));
        float scopeSegments = mix(0.12, 1.0, smoothstep(0.35, 0.78, eightFold));
        float3 scopeWhite = float3(1.55, 1.48, 1.38) * mix(0.34, 1.0, scopeSegments * scopeRing);
        float3 scopeRed = float3(1.2, 0.055, 0.11) * (1.0 - scopeSegments) * scopeRing;
        radiance = float3(0.035) + scopeWhite + scopeRed;
        observerCard = smoothstep(0.84, 0.975, polar);
    }

    return max(float3(0.004), radiance * mix(1.0, env == 3 ? 0.035 : 0.16, observerCard));
}

/// 环境按 Rec.709 亮度折叠为标量（照明体颜色由 SPD 严格给出，环境只保留明度图案）。
float environmentLuminance(constant Uniforms &u, float3 direction) {
    float3 r = environmentRadiance(u, direction);
    return dot(r, float3(0.2126, 0.7152, 0.0722));
}

/// 读取 spectralTable 中 base 行的第 bin 个分量（每行 4 bin）。
float specValue(constant float4 *table, int base, int bin) {
    return table[base + (bin >> 2)][bin & 3];
}

/// 晶体主轴 j（0=X/Np, 1=Y/Nm, 2=Z/Ng）在波长 bin 的吸收系数。
/// 每轴占 8 个 float4 行（31 bin + 末行补零），轴间步长是 8 行而非 1——
/// 误用 SPEC_AXIS_X_BASE + j 会读到轴 0 光谱错位 40/80nm 的数据，
/// 使多色性颜色整体失真（如祖母绿绿段误读橙红段吸收而呈紫色）。
float specAxisValue(constant float4 *table, int j, int bin) {
    return specValue(table, SPEC_AXIS_X_BASE + 8 * j, bin);
}

/// 凸多面体单族波长的透射路径 + 最多 8 次反射反弹。
//  同族波长（蓝 400–490 / 绿 500–570 / 红 580–700）共享同一条几何路径与
//  Fresnel 分裂（无色散差异），逐波长仅 Beer–Lambert 吸收不同；
//  三族各用一个 IOR（B/G 色散），在 fragment 中合并为完整 31 bin 光谱。
//
//  多色性：α_eff(λ) = ½Σ_j(1−(d·e_j)²)α_j(λ)（非偏振光两偏振各半）。
void traceGemSpectral(constant Uniforms &u,
                      constant float4 *planeTable,
                      constant float4 *spectral,
                      float3 rayOrigin, float3 rayDirection,
                      float ior,
                      int binBegin, int binCount,
                      thread float *radiance,
                      thread bool &missed) {
    float nearT;
    float farT;
    float3 entryNormal;
    if (!intersectConvex(u, planeTable, rayOrigin, rayDirection, nearT, farT, entryNormal)
        || nearT < 0.0) {
        missed = true;
        return;
    }

    int iBase = illumBase(u);
    float3 entryPoint = rayOrigin + rayDirection * nearT;
    float entryFresnel = dielectricFresnel(dot(-rayDirection, entryNormal), 1.0, ior);
    float entryLuminance = environmentLuminance(u, reflect(rayDirection, entryNormal));
    float3 insideDirection = refract(rayDirection, entryNormal, 1.0 / ior);
    if (dot(insideDirection, insideDirection) < 1e-7) {
        for (int b = 0; b < binCount; b += 1) {
            int bin = binBegin + b;
            radiance[bin] = entryLuminance * specValue(spectral, iBase, bin) * entryFresnel;
        }
        missed = false;
        return;
    }

    thread float throughput[SPEC_BINS];
    for (int b = 0; b < binCount; b += 1) {
        int bin = binBegin + b;
        radiance[bin] = entryLuminance * specValue(spectral, iBase, bin) * entryFresnel;
        throughput[bin] = 1.0 - entryFresnel;
    }
    float3 insideOrigin = entryPoint + insideDirection * EPSILON;
    int maxBounces = u.flags.y;
    for (int bounce = 0; bounce < MAX_BOUNCES; bounce += 1) {
        if (bounce >= maxBounces) break;
        float boundaryT;
        float3 boundaryNormal;
        if (!nextBoundary(u, planeTable, insideOrigin, insideDirection, boundaryT, boundaryNormal)) break;
        float3 boundaryPoint = insideOrigin + insideDirection * boundaryT;
        // 该光段方向上的逐波长有效吸收（局部系对角张量；非偏振光两偏振各半）
        float3 w = 0.5 * (float3(1.0) - insideDirection * insideDirection);
        float maxThroughput = 0.0;
        for (int b = 0; b < binCount; b += 1) {
            int bin = binBegin + b;
            float segmentAbsorption =
                w.x * specValue(spectral, SPEC_AXIS_X_BASE, bin)
              + w.y * specValue(spectral, SPEC_AXIS_Y_BASE, bin)
              + w.z * specValue(spectral, SPEC_AXIS_Z_BASE, bin);
            throughput[bin] *= exp(-max(segmentAbsorption, 0.0) * boundaryT);
            maxThroughput = max(maxThroughput, throughput[bin]);
        }
        float fresnel = dielectricFresnel(dot(insideDirection, boundaryNormal), ior, 1.0);
        float3 exitDirection = refract(insideDirection, -boundaryNormal, ior);
        if (dot(exitDirection, exitDirection) > 1e-7) {
            float exitLuminance = environmentLuminance(u, exitDirection);
            for (int b = 0; b < binCount; b += 1) {
                int bin = binBegin + b;
                radiance[bin] += throughput[bin] * (1.0 - fresnel)
                              * exitLuminance * specValue(spectral, iBase, bin);
                throughput[bin] *= fresnel;
            }
        } else {
            for (int b = 0; b < binCount; b += 1) {
                int bin = binBegin + b;
                throughput[bin] *= fresnel;
            }
        }
        if (maxThroughput * fresnel < 0.002) break;
        insideDirection = reflect(insideDirection, boundaryNormal);
        insideOrigin = boundaryPoint + insideDirection * EPSILON;
    }
    missed = false;
}

// MARK: - 偏光实验室 · 锥光干涉图（conoscopic figure）
//
//  焦平面每像素对应板内一个波向 d(θ,φ)，样品等效为厚度 d 的平行板（教学模型）。
//  Jones 演算：起偏 P(p) → 延迟器（方位 ψ、半相位 Δ、旋光 ρ）→ 中偏光片 T(t,可选)
//  → 分析镜 A(β)。p/t/β 都是实验室系绝对方位（polarLab.x / polarLab.y / polarParams.y）。
//   - 一轴晶：Δ = π·d·(n_eff(θ)−ω)/λ·1e6，ψ=φ（理想模型光轴恒垂直台面，
//     台角对一轴晶图式无作用）；黑十字+同心等色环自然产生
//   - 水晶：光轴附近线性双折射为零，圆双折射（旋光）使中心有色不黑 → 牛眼
//   - 二轴晶：两光轴 ±V 出露点，Biots 振动方位平分线 + sinθ1sinθ2 延迟 → 双眼弯臂
//   - 均质体：Δ≡ρ≡0，正交下全暗
//  全部常数取自系统笔记钠光 589.3nm 实测值，无贴图/无程序化伪饰。

#define KIND_ISO 0
#define KIND_UNI_POS 1
#define KIND_UNI_NEG 2
#define KIND_BI_POS 3
#define KIND_BI_NEG 4
#define CONO_MAX_ROT 6
#define CONO_WL_NM 589.3

/// 沿光轴比旋光率（°/mm）插值；buffer(4) 打包为 float2 对（每行 2 对）
float conoRotationRate(constant float4 *rotTable, int count, float wl) {
    if (count <= 0 || rotTable == nullptr) return 0.0;
    float wl0 = rotTable[0].x, r0 = rotTable[0].y;
    if (wl <= wl0) return r0;
    for (int i = 0; i < count; i += 1) {
        float wa = rotTable[i >> 1][(i & 1) * 2];
        float ra = rotTable[i >> 1][(i & 1) * 2 + 1];
        if (wl <= wa) {
            float wb = rotTable[(i - 1) >> 1][((i - 1) & 1) * 2];
            float rb = rotTable[(i - 1) >> 1][((i - 1) & 1) * 2 + 1];
            float t = (wl - wb) / max(1e-6, wa - wb);
            return rb + (ra - rb) * t;
        }
    }
    float wlLast = rotTable[(count - 1) >> 1][((count - 1) & 1) * 2];
    float rLast = rotTable[(count - 1) >> 1][((count - 1) & 1) * 2 + 1];
    (void)wlLast;
    return rLast;
}

/// 给定板内波向 d，返回延迟器方位 psi 与（半相位 delta、旋光 rho，弧度）
void conoEigen(constant Uniforms &u, constant float4 *rotTableDummy,
               float3 d, float wl,
               thread float &psi, thread float &delta, thread float &rho) {
    psi = 0.0; delta = 0.0; rho = 0.0;
    int kind = int(round(u.crystalA.x));
    float thickness = u.polarParams.w;
    // 样品总方位角（台角 + 宝石自转）：水平拖过宝石之后干涉图必须跟着变，
    // 只取台角会把「手拨样品」这一步漏掉，而画面只是「转台没反应」。
    float stage = specimenSpin(u);
    if (kind == KIND_ISO) return;

    if (kind == KIND_UNI_POS || kind == KIND_UNI_NEG) {
        float nO = u.crystalA.y, nE = u.crystalA.z;
        float sinT = length(d.xy), cosT = max(d.z, 0.05);
        float phi = atan2(d.y, d.x);
        float nEff = nO * nE / sqrt(nE * nE * cosT * cosT + nO * nO * sinT * sinT);
        float dn = nEff - nO;                       // 带光性符号
        float L = thickness / cosT;
        float Rnm = dn * L * 1e6;
        delta = M_PI_F * Rnm / wl;
        // 实验室（偏光片）系方位：锥光图式中晶体不随载物台转动，
        // 黑十字始终沿偏光方向；载物台角仅影响单偏光/正交 3D 分支。
        psi = phi;
        return;  // 旋光项在下方统一读取（需 buffer）
    }
    // 二轴晶：载物台旋转 = 晶体绕视线（Bxa）刚体旋转——两光轴出露点与等色环
    // 绕中心公转（出露半径 = 光轴倾角 V 的投影），黑臂由「振动平分线对齐固定
    // 偏光片」产生，只扫动开合、不随台公转。旧实现把 m1/m2 钉死在屏幕 E-W、
    // 仅给平分线叠加 stage，出露点不动——与真实锥光（polarDielectricBasis 经
    // indicatrixFrame 随台刚性旋转光轴）自相矛盾。平分线直接由旋转后的光轴
    // 构造，不再叠加 stage（两次计角会转过头）。
    float V = u.crystalB.w;
    float sV = sin(V), cV = cos(V), sn = sin(stage), cs = cos(stage);
    float3 m1 = float3(sV * cs, sV * sn, cV);
    float3 m2 = float3(-sV * cs, -sV * sn, cV);
    float3 u1 = m1 - dot(m1, d) * d;
    float3 u2 = m2 - dot(m2, d) * d;
    float n1 = length(u1), n2 = length(u2);
    float bx, by;
    if (n1 > 1e-6 && n2 > 1e-6) {
        bx = u1.x / n1 + u2.x / n2;
        by = u1.y / n1 + u2.y / n2;
        psi = atan2(by, bx);
    } else {
        // 出露点/中心退化：振动方位沿光轴面迹线及其垂线（±90° 等价），取垂线
        bx = -sn; by = cs;
        psi = atan2(by, bx);
    }
    float s1 = sqrt(max(0.0, 1.0 - dot(d, m1) * dot(d, m1)));
    float s2 = sqrt(max(0.0, 1.0 - dot(d, m2) * dot(d, m2)));
    float L = thickness / max(d.z, 0.05);
    float Rnm = (u.crystalB.z - u.crystalB.x) * L * 1e6 * s1 * s2;
    delta = M_PI_F * Rnm / wl;
}

// 锥光白光分支读取 D65 常量表（绕过 illumBase 的光源选择）
float specValueIllumWhite(float wlUnused, int bin) {
    (void)wlUnused;
    return kD65White[bin];
}

/// Jones：|A · T? · R(-ψ)[cosΔ I + i sinΔ(Δσz+ρσy)/Γ] R(ψ) · P(p)|²。
/// p = 起偏绝对方位（polarLab.x）；β = 分析镜绝对方位；T = 可选中偏光片，
/// 紧邻分析镜下游（晶片→T→A，中间无旋光介质，等效于先投影 T 再乘 cos∠(A,T)）。
float conoJones(constant Uniforms &u, float3 d, float wl, float beta,
                float rhoDegPerMm) {
    float psi, delta, rhoUnused;
    conoEigen(u, nullptr, d, wl, psi, delta, rhoUnused);
    float pAng = u.polarLab.x;
    bool midOn = u.polarLab.z > 0.5;
    float tAng = u.polarLab.y;
    // 水晶旋光：仅一轴晶且提供旋光色散；离轴按 cosθ 衰减（严格沿 c 轴测量）
    int kind = int(round(u.crystalA.x));
    float rho = 0.0;
    if (kind == KIND_UNI_POS || kind == KIND_UNI_NEG) {
        // 离轴衰减只与 cosθ 有关（sinθ 在这一支用不上；原先是声明了没用的局部量）
        float cosT = max(d.z, 0.05);
        float L = u.polarParams.w / cosT;
        rho = rhoDegPerMm * L * (M_PI_F / 180.0) * cosT;
    }
    float gamma = sqrt(delta * delta + rho * rho);
    // 无延迟：纯 Malus 链 P→(T)→A（P 转了以后不再是 cos²β，而是相对夹角）
    if (gamma < 1e-9) {
        float cp = cos(beta - pAng);
        if (!midOn) return cp * cp;
        float ctP = cos(tAng - pAng), caT = cos(beta - tAng);
        return ctP * ctP * caT * caT;
    }
    float c = cos(gamma), s = sin(gamma);
    float a = s * delta / gamma;       // iσz 系数（虚部）
    float q = s * rho / gamma;         // iσy 系数（实部非对角）
    // R(ψ)(cos p, sin p) = (cos(ψ+p), sin(ψ+p))：把起偏方位旋入延迟器主轴系
    float ct = cos(psi + pAng), st = sin(psi + pAng);
    // M'(ct,st)
    float xr = c * ct + q * st,  xi = a * ct;
    float yr = -q * ct + c * st, yi = -a * st;
    // R(-ψ)
    float Xr = ct * xr + st * yr, Xi = ct * xi + st * yi;
    float Yr = -st * xr + ct * yr, Yi = -st * xi + ct * yi;
    if (midOn) {
        // 中偏光片 T：投影到 t 方向（振幅 E_T），紧邻的分析镜 A 再乘 cos(β−t)
        float mr = cos(tAng) * Xr + sin(tAng) * Yr;
        float mi = cos(tAng) * Xi + sin(tAng) * Yi;
        float caT = cos(beta - tAng);
        return caT * caT * (mr * mr + mi * mi);
    }
    float cb = cos(beta), sb = sin(beta);
    float Er = cb * Xr + sb * Yr, Ei = cb * Xi + sb * Yi;
    return Er * Er + Ei * Ei;
}

fragment float4 optics_cono_fragment(VertexOut in [[stage_in]],
                                     constant Uniforms &u [[buffer(1)]],
                                     constant float4 *rotTable [[buffer(4)]]) {
    // 屏幕 → 板内波向（视场圆形，边缘对应 thetaMax）
    float2 centered = (in.uv - 0.5) * 2.0;
    centered.x *= u.resolution.x / max(u.resolution.y, 1.0);
    float rr = length(centered);
    float theta = min(rr, 1.0) * u.polarFlags.y;
    float phi = atan2(centered.y, centered.x);
    float st = sin(theta), ct = cos(theta);
    float3 d = float3(st * cos(phi), st * sin(phi), ct);
    float beta = u.polarParams.y;
    int rotCount = int(round(u.polarFlags.z));

    float X, Y, Z, Xw, Yw, Zw;
    X = Y = Z = Xw = Yw = Zw = 0.0;
    bool sodium = u.polarFlags.x > 0.5;
    if (sodium) {
        // 单色光不做色适应：589.3nm CIE 色度（Y 归一），强度由 Jones 调制
        float rate = conoRotationRate(rotTable, rotCount, CONO_WL_NM);
        float I = conoJones(u, d, CONO_WL_NM, beta, rate);
        X = I * 1.325; Y = I * 1.0; Z = I * 0.0026;
        Xw = Yw = Zw = 1.0;
    } else {
        for (int b = 0; b < SPEC_BINS; b += 1) {
            float wl = 400.0 + 10.0 * b;
            float S = specValueIllumWhite(wl, b);
            float rate = conoRotationRate(rotTable, rotCount, wl);
            float I = conoJones(u, d, wl, beta, rate);
            X += kCMF[b].x * I * S; Y += kCMF[b].y * I * S; Z += kCMF[b].z * I * S;
            Xw += kCMF[b].x * S;    Yw += kCMF[b].y * S;    Zw += kCMF[b].z * S;
        }
    }
    float3 xyz = float3(X / max(Xw, 1e-6), Y / max(Yw, 1e-6), Z / max(Zw, 1e-6));
    float3 color = kXYZToRGB * xyz;
    color = max(color, float3(0.0)) * 1.25;
    // 圆形视场外全黑（镜筒；theta 被钳制会在边缘产生伪条纹，直接遮黑）
    if (rr > 1.0) color = float3(0.0);
    color = clamp(color, 0.0, 1.0);
    color = pow(color, float3(1.0 / 2.2));
    return float4(color, 1.0);
}

// MARK: - 真实锥光（mode 3）：理想平行晶片的方向像（与 3D 正交偏光共用相机观察轴）
//
//  每个像素 = 一个空气中波向 (θ,φ)；观察轴取当前相机视线，而相机在偏光实验室里
//  恒停在沿台面法线的顶视位（共轴观察位）。所以本模式的「出露点偏心」不再靠
//  拖斜视角得到，而是靠**扳歪宝石**（gemTumble）与面板上的光轴定向滑杆 ——
//  这与真实偏光仪一致：物镜轴是仪器，样品才是被操作的那个。
//  样品是垂直物镜轴夹持的平行平面晶片（教学标准样品），由 traceSlabPolar 演算：
//  光线只穿过两个平行表面，无亭部刻面、无全内反射、无几何随机性，单帧即收敛。
//  注意这里**不**走 traceGemPolar 的刻面路径追踪 —— 那条路只服务于 3D 正交偏光
//  （optics_fragment 的 POLAR_MODE_CROSSED3D 分支）。
//
//  样品的两个姿态旋钮都经 indicatrixFrame(u, stage, tumble, …) 刚性旋转光率体
//  （Rz(stage)·T：T = 宝石在台面系里的姿态，内层；stage = 台角 + 宝石自转，外层），
//  而偏光片固定在实验室系，所以干涉图是否随样品转动取决于**光轴相对台面法线的夹角**：
//    · 一轴晶 + 光轴∥台面法线：每条光线的相位延迟只由波向与光轴的夹角决定，
//      绕光轴自转不改变任何一条光线 → 黑十字与等色环整体不动（这是判据，不是缺陷）。
//    · 光轴斜置（含把宝石扳歪），或二轴晶（光轴面绕法线转）→ 干涉图随样品扫动。

#define CONO_GEM_TABLE_R 0.5     // 台面内采样半径（腰围半径 1；标准圆钻台面 ~0.53–0.57）

// 前向声明：偏振路径追踪主核定义于本文件后文。
// lowerExitOnly 只服务 3D 正交偏光的「只累加亭部出射」分支；当前三个调用点都走默认
// 值 false（真实锥光已改走 traceSlabPolar，不再经过这里）。
// meshNodeCount > 0 时几何走三角网格（原矿晶形，buffer 8/9），planeTable 被忽略——
// 偏光 Jones 演算本身与几何表示无关，只要求「入射 t+法线」和「体内下一边界」两个接口。
void traceGemPolar(constant Uniforms &u,
                   constant float4 *planeTable,
                   constant float4 *spectral,
                   constant float4 *rotTable,
                   float3 rayOrigin, float3 rayDirection,
                   float ior, int binBegin, int binCount,
                   thread float *radiance,
                   thread bool &missed,
                   int meshNodeCount,
                   constant float4 *meshNodes,
                   constant float4 *meshTris,
                   bool lowerExitOnly = false);

// 网格 BVH 求交与网格光谱路径追踪定义于文件末尾（网格光学路径章节）；
// traceGemPolar 与 optics_fragment 的原矿晶形分支会调用，MSL 先声明后使用。
bool intersectMeshBVH(int nodeCount,
                      constant float4 *nodeTable,
                      constant float4 *triTable,
                      float3 origin, float3 direction,
                      thread float &hitT, thread float3 &hitNormal);
void traceMeshSpectral(constant Uniforms &u,
                       constant float4 *spectral,
                       int nodeCount,
                       constant float4 *nodeTable,
                       constant float4 *triTable,
                       float3 rayOrigin, float3 rayDirection,
                       float ior, float frost, int binBegin, int binCount,
                       thread float *radiance,
                       thread bool &missed);

// MARK: - 真实锥光 · 理想平行晶片
//
// 以下辅助函数定义于本文件后部（3D 正交偏光章节），先给出原型：
void orthoBasis(float3 d, thread float3 &u, thread float3 &v);
// tumble 参数 = 宝石在台面系里的姿态 T（内层姿态），见 indicatrixFrame 定义处的
// 顺序说明。真实锥光传 u.gemTumble；3D 正交偏光传 kIdentityQuat（GemPose 已吸收）。
void indicatrixFrame(constant Uniforms &u, float stage, float4 tumble,
                     thread float3 ax[3],
                     thread float3 &m1, thread float3 &m2);
bool polarDielectricBasis(constant Uniforms &u, float3 dir, float stage, float4 tumble,
                          thread float3 &e1, thread float3 &e2);
void dmRealTransform(thread float &d0, thread float &d1v,
                     thread float &zr, thread float &zi,
                     float m00, float m01, float m10, float m11);
float sincAbs(float x);
//
// 真实锥光的试验样品是垂直物镜轴夹持的平行平面晶片（教学标准样品形状），
// 而非刻面宝石：光线只穿过两个平行表面，无亭部刻面、无全内反射、无几何
// 随机性——每个传播方向的结果完全确定，单帧即可收敛（无蒙特卡洛闪烁）。
// 晶体光学（双折射/旋光/吸收/色散/二轴晶 Biot 定律）与刻面路径追踪完全一致。
//
// 宝石姿态（gemTumble）在这条路径里**只转光率体、不转板**：板代表「被物镜
// 夹持的样品面」，它的朝向是仪器给的（⊥ 物镜轴）；光率体固定在晶体上，
// 所以把宝石扳歪 = 光轴相对物镜轴倾斜。这与面板上既有的「光轴定向」滑杆
// 走的是同一条路（那条也从不动板），两者叠加互不冲突。
//
// 观察轴 fwd 取自当前 3D 相机视线：3D 正交偏光下把视角拖斜，光轴相对观察轴
// 倾斜，锥光图出露点随即偏心——两个模式共享同一相机与同一世界系光轴定向。
//
// 逆向光路（互易）：物镜侧注入分析镜 A 偏振 → 前表面空气→晶 Fresnel 折射
// → 晶内一段（Beer 吸收 + 双折射相位 + 旋光）→ 后表面晶→空气折射
// → 投影到起偏镜 P（下方均匀亮漫射器）。

void traceSlabPolar(constant Uniforms &u,
                    constant float4 *spectral,
                    constant float4 *rotTable,
                    float3 dir,
                    float3 fwd,
                    float tUnits,
                    float ior,
                    int binBegin, int binCount,
                    thread float *radiance) {
    // 仅清零本次负责的波段（蓝/绿/红三组分三次调用、+= 累积）。
    // 切勿写 radiance[b]——后两次调用（binBegin=10/18）会把已累积的
    // 400–520nm 蓝绿波段抹零，导致白光下整体缺蓝而偏黄。
    for (int b = 0; b < binCount; b += 1) radiance[binBegin + b] = 0.0;

    // 样品总方位角（载物台角 + 宝石自转）；偏光片固定在实验室系，不随样品转。
    float stage = specimenSpin(u);
    float beta = u.polarParams.y;
    // 起偏/分析镜各自独立可旋转，均固定于实验室水平面（载物台只转宝石）
    float pAng = u.polarLab.x;
    float3 Pworld = float3(cos(pAng), sin(pAng), 0.0);
    float3 Aworld = float3(cos(beta), sin(beta), 0.0);
    // 中偏光片 T 在 A 之下、晶片之上（逆向光路从上往下先遇 A 再遇 T）
    bool midOn = u.polarLab.z > 0.5;
    float tAng = u.polarLab.y;
    float3 Tworld = float3(cos(tAng), sin(tAng), 0.0);

    // —— 前表面（物镜侧）：外法线 -fwd 指向入射来向 ——
    float3 nf = -fwd;
    float cosI = clamp(dot(-dir, nf), 0.0, 1.0);
    float3 ki = refract(dir, nf, 1.0 / ior);
    if (dot(ki, ki) < 1e-7) return;   // 平行板几何下不会 TIR，保险起见

    float3 uB, vB;
    orthoBasis(dir, uB, vB);
    float3 s0 = cross(dir, nf);
    if (length(s0) < 1e-4) s0 = uB;
    s0 = normalize(s0);
    float3 p0 = normalize(cross(s0, dir));
    float rs0, rp0, ts0, tp0, ps0, pp0;
    fresnelSP(1.0, ior, cosI, rs0, rp0, ts0, tp0, ps0, pp0);

    // 逆向入口是分析镜 A：空气中 E ⊥ 入射方向
    float3 pair = Aworld - dir * dot(Aworld, dir);
    if (length(pair) < 1e-4) pair = float3(0, 1, 0) - dir * dot(float3(0, 1, 0), dir);
    pair = normalize(pair);
    // 中偏光片 T（A 之下、晶片之上）：逆向光路先 A 后 T。两片紧邻、中间无
    // 旋光介质，合成振幅 = (E·Â)·cos∠(A,T)，偏振沿 T 的投影方向——
    // 若把 T 插到与任一主偏光片同侧，会被紧邻的同侧片吞并，故只有这一处生效。
    float midAmp = 1.0;
    if (midOn) {
        float3 tv = Tworld - dir * dot(Tworld, dir);
        float tl = length(tv);
        if (tl > 1e-4) {
            tv /= tl;
            midAmp = dot(pair, tv);
            pair = tv;
        }
        // T∥dir 退化：该片对这一波向无偏振作用，跳过
    }
    float es0 = midAmp * dot(pair, s0), ep0 = midAmp * dot(pair, p0);
    float3 pt0 = normalize(cross(s0, ki));
    float3 eTrans0 = ts0 * es0 * s0 + tp0 * ep0 * pt0;

    // —— 晶内本征偏振（世界系光率体：宝石姿态 T → 台角 → 光轴定向）——
    // 复合顺序 Rz(stage)·T（T 在内层），理由见 indicatrixFrame 定义处。
    // 这里传 u.gemTumble 是本轮的核心：扳歪宝石就是扳歪它的光率体，光轴相对
    // 物镜轴倾斜 → 出露点偏心、等色环扫过视域。漏传（传 kIdentityQuat）时
    // 画面在竖直拖拽下**完全不动**，而「拖了没反应」很容易被误判成「手势没接线」，
    // 于是跑到错误的地方去找 —— 所以这一处必须与 GemPose 同源。
    float3 cax[3]; float3 biAxis1, biAxis2;
    indicatrixFrame(u, stage, u.gemTumble, cax, biAxis1, biAxis2);
    int kind = int(round(u.crystalA.x));
    float3 e1, e2;
    if (!polarDielectricBasis(u, ki, stage, u.gemTumble, e1, e2)) {
        // 均质/沿轴退化：垂直波向平面内 590nm 吸收二次型本征向量
        float3 ku, kv;
        orthoBasis(ki, ku, kv);
        int refBin = 19;
        float b11 = 0, b22 = 0, b12 = 0;
        for (int j = 0; j < 3; j += 1) {
            float uj = dot(ku, cax[j]), vj = dot(kv, cax[j]);
            float aj = specAxisValue(spectral, j, refBin);
            b11 += uj * uj * aj; b22 += vj * vj * aj; b12 += uj * vj * aj;
        }
        // 二次型无择优轴（均质各向等吸收，或无色品种吸收全为 0）时
        // atan2(0,0) 在 GPU 上结果未定义（常为 NaN）→ γ 污染整条 Jones 链全暗；
        // 此时任意正交基都是本征基，γ 取 0。
        float anis = max(fabs(b11 - b22), 2.0 * fabs(b12));
        float gamma = anis < 1e-6 ? 0.0 : 0.5 * atan2(2.0 * b12, b11 - b22);
        e1 = cos(gamma) * ku + sin(gamma) * kv;
        e2 = -sin(gamma) * ku + cos(gamma) * kv;
    }
    float p1 = dot(eTrans0, e1), p2 = dot(eTrans0, e2);

    // —— 晶内唯一光段：板两表面垂直 fwd、间距 tUnits ——
    const float mmPerUnit = 4.0;
    float segT = tUnits / max(fabs(dot(ki, fwd)), 1e-4);
    float segLmm = segT * mmPerUnit;

    float dnEff = 0.0;
    if (kind == KIND_UNI_POS || kind == KIND_UNI_NEG) {
        float cosA = clamp(fabs(dot(ki, cax[2])), 0.0, 1.0);
        float sinA = sqrt(max(0.0, 1.0 - cosA * cosA));
        float nO = u.crystalA.y, nE = u.crystalA.z;
        float nEff = nO * nE / sqrt(max(1e-7,
            nE * nE * cosA * cosA + nO * nO * sinA * sinA));
        dnEff = nEff - nO;
    } else if (kind == KIND_BI_POS || kind == KIND_BI_NEG) {
        float s1 = sqrt(max(0.0, 1.0 - dot(ki, biAxis1) * dot(ki, biAxis1)));
        float s2 = sqrt(max(0.0, 1.0 - dot(ki, biAxis2) * dot(ki, biAxis2)));
        dnEff = (u.crystalB.z - u.crystalB.x) * s1 * s2;
    }
    float retardMm = dnEff * segLmm;
    int rotCount = int(round(u.polarFlags.z));
    bool rotActive = rotCount > 0 && (kind == KIND_UNI_POS || kind == KIND_UNI_NEG);

    bool sodiumLight = u.flags.w == SODIUM_ILLUM_INDEX;
    float lcMm = sodiumLight ? 0.58 : 0.012;
    int iBase = illumBase(u);

    thread float d1[SPEC_BINS], d2[SPEC_BINS], xr[SPEC_BINS], xi[SPEC_BINS];
    for (int b = 0; b < binCount; b += 1) {
        d1[b] = p1 * p1; d2[b] = p2 * p2;
        xr[b] = p1 * p2; xi[b] = 0.0;
    }
    for (int b = 0; b < binCount; b += 1) {
        int bin = binBegin + b;
        float a1v = 0.0, a2v = 0.0;
        for (int j = 0; j < 3; j += 1) {
            float q1 = dot(e1, cax[j]), q2 = dot(e2, cax[j]);
            float aj = specAxisValue(spectral, j, bin);
            a1v += q1 * q1 * aj;
            a2v += q2 * q2 * aj;
        }
        float att1 = exp(-0.5 * max(a1v, 0.0) * segT);
        float att2 = exp(-0.5 * max(a2v, 0.0) * segT);
        d1[b] *= att1 * att1; d2[b] *= att2 * att2;
        xr[b] *= att1 * att2; xi[b] *= att1 * att2;
        float wlNm = 400.0 + 10.0 * bin;
        float hd = M_PI_F * dnEff * segLmm * 1e6 / wlNm;
        float cr = cos(2.0 * hd), sr = sin(2.0 * hd);
        float ozr = xr[b], ozi = xi[b];
        xr[b] = cr * ozr - sr * ozi;
        xi[b] = sr * ozr + cr * ozi;
        if (rotActive) {
            // 圆双折射 Δn_g=ρ(°/mm)·λ(nm)/180e6；离轴线双折射主导时竞争因子 f→0
            float rate = conoRotationRate(rotTable, rotCount, wlNm);
            float dnG = fabs(rate) * wlNm / 1.8e8;
            float rotW = dnG / sqrt(dnG * dnG + dnEff * dnEff);
            float chi = rate * (M_PI_F / 180.0) * segLmm * rotW;
            float cc = cos(chi), ss = sin(chi);
            float rd0 = d1[b], rd1 = d2[b], rzr = xr[b], rzi = xi[b];
            dmRealTransform(rd0, rd1, rzr, rzi, cc, -ss, ss, cc);
            d1[b] = rd0; d2[b] = rd1; xr[b] = rzr; xi[b] = rzi;
        }
    }

    // —— 后表面（起偏镜侧）：外法线 +fwd，晶→空气 ——
    float3 nb = fwd;
    float3 sB = cross(ki, nb);
    if (length(sB) < 1e-4) sB = uB;
    sB = normalize(sB);
    float rsB, rpB, tsB, tpB, psB, ppB;
    fresnelSP(ior, 1.0, clamp(dot(ki, nb), 0.0, 1.0),
              rsB, rpB, tsB, tpB, psB, ppB);
    float3 exitDirection = refract(ki, -nb, ior);
    if (dot(exitDirection, exitDirection) < 1e-7) return;
    // ρ 变换到【晶内】(sB, piB) 基、乘振幅透射；分析镜 p 投影读空气侧 ptB
    // （与刻面路径追踪同一物理：直接用空气基会在斜出射时反转 0°/90° 明暗）
    float3 piB = normalize(cross(sB, ki));
    float m00 = dot(sB, e1),  m01 = dot(sB, e2);
    float m10 = dot(piB, e1), m11 = dot(piB, e2);
    float3 ptB = normalize(cross(sB, exitDirection));
    float3 aOut = Pworld - exitDirection * dot(Pworld, exitDirection);
    if (length(aOut) < 1e-4) { float3 ux, vx; orthoBasis(exitDirection, ux, vx); aOut = vx; }
    aOut = normalize(aOut);
    float ka_s = dot(aOut, sB), ka_p = dot(aOut, ptB);
    float visTotal = sincAbs(M_PI_F * fabs(retardMm) / max(lcMm, 1e-6));
    const float stageDiffuser = 1.0;
    for (int b = 0; b < binCount; b += 1) {
        int bin = binBegin + b;
        float sd0 = d1[b], sd1 = d2[b], szr = xr[b], szi = xi[b];
        dmRealTransform(sd0, sd1, szr, szi, m00, m01, m10, m11);
        sd0 *= tsB * tsB; sd1 *= tpB * tpB;
        szr *= tsB * tpB * visTotal; szi *= tsB * tpB * visTotal;
        float intensity = ka_s * ka_s * sd0 + ka_p * ka_p * sd1
                        + 2.0 * ka_s * ka_p * szr;
        radiance[bin] += intensity * stageDiffuser * specValue(spectral, iBase, bin);
    }
}

/// Pass 1：理想平行晶片方向像。结果由传播方向唯一确定（无刻面几何随机性），
/// 单帧即为最终图像，无需渐进累积；乒乓纹理/双 pass 框架保留以兼容模拟器。
/// 观察轴取当前相机视线（偏光实验室里恒为沿台面法线的共轴观察位）；
/// 屏幕基锚定实验室偏光片（见函数体内注释），干涉图只偏心、不随样品转动滚转。
fragment float4 optics_cono_gem_accum_fragment(
        VertexOut in [[stage_in]],
        constant Uniforms &u [[buffer(1)]],
        constant float4 *planeTable [[buffer(2)]],
        constant float4 *spectral [[buffer(3)]],
        constant float4 *rotTable [[buffer(4)]],
        texture2d<float> accumPrev [[texture(1)]]) {
    float2 centered = (in.uv - 0.5) * 2.0;
    centered.x *= u.resolution.x / max(u.resolution.y, 1.0);
    float rr = length(centered);

    float3 linear = float3(0.0);
    if (rr <= 1.0) {
        float theta = min(rr, 1.0) * u.polarFlags.y;
        float phi = atan2(centered.y, centered.x);
        // 观察轴取自当前 3D 相机；屏幕基则**锚定实验室偏光片**，不取相机 right/up——
        // 相机 right = (cos yaw, sin yaw, 0) 的方位角随拖拽转，等于把整幅干涉图在屏上
        // 带着滚转。一轴晶（光轴∥台面法线）的判据是黑十字只随观察轴偏心、永不偏转；
        // 锥光图的屏（分析镜十字丝）相对镜体也是固定的。取 P 的横向投影作屏幕右轴，
        // 无论视线怎么斜，起偏/分析镜方位在屏上始终不动。（顶视时 fwd=-z、right=P=x、
        // up=A=y，与旧实现完全一致——回正姿态下的画面不变。）
        float3 fwd = normalize(u.cameraForwardExposure.xyz);
        float3 labP = float3(cos(u.polarLab.x), sin(u.polarLab.x), 0.0);
        // 视轴与 P 重合时 P 无横向分量，退用分析镜方位（P⊥A，两者不会同时退化）
        float3 polRef = fabs(dot(labP, fwd)) > 0.999
            ? float3(cos(u.polarParams.y), sin(u.polarParams.y), 0.0)
            : labP;
        float3 right = normalize(polRef - fwd * dot(polRef, fwd));
        float3 up = normalize(cross(right, fwd));
        float3 radial = right * cos(phi) + up * sin(phi);
        // 后焦面位置正比于【正向出射】波矢横向分量；逆向追踪 dir 横向取 −radial
        float3 dir = normalize(fwd * cos(theta) - radial * sin(theta));

        float ior = u.bodyColorIor.w;
        float dispersion = u.observerDispersion.w;
        float redIor = max(1.001, ior - dispersion * 0.48);
        float blueIor = ior + dispersion * 0.52;
        float tUnits = u.polarParams.w / 4.0;   // 板厚 mm → 几何单位（1 单位=4mm）

        // 每方向一次确定性演算：B/G/R 三族波长共享几何、各自 IOR（色散）
        thread float spectralRadiance[SPEC_BINS];
        for (int b = 0; b < SPEC_BINS; b += 1) spectralRadiance[b] = 0.0;
        traceSlabPolar(u, spectral, rotTable, dir, fwd, tUnits,
                       blueIor, 0, 10, spectralRadiance);
        traceSlabPolar(u, spectral, rotTable, dir, fwd, tUnits,
                       ior, 10, 8, spectralRadiance);
        traceSlabPolar(u, spectral, rotTable, dir, fwd, tUnits,
                       redIor, 18, 13, spectralRadiance);

        if (u.flags.w == SODIUM_ILLUM_INDEX) {
            float Y = 0.0;
            for (int b = 0; b < SPEC_BINS; b += 1) Y += kCMF[b].y * spectralRadiance[b];
            linear = kXYZToRGB * float3(1.325, 1.0, 0.0026) * Y;
        } else {
            int whiteBase = illumBase(u);
            float X = 0.0, Y = 0.0, Z = 0.0;
            float Xw = 0.0, Yw = 0.0, Zw = 0.0;
            for (int b = 0; b < SPEC_BINS; b += 1) {
                float s = specValue(spectral, whiteBase, b);
                X += kCMF[b].x * spectralRadiance[b];
                Y += kCMF[b].y * spectralRadiance[b];
                Z += kCMF[b].z * spectralRadiance[b];
                Xw += kCMF[b].x * s;
                Yw += kCMF[b].y * s;
                Zw += kCMF[b].z * s;
            }
            linear = kXYZToRGB * float3(X / max(Xw, 1e-6),
                                        Y / max(Yw, 1e-6),
                                        Z / max(Zw, 1e-6));
        }
        linear *= exp2(u.cameraForwardExposure.w);
        if (!isfinite(linear.x) || !isfinite(linear.y) || !isfinite(linear.z))
            linear = float3(0.0);
        linear = clamp(linear, float3(0.0), float3(64.0));
    }

    // 确定性结果：直接输出（无 running average，参数/相机变化当帧即最终图像）
    return float4(linear, 1.0);
}

/// Pass 2：采样收敛后的线性累积纹理，ACES 色调映射 + 圆形镜筒遮罩到 drawable。
/// 累积纹理以低分辨率渲染（最长边约 480px），此处 linear 过滤放大——锥光图为
/// 平滑低频图案，线性放大无可见锯齿，单帧路径追踪成本却降低数倍（真机发热/卡顿）。
fragment float4 optics_cono_gem_display_fragment(
        VertexOut in [[stage_in]],
        constant Uniforms &u [[buffer(1)]],
        texture2d<float> accumCurr [[texture(1)]]) {
    constexpr sampler conoLinear(filter::linear, address::clamp_to_edge);
    float2 centered = (in.uv - 0.5) * 2.0;
    centered.x *= u.resolution.x / max(u.resolution.y, 1.0);
    float rr = length(centered);
    float3 acc = accumCurr.sample(conoLinear, in.uv).rgb;
    float3 display = clamp(
        (acc * (2.51 * acc + 0.03)) / (acc * (2.43 * acc + 0.59) + 0.14),
        0.0, 1.0);
    display = pow(max(display, float3(0.0)), float3(1.0 / 2.2));
    if (rr > 1.0) display = float3(0.0);
    return float4(display, 1.0);
}

// MARK: - 偏光实验室 · 正交偏光 3D（Jones 强度沿光谱）
//
//  起偏镜 P 与分析镜 A 均垂直观察轴；入射振幅投影到晶体两特征偏振
//  （传播方向垂直平面内吸收二次型的 2×2 本征向量），两通道各自 Beer–Lambert
//  衰减后经分析镜 Malus 投影。非相干宽谱近似（无交叉相位项）：
//    I = T1(P·v1)²(A·v1)² + T2(P·v2)²(A·v2)²
//  正交时 ∝ ¼sin²2θ(T1+T2) → 非均质四明四暗、均质全暗；
//  单偏光（β=0）转动载物台在 o/e 通道间切换 → 多色性。
//  一期 IOR 不分裂（无双影），Fresnel/几何沿用平均折射率；特征基按入射
//  内方向建立后全程固定（教学近似，斜视角下呈现 e 光多色性）。

/// 垂直 d 的任意右手正交基
void orthoBasis(float3 d, thread float3 &u, thread float3 &v) {
    float3 ref = fabs(d.z) < 0.9 ? float3(0, 0, 1) : float3(1, 0, 0);
    u = normalize(cross(ref, d));
    v = normalize(cross(d, u));
}

// 前向声明：宝石姿态四元数运算定义于「宝石姿态」章节（本文件后部，与 Swift 侧
// PolarGemPose 逐字同式）。indicatrixFrame 用它做内层姿态复合，故先声明。
float3 quatRotate(float4 q, float3 v);
bool quatIsIdentity(float4 q);

/// 光率体在宝石静止系中的定向：z' = 光轴（一轴晶 c）/锐角平分线 Bxa（二轴晶），
/// 可相对台面法线倾斜 τ（方位 α、光轴面滚转 roll）——真实宝石极少严格定向。
/// 返回三主轴 ax[0..2] 与二轴晶两光轴 m1/m2。
///
/// 两个姿态旋钮都在这里合成，顺序**不可交换**（真值来源见 GemPose 头注释）：
///
///     R(世界←宝石) = Rz(stage) · T
///
///   · `tumble` = T：宝石在台面系里的姿态（手指竖直拖拽扳的）。台子本身不动，
///     所以它在**内层** —— 先让光率体随宝石歪，再随台公转。
///   · `stage`：样品总方位角 = 载物台角 + 宝石自转（specimenSpin）。在**外层**：
///     真台子是宝石随台公转，不是「绕一根已经被扳歪的轴转台」。
///     顺序写反（T·Rz(stage)）的症状是转台时宝石翻着往外飘，而台角读数、
///     盘面刻度、消光位徽章**全都还是对的** —— 没有任何断言看得见。
///
/// 两种调用姿势（传错会静默把姿态算两遍，画面「转一下顶两下」，不报错）：
///   · 3D 正交偏光（traceGemPolar）：进来的坐标已经**在宝石系**（GemPose 已把 T
///     与 stage 都吸收掉）→ 传 stage=0、tumble=kIdentityQuat。
///   · 真实锥光（traceSlabPolar）：坐标在世界系 → 传 specimenSpin(u) 与 u.gemTumble。
void indicatrixFrame(constant Uniforms &u, float stage, float4 tumble,
                     thread float3 ax[3],
                     thread float3 &m1, thread float3 &m2) {
    float tau = u.crystalOrient.x, alpha = u.crystalOrient.y, roll = u.crystalOrient.z;
    float sT = sin(tau), cT = cos(tau);
    float3 c = float3(sT * cos(alpha), sT * sin(alpha), cT);
    float3 e = fabs(c.z) < 0.9999 ? normalize(cross(float3(0, 0, 1), c))
                                  : float3(1, 0, 0);
    float3 g = normalize(cross(c, e));
    float3 xp = cos(roll) * e + sin(roll) * g;
    float3 yp = normalize(cross(c, xp));
    ax[0] = xp; ax[1] = yp; ax[2] = c;
    float V = u.crystalB.w;
    m1 = cos(V) * c + sin(V) * xp;
    m2 = cos(V) * c - sin(V) * xp;
    // 内层：宝石姿态 T（宝石静止系 → 台面系）。恒等姿态整支跳过 ——
    // 这样正交 3D 与理想锥光（都传 kIdentityQuat / 不经过这里）逐字不变，
    // 不会因为这次改动挪动任何一个既有验收像素。
    if (!quatIsIdentity(tumble)) {
        ax[0] = quatRotate(tumble, ax[0]);
        ax[1] = quatRotate(tumble, ax[1]);
        ax[2] = quatRotate(tumble, ax[2]);
        m1 = quatRotate(tumble, m1);
        m2 = quatRotate(tumble, m2);
    }
    if (stage != 0.0) {
        float cs = cos(stage), sn = sin(stage);
        for (int i = 0; i < 3; i += 1) {
            ax[i] = float3(cs * ax[i].x - sn * ax[i].y,
                           sn * ax[i].x + cs * ax[i].y, ax[i].z);
        }
        m1 = float3(cs * m1.x - sn * m1.y, sn * m1.x + cs * m1.y, m1.z);
        m2 = float3(cs * m2.x - sn * m2.y, sn * m2.x + cs * m2.y, m2.z);
    }
}

/// 非均质体在波向 dir 上的双折射介电本征偏振（一轴 o/e；二轴 Biots 平分线）。
/// 均质体或沿轴退化时返回 false，由调用方沿用既有偏振方向。
bool polarDielectricBasis(constant Uniforms &u, float3 dir, float stage, float4 tumble,
                          thread float3 &e1, thread float3 &e2) {
    int kind = int(round(u.crystalA.x));
    float3 cax[3]; float3 m1, m2;
    indicatrixFrame(u, stage, tumble, cax, m1, m2);
    if (kind == KIND_UNI_POS || kind == KIND_UNI_NEG) {
        float3 eo = cross(cax[2], dir);
        if (length(eo) <= 1e-3) return false;
        e2 = normalize(eo);
        e1 = normalize(cross(dir, e2));
        return true;
    }
    if (kind == KIND_BI_POS || kind == KIND_BI_NEG) {
        float3 q1 = m1 - dir * dot(m1, dir);
        float3 q2 = m2 - dir * dot(m2, dir);
        float l1 = length(q1), l2 = length(q2);
        if (l1 <= 1e-3 || l2 <= 1e-3) return false;
        float3 bis = q1 / l1 + q2 / l2;
        if (length(bis) <= 1e-3) {
            // 波向正对 Bxa：两投影严格相消，本征方向退化为光轴面内/外的 ax[0]/ax[1]
            e1 = cax[1];
            e2 = cax[0];
            return true;
        }
        e1 = normalize(bis);
        e2 = normalize(cross(dir, e1));
        return true;
    }
    return false;
}

// MARK: - 偏振密度矩阵（2×2 Hermitian）
// 部分相干光无法用单一 Jones 矢量描述。用 ρ=[[d0,z],[z*,d1]]（4 个实数）跟踪：
// d0/d1 两正交偏振模的能量占比，z 为互相干。钠光 D 双线相干长度仅 ~0.58mm，
// 超过的双折射程差使条纹洗消，只保留按刻面方向的明暗（真实偏光仪观感）。

/// 实坐标变换 ρ' = M ρ Mᵀ（本征基 ↔ s/p 基）
void dmRealTransform(thread float &d0, thread float &d1v,
                     thread float &zr, thread float &zi,
                     float m00, float m01, float m10, float m11) {
    float nd0 = m00 * m00 * d0 + m01 * m01 * d1v + 2.0 * m00 * m01 * zr;
    float nd1 = m10 * m10 * d0 + m11 * m11 * d1v + 2.0 * m10 * m11 * zr;
    float nzr = m00 * m10 * d0 + m01 * m11 * d1v + (m00 * m11 + m01 * m10) * zr;
    float nzi = (m00 * m11 - m01 * m10) * zi;
    d0 = nd0; d1v = nd1; zr = nzr; zi = nzi;
}

/// 对角复 Jones 作用 ρ' = J ρ J†，J = diag(a·e^{iα}, b·e^{iβ})
void dmDiagJones(thread float &d0, thread float &d1v,
                 thread float &zr, thread float &zi,
                 float a, float alpha, float b, float beta) {
    d0 *= a * a; d1v *= b * b;
    float cr = cos(alpha - beta), sr = sin(alpha - beta);
    float ozr = zr, ozi = zi;
    zr = a * b * (cr * ozr - sr * ozi);
    zi = a * b * (sr * ozr + cr * ozi);
}

/// |sinc(x)| = |sin x / x|
float sincAbs(float x) {
    float ax = fabs(x);
    return ax < 1e-6 ? 1.0 : fabs(sin(ax) / ax);
}

// MARK: - 宝石姿态（世界系 ⇄ 宝石静止系）
//
// 与 Swift 侧 PolarGemPose 逐字同式，两边**必须一起改**：只改一边的症状是
// 「光轴紫虚线 / 干涉图与宝石几何各自转各自的」，编译与断言都不会响。

/// 单位四元数 (ix,iy,iz,r) 旋转向量：v' = v + 2u×(u×v + r·v)
float3 quatRotate(float4 q, float3 v) {
    float3 u = q.xyz;
    return v + 2.0 * cross(u, cross(u, v) + q.w * v);
}

/// 恒等姿态四元数：姿态已由别处吸收（3D 正交偏光的 GemPose）或该路径不吃姿态
/// （理想锥光）时传它 —— 见 indicatrixFrame 定义处的「两种调用姿势」。
constant float4 kIdentityQuat = float4(0.0, 0.0, 0.0, 1.0);

/// 是否恒等姿态。全零四元数按恒等处理：quatRotate 遇到全零会把向量塌成 0
/// （光率体缩成一点、画面全黑，且不报任何错），必须在入口挡住。
/// 轴分量阈值 1e-12（相对模长）对应 ~1e-6 rad ≈ 0.00006°，远小于手指一步
/// （0.9°/pt），只把「根本没扳过」与「扳过」分开，不会吞掉真实姿态。
bool quatIsIdentity(float4 q) {
    float n2 = dot(q, q);
    if (n2 < 1e-6) return true;
    return dot(q.xyz, q.xyz) < n2 * 1e-12;
}

/// 宝石姿态：R(世界←宝石) = Rz(stage) · T，T = gemTumble（宝石相对台面系的姿态）。
/// 台角在外层：真台子是宝石随台公转；若换成 Rz 在内层，转台时宝石的公转轴会
/// 变成倾斜后的「台面法线」，读数与刻度却全都还是对的——只有画面不对。
struct GemPose {
    float4 q;     // T
    float4 qInv;  // T⁻¹
    float cs, sn; // cos/sin(台角)

    /// 世界系 → 宝石静止系：先退台角，再取倾转的逆
    float3 toGem(float3 v) {
        float3 t = float3(cs * v.x + sn * v.y, -sn * v.x + cs * v.y, v.z);
        return quatRotate(qInv, t);
    }

    /// 宝石静止系 → 世界系
    float3 toWorld(float3 v) {
        float3 t = quatRotate(q, v);
        return float3(cs * t.x - sn * t.y, sn * t.x + cs * t.y, t.z);
    }
};

GemPose makeGemPose(constant Uniforms &u) {
    GemPose p;
    // 非单位 / 全零四元数一律退化为 identity：全零会让整个宝石塌进原点
    // （画面全黑，且不报任何错），退化比崩溃更难查，所以在这里挡住。
    float4 q = u.gemTumble;
    float n2 = dot(q, q);
    p.q = n2 < 1e-6 ? float4(0.0, 0.0, 0.0, 1.0) : q * rsqrt(n2);
    p.qInv = float4(-p.q.xyz, p.q.w);
    // 样品总方位角 = 载物台角（仪器读数）+ 宝石自转（手指水平拖）。
    // 只取 polarParams.z 时，水平拖后宝石几何转了、光轴紫虚线不动（标记脱宝石）。
    float stage = specimenSpin(u);
    p.cs = cos(stage);
    p.sn = sin(stage);
    return p;
}

void traceGemPolar(constant Uniforms &u,
                   constant float4 *planeTable,
                   constant float4 *spectral,
                   constant float4 *rotTable,
                   float3 rayOrigin, float3 rayDirection,
                   float ior,
                   int binBegin, int binCount,
                   thread float *radiance,
                   thread bool &missed,
                   int meshNodeCount,
                   constant float4 *meshNodes,
                   constant float4 *meshTris,
                   bool lowerExitOnly) {
    // 宝石姿态 = 载物台绕世界 z 转（stage）× 宝石在台面系里的倾转（gemTumble）。
    // 把世界系光线/偏光轴变换到宝石静止系做几何与晶体光学计算；查询环境时方向
    // 再转回世界系。网格几何、光率体、光轴标记走的是**同一个** pose ——
    // 这正是「扳宝石」相对「转相机」的关键：以前倾转只转光率体不转网格，
    // 会出现「光轴转了晶体不转」的割裂。刻面亮斑随台转动闪烁，重现四明四暗。
    GemPose pose = makeGemPose(u);
    rayOrigin = pose.toGem(rayOrigin);
    rayDirection = pose.toGem(rayDirection);

    float nearT, farT;
    float3 entryNormal;
    if (meshNodeCount > 0) {
        // 原矿晶形：网格入射。网格存外向法线（SMORF 面法向参考定向，与
        // traceMeshSpectral 同约定），从体外入射 dot(d,n)<0 自然成立。
        if (!intersectMeshBVH(meshNodeCount, meshNodes, meshTris,
                              rayOrigin, rayDirection, nearT, entryNormal)
            || nearT < 0.0) { missed = true; return; }
    } else if (!intersectConvex(u, planeTable, rayOrigin, rayDirection, nearT, farT, entryNormal)
        || nearT < 0.0) { missed = true; return; }

    int iBase = illumBase(u);
    float3 entryPoint = rayOrigin + rayDirection * nearT;
    float3 insideDirection = refract(rayDirection, entryNormal, 1.0 / ior);
    if (dot(insideDirection, insideDirection) < 1e-7) { missed = false; return; }

    // 起偏/分析镜固定于实验室系（各自独立绝对方位），变换到宝石系。
    // 所有界面按 s/p 振幅 Fresnel 变换：s 向量 = k×N，p 在入射面内 ⟂k。
    float3 uB, vB;
    orthoBasis(insideDirection, uB, vB);
    float beta = u.polarParams.y;
    float pAng = u.polarLab.x;
    bool midOn = u.polarLab.z > 0.5;
    float tAng = u.polarLab.y;
    float3 Pworld = pose.toGem(float3(cos(pAng), sin(pAng), 0));
    float3 Aworld = pose.toGem(float3(cos(beta), sin(beta), 0));
    // 中偏光片固定在宝石与上偏光片（分析镜）之间——正交偏光 3D 的正向光路
    // 是 P（下方入射）→ 宝石 → T → A（上方出射投影）。
    float3 Tworld = pose.toGem(float3(cos(tAng), sin(tAng), 0));
    // 逆向光路：mode 1 从台面侧注入起偏态 P、出射时投影分析镜 A；
    // 真实锥光按光路互易性，台面侧（上方）是分析镜 A、亭部出射侧（下方）才是
    // 起偏镜 P——倾斜亭部刻面上两者投影面不同，接反会使黑十字偏转 45°。
    float3 entryAxis = lowerExitOnly ? Aworld : Pworld;
    // 空气中入射光经偏光片后的 E（⊥ 空气中入射方向）
    float3 pair = entryAxis - rayDirection * dot(entryAxis, rayDirection);
    if (length(pair) < 1e-4) pair = float3(0, 1, 0) - rayDirection * dot(float3(0, 1, 0), rayDirection);
    pair = normalize(pair);

    // 入射界面 s/p 基（近垂直入射时 s 退化，取垂直 k 的任意稳定方向）
    float3 s0 = cross(rayDirection, entryNormal);
    if (length(s0) < 1e-4) s0 = uB;
    s0 = normalize(s0);
    float3 p0 = normalize(cross(s0, rayDirection));
    float rs0, rp0, ts0, tp0, ps0, pp0;
    fresnelSP(1.0, ior, dot(-rayDirection, entryNormal), rs0, rp0, ts0, tp0, ps0, pp0);
    float es0 = dot(pair, s0), ep0 = dot(pair, p0);
    // 入射面直接反射（空气中）受分析镜约束
    float3 reflAir = reflect(rayDirection, entryNormal);
    float3 prAir = normalize(cross(s0, reflAir));
    float3 eRefl0 = rs0 * es0 * s0 + rp0 * ep0 * prAir;
    float entryLuminance = environmentLuminance(u, pose.toWorld(reflAir));
    float3 aRefl = Aworld - reflAir * dot(Aworld, reflAir);
    if (length(aRefl) < 1e-4) { float3 ux, vx; orthoBasis(reflAir, ux, vx); aRefl = vx; }
    aRefl = normalize(aRefl);
    float entryLeak = dot(eRefl0, aRefl) * dot(eRefl0, aRefl);
    // 折射入晶内的透射场（s/p 振幅透射系数不同；p 基随折射方向改变）
    float3 pt0 = normalize(cross(s0, insideDirection));
    float3 eTrans0 = ts0 * es0 * s0 + tp0 * ep0 * pt0;

    // 晶体三主轴（宝石静止系；载物台角与宝石倾转已由 GemPose 吸收，所以这里
    // 传 kIdentityQuat —— 传 u.gemTumble 会把姿态算两遍，画面上是「拖一下
    // 宝石转两倍」，同样不报错。crystalOrient 是另一个旋钮：光轴相对**晶体
    // 自身**的定向）
    float3 cax[3]; float3 biAxis1, biAxis2;
    indicatrixFrame(u, 0.0, kIdentityQuat, cax, biAxis1, biAxis2);

    // 初始两特征偏振方向：非均质体由双折射介电几何决定（无色品种吸收为零，
    // 不能靠吸收二次型定向）；均质体退化为 590nm 吸收二次型本征向量。
    int kind = int(round(u.crystalA.x));
    float3 e1, e2;
    if (!polarDielectricBasis(u, insideDirection, 0.0, kIdentityQuat, e1, e2)) {
        // 均质/沿轴退化：垂直传播方向平面内 590nm 吸收二次型的本征向量角 γ
        int refBin = 19;
        float b11 = 0, b22 = 0, b12 = 0;
        for (int j = 0; j < 3; j += 1) {
            float uj = dot(uB, cax[j]), vj = dot(vB, cax[j]);
            float aj = specAxisValue(spectral, j, refBin);
            b11 += uj * uj * aj; b22 += vj * vj * aj; b12 += uj * vj * aj;
        }
        // 二次型无择优轴（均质各向等吸收，或无色品种吸收全为 0）时
        // atan2(0,0) 在 GPU 上结果未定义（常为 NaN）→ γ 污染整条 Jones 链全暗；
        // 此时任意正交基都是本征基，γ 取 0。
        float anis = max(fabs(b11 - b22), 2.0 * fabs(b12));
        float gamma = anis < 1e-6 ? 0.0 : 0.5 * atan2(2.0 * b12, b11 - b22);
        e1 = cos(gamma) * uB + sin(gamma) * vB;
        e2 = -sin(gamma) * uB + cos(gamma) * vB;
    }
    // 初始 Jones 振幅：晶内透射场投影到本征基（ts/tp 已含能量透射）
    float p1 = dot(eTrans0, e1), p2 = dot(eTrans0, e2);

    // 偏振密度矩阵跟踪（本征基 e1/e2）：d1/d2 能量，z=xr+i·xi 互相干。
    // 段内 Beer 振幅吸收、双折射给两模相对相位；界面做 s/p 基变换与复 Fresnel。
    // 钠光 D 双线相干长度 ~0.58mm；白光宽谱更短（~0.012mm），条纹快速洗消。
    bool sodiumLight = u.flags.w == SODIUM_ILLUM_INDEX;
    float lcMm = sodiumLight ? 0.58 : 0.012;
    thread float d1[SPEC_BINS], d2[SPEC_BINS], xr[SPEC_BINS], xi[SPEC_BINS];
    for (int b = 0; b < binCount; b += 1) {
        int bin = binBegin + b;
        // 真实锥光方向像：台面上方为黑色镜筒/物镜，表面直接反射项记零
        radiance[bin] = lowerExitOnly ? 0.0
                                      : entryLuminance * specValue(spectral, iBase, bin) * entryLeak;
        d1[b] = p1 * p1; d2[b] = p2 * p2;
        xr[b] = p1 * p2; xi[b] = 0.0;
    }
    // 归一化几何单位 → mm（典型 8mm 腰围映射到 2 单位）
    const float mmPerUnit = 4.0;
    // 沿光路累计的双折射程差（mm，与波长无关）；出射时据此一次性施加相干包络
    float retardMm = 0.0;
    int rotCount = int(round(u.polarFlags.z));
    float3 insideOrigin = entryPoint + insideDirection * EPSILON;
    int maxBounces = u.flags.y;
    for (int bounce = 0; bounce < MAX_BOUNCES; bounce += 1) {
        if (bounce >= maxBounces) break;
        float boundaryT; float3 boundaryNormal;
        // 网格分支直接用存储的外向法线：凸体晶形从体内射向边界时
        // dot(d,n)>0（出射面），与 nextBoundary 的半空间语义一致。
        // 网格求交不做背面剔除——凹几何下若命中来侧，Jones 链按出射处理的
        // 退化与凸体路径把凹面当穿透的退化同级，晶形数据不出现这种构型。
        bool boundary = meshNodeCount > 0
            ? intersectMeshBVH(meshNodeCount, meshNodes, meshTris,
                               insideOrigin, insideDirection, boundaryT, boundaryNormal)
            : nextBoundary(u, planeTable, insideOrigin, insideDirection, boundaryT, boundaryNormal);
        if (!boundary) break;
        float3 boundaryPoint = insideOrigin + insideDirection * boundaryT;

        // 该光段的双折射有效折射率差 dn_eff（带光性符号；均质 ≡0）
        float dnEff = 0.0;
        if (kind == KIND_UNI_POS || kind == KIND_UNI_NEG) {
            // 一轴晶：波向与光轴 c（局部 z，载物台旋转不改变 z）夹角
            float cosA = clamp(fabs(dot(insideDirection, cax[2])), 0.0, 1.0);
            float sinA = sqrt(max(0.0, 1.0 - cosA * cosA));
            float nO = u.crystalA.y, nE = u.crystalA.z;
            float nEff = nO * nE / sqrt(max(1e-7,
                nE * nE * cosA * cosA + nO * nO * sinA * sinA));
            dnEff = nEff - nO;
        } else if (kind == KIND_BI_POS || kind == KIND_BI_NEG) {
            // 二轴晶：(γ−α)·sinθ1·sinθ2（定向光轴，倾斜光轴时同样成立）
            float s1 = sqrt(max(0.0, 1.0 - dot(insideDirection, biAxis1) * dot(insideDirection, biAxis1)));
            float s2 = sqrt(max(0.0, 1.0 - dot(insideDirection, biAxis2) * dot(insideDirection, biAxis2)));
            dnEff = (u.crystalB.z - u.crystalB.x) * s1 * s2;
        }

        // 段传播：Beer 振幅衰减；两模累积程差使互相干 z 旋转 2δ；累计总程差
        // （出射时再套有限相干包络，多段小程差时包络失效产生摩尔纹）。
        // 水晶旋光：沿光轴把线偏振面旋转 χ=ρ(λ)·L。离轴后线双折射迅速主导，
        // 本征模由圆偏振退化为线偏振，故旋光权重按圆/线双折射竞争因子
        // f=Δn_g/√(Δn_g²+Δn_eff²) 衰减（牛眼仅占光轴周围约 5° 视域）。
        float maxThroughput = 0.0;
        float segLmm = boundaryT * mmPerUnit;
        retardMm += dnEff * segLmm;
        bool rotActive = rotCount > 0 && (kind == KIND_UNI_POS || kind == KIND_UNI_NEG);
        float rotWeight3D = 0.0;
        if (rotActive && !lowerExitOnly) {
            float ca = clamp(fabs(dot(insideDirection, cax[2])), 0.0, 1.0);
            rotWeight3D = ca * ca;   // mode 1 透视观察沿用既有 cos²θ 近似
        }
        for (int b = 0; b < binCount; b += 1) {
            int bin = binBegin + b;
            float a1v = 0.0, a2v = 0.0;
            for (int j = 0; j < 3; j += 1) {
                float q1 = dot(e1, cax[j]), q2 = dot(e2, cax[j]);
                float aj = specAxisValue(spectral, j, bin);
                a1v += q1 * q1 * aj;
                a2v += q2 * q2 * aj;
            }
            float att1 = exp(-0.5 * max(a1v, 0.0) * boundaryT);
            float att2 = exp(-0.5 * max(a2v, 0.0) * boundaryT);
            d1[b] *= att1 * att1; d2[b] *= att2 * att2;
            xr[b] *= att1 * att2; xi[b] *= att1 * att2;
            // 两模相位 ±δ/2 → 互相干项相位 +2δ（δ=2π·dn·L(mm)·1e6/λ，此处 hd=δ/2）
            float wlNm = 400.0 + 10.0 * bin;
            float hd = M_PI_F * dnEff * segLmm * 1e6 / wlNm;
            float cr = cos(2.0 * hd), sr = sin(2.0 * hd);
            float ozr = xr[b], ozi = xi[b];
            xr[b] = cr * ozr - sr * ozi;
            xi[b] = sr * ozr + cr * ozi;
            // 旋光：本征基内实旋转 χ。真实锥光用圆/线双折射竞争因子；mode 1 用 cos²θ
            if (rotActive) {
                float rate = conoRotationRate(rotTable, rotCount, wlNm);
                float rotW = rotWeight3D;
                if (lowerExitOnly) {
                    // 圆双折射 Δn_g=ρ(°/mm)·λ(nm)/180e6；离轴线双折射主导时 f→0
                    float dnG = fabs(rate) * wlNm / 1.8e8;
                    rotW = dnG / sqrt(dnG * dnG + dnEff * dnEff);
                }
                float chi = rate * (M_PI_F / 180.0) * segLmm * rotW;
                float cc = cos(chi), ss = sin(chi);
                float rd0 = d1[b], rd1 = d2[b], rzr = xr[b], rzi = xi[b];
                dmRealTransform(rd0, rd1, rzr, rzi, cc, -ss, ss, cc);
                d1[b] = rd0; d2[b] = rd1; xr[b] = rzr; xi[b] = rzi;
            }
            maxThroughput = max(maxThroughput, max(d1[b], d2[b]));
        }

        // 界面 s/p 振幅 Fresnel（含 TIR 相位差）
        float3 sB = cross(insideDirection, boundaryNormal);
        if (length(sB) < 1e-4) sB = uB;
        sB = normalize(sB);
        float3 piB = normalize(cross(sB, insideDirection));
        float rsB, rpB, tsB, tpB, psB, ppB;
        fresnelSP(ior, 1.0, dot(insideDirection, boundaryNormal),
                  rsB, rpB, tsB, tpB, psB, ppB);
        float reflectR = 0.5 * (rsB * rsB + rpB * rpB);
        float3 exitDirection = refract(insideDirection, -boundaryNormal, ior);
        // 本征基 → 界面 s/p 基的实变换矩阵（与波长无关，循环外算一次）
        float m00 = dot(sB, e1),  m01 = dot(sB, e2);
        float m10 = dot(piB, e1), m11 = dot(piB, e2);
        // 真实锥光方向像：仅接受从亭部出射到下方偏光漫射半球的折射支
        bool acceptExit = (tsB + tpB > 0.0) && dot(exitDirection, exitDirection) > 1e-7
                          && (!lowerExitOnly || exitDirection.z < -0.02);
        if (acceptExit) {
            // 出射：ρ 先变换到【晶内】(sB, piB) s/p 基，再乘振幅透射 diag(ts,tp)；
            // 透射后 s 方向不变、p 振幅沿空气侧 ptB，故分析镜投影的 p 分量读 ptB。
            // 切勿直接把 ρ 变换到空气侧 ptB——折射前晶内 p 模沿 piB，两基相差整个
            // 折射偏折角（亭面可达 30–60°），直接投影会拧转偏振方位，使 0°/90° 明暗反转。
            float3 ptB = normalize(cross(sB, exitDirection));
            // 锥光逆向光路的下方元件是起偏镜 P；mode 1 为分析镜 A
            float3 exitAxis = lowerExitOnly ? Pworld : Aworld;
            float3 aOut = exitAxis - exitDirection * dot(exitAxis, exitDirection);
            if (length(aOut) < 1e-4) { float3 ux, vx; orthoBasis(exitDirection, ux, vx); aOut = vx; }
            aOut = normalize(aOut);
            float ka_s = dot(aOut, sB), ka_p = dot(aOut, ptB);
            // 中偏光片 T（仅正向光路；宝石→T→A 紧邻，中间无旋光介质）：
            // 先投影到 T 的出射面内方向，紧邻的分析镜 A 再取 cos∠(A,T)。
            // 无中片时 kt=ka、midFac=1，退化为原 Malus 投影。T∥出射方向退化时
            // 对该出射支无偏振作用，跳过。
            float kt_s = ka_s, kt_p = ka_p, midFac = 1.0;
            if (!lowerExitOnly && midOn) {
                float3 tOut = Tworld - exitDirection * dot(Tworld, exitDirection);
                if (length(tOut) > 1e-4) {
                    tOut = normalize(tOut);
                    kt_s = dot(tOut, sB);
                    kt_p = dot(tOut, ptB);
                    midFac = ka_s * kt_s + ka_p * kt_p;
                }
            }
            // 凡穿过宝石的光都来自下方起偏镜后的亮漫射器（无论从哪个刻面出射：
            // 亭部 TIR 把光路折回冠部也是同一光源）；表面直接反射另采黑镜筒环境。
            const float stageDiffuser = 1.0;
            // 有限相干包络按累计总程差一次性施加：钠光 D 双线 Lc≈0.58mm，
            // 超过后条纹洗消为平均亮度，程差为 0 时严格 Malus（均质全暗）。
            float visTotal = sincAbs(M_PI_F * fabs(retardMm) / max(lcMm, 1e-6));
            for (int b = 0; b < binCount; b += 1) {
                int bin = binBegin + b;
                float sd0 = d1[b], sd1 = d2[b], szr = xr[b], szi = xi[b];
                dmRealTransform(sd0, sd1, szr, szi, m00, m01, m10, m11);
                sd0 *= tsB * tsB; sd1 *= tpB * tpB;
                szr *= tsB * tpB * visTotal; szi *= tsB * tpB * visTotal;
                float intensity = midFac * midFac
                    * (kt_s * kt_s * sd0 + kt_p * kt_p * sd1
                       + 2.0 * kt_s * kt_p * szr);
                radiance[bin] += intensity
                    * stageDiffuser * specValue(spectral, iBase, bin);
            }
        }

        if (maxThroughput * reflectR < 0.002) break;
        // 反射续传：ρ→s/p→复 Fresnel 反射（TIR 相位差使偏振椭圆化）→新本征基
        float3 reflDir = reflect(insideDirection, boundaryNormal);
        float3 prB = normalize(cross(sB, reflDir));
        float3 ne1, ne2;
        // 同 traceGemPolar 主路径：坐标已在宝石系（stage=0），姿态传恒等。
        if (!polarDielectricBasis(u, reflDir, 0.0, kIdentityQuat, ne1, ne2)) {
            float3 e1r = reflect(e1, boundaryNormal);
            float3 e2r = reflect(e2, boundaryNormal);
            ne1 = e1r - reflDir * dot(e1r, reflDir);
            if (length(ne1) < 1e-4) ne1 = e2r - reflDir * dot(e2r, reflDir);
            ne1 = normalize(ne1);
            ne2 = normalize(cross(reflDir, ne1));
        }
        float n00 = dot(ne1, sB),  n01 = dot(ne1, prB);
        float n10 = dot(ne2, sB),  n11 = dot(ne2, prB);
        for (int b = 0; b < binCount; b += 1) {
            float rd0 = d1[b], rd1 = d2[b], rzr = xr[b], rzi = xi[b];
            dmRealTransform(rd0, rd1, rzr, rzi, m00, m01, m10, m11);
            dmDiagJones(rd0, rd1, rzr, rzi, rsB, psB, rpB, ppB);
            dmRealTransform(rd0, rd1, rzr, rzi, n00, n01, n10, n11);
            d1[b] = rd0; d2[b] = rd1; xr[b] = rzr; xi[b] = rzi;
        }
        e1 = ne1; e2 = ne2;
        insideDirection = reflDir;
        insideOrigin = boundaryPoint + insideDirection * EPSILON;
    }
    missed = false;
}

float3 sceneBackground(constant Uniforms &u, float2 uv, float3 rayOrigin, float3 rayDirection) {
    float vertical = smoothstep(0.0, 1.0, uv.y);
    float3 color = u.backgroundAbsorption.xyz * mix(0.94, 1.035, vertical);
    if (rayDirection.z < -0.0001) {
        float floorT = (-1.08 - rayOrigin.z) / rayDirection.z;
        if (floorT > 0.0) {
            float3 floorPoint = rayOrigin + rayDirection * floorT;
            float shadow = exp(-2.0 * dot(floorPoint.xy, floorPoint.xy));
            color *= 1.0 - shadow * 0.105;
        }
    }
    float2 centered = uv - 0.5;
    color *= 1.0 - 0.055 * dot(centered, centered);
    return color;
}

/// 正交偏光 3D 的可见载物台：一块**恒水平**（盘面法线 = 世界 z = 台面法线）的圆盘，
/// 盘心在样品**竖直**正下方 z = −(gemR + 0.14)。盘面刻 12 条 30° 分度线与一根
/// 金色基准指针（随 polarParams.z 载物台角刚性转动），盘缘外有一颗焊在镜座上的
/// 索引点（**永不随台转**）。
///
/// ⚠️ 盘面法线**必须**是台面法线（世界 z），不能取相机视线 —— 十五期曾写成
/// 「⊥ 观察轴」，那会让盘的转轴变成视线：
///
///     宝石的姿态是 Rz(specimenSpin)，绕**世界 z** 转（真台子是绕竖直轴转的）；
///     盘却绕视线转。3D 相机默认仰角 0.42 rad ≈ 24°，两轴差 24° ——
///     于是「转台」时盘的刻度绕视线走、宝石绕竖直轴走，宝石相对盘面**滑动**。
///     读数（台角、倾角）、刻度、索引点、消光位徽章**全都还是对的**，只有画面
///     不对，且只在相机不竖直时才现形 —— 顶视下两轴重合，怎么改都看不出来。
///     这一条是用户报的现象：「旋转载物台时宝石要和载物台保持相对静止」。
///
/// 真实仪器也是恒水平的：物镜轴竖直、台面 ⊥ 它。所以「盘 ⊥ 观察轴」只在
/// **观察轴竖直**（追光/理想锥光用的顶视）时才与这里等价；其余视角下它等于
/// 把台子画成一块斜板。
///
/// 盘面方位 0° 取**实验室系 +x**，与偏光片方位角（polarParams.y / polarLab.xy，
/// 同为 0° = +x）共用一把尺 —— 真实偏光仪的台面刻度与偏光片读数就是同一参照。
/// 改用屏幕基（十五期）会让「台 45° 消光」这类对应关系随相机朝向失义。
///
/// 三条要守住的约定：
///  ① 盘面只读 polarParams.z（纯载物台角），**不读 polarLab.w**（宝石自转）——
///     手指水平拖转的是宝石，台面是台上的固定参照，刻度不该跟着动；
///  ② 盘面只随台角转、**不随宝石倾转**（gemTumble 不参与）——真台子是死的；
///  ③ 只在宝石未命中的像素绘制，宝石天然挡在盘前，无需深度比较。
float3 polarStageDisc(constant Uniforms &u, float gemR, float3 ro, float3 rd, float3 base) {
    // 圆盘几何随宝石归一化半径自适应：半径比腰围大一圈。
    // gemR 为网格顶点最大模长（meshParams.z）；凸体琢型给固定兜底 1.35。
    gemR = max(gemR, 0.6);
    const float STAGE_Z = -gemR - 0.14;
    const float STAGE_R = gemR * 1.55;
    if (rd.z >= -1e-4) return base;            // 盘面朝上：水平/仰视不画
    float t = (STAGE_Z - ro.z) / rd.z;
    if (t <= 0.0) return base;
    float3 p = ro + rd * t;
    float r = length(p.xy);
    if (r > STAGE_R * 1.18) return base;       // 盘座外回到背景
    float az = atan2(p.y, p.x);
    float stage = u.polarParams.z;

    // 镜座环带（盘缘外 0..18%）：暗金属，不随台转
    if (r > STAGE_R) {
        float q = (r - STAGE_R) / (STAGE_R * 0.18);
        return float3(0.025, 0.026, 0.030) * (1.15 - 0.35 * q);
    }

    // 盘面：透偏光片的漫射亮底（base 已含 Malus 明暗）+ 径向轻微压暗
    float rn = r / STAGE_R;
    float3 col = base * (0.86 - 0.22 * rn) + float3(0.010, 0.011, 0.014);
    // 盘缘亮环
    float rim = exp(-pow((r - STAGE_R) / 0.020, 2.0));
    col = mix(col, float3(0.58, 0.59, 0.62), rim * 0.85);

    // 盘面相对载物台静止系的方位角 [0, 2π)
    float rel = fmod(az - stage, 6.2831853);
    if (rel < 0.0) rel += 6.2831853;
    float dang = min(rel, 6.2831853 - rel);

    // 12 条 30° 分度刻线（盘缘内侧短刻线，随台转）
    float sector = rel * 6.0 / 3.14159265;     // 0..12
    float tickAng = abs(sector - round(sector)) * (3.14159265 / 6.0);
    if (tickAng < 0.012 && r > STAGE_R * 0.78 && r < STAGE_R * 0.96) {
        col *= 0.42;
    }
    // 金色基准指针：0.42R → 盘缘，随台转
    if (dang < 0.022 && r > STAGE_R * 0.42 && r < STAGE_R * 0.99) {
        col = mix(col, float3(0.88, 0.68, 0.24), 0.88);
    }
    // 镜座固定索引刻点：盘面方位 0°（实验室 +x）、盘缘外侧，不随台转
    float fixedErr = min(abs(az), 6.2831853 - abs(az));
    if (fixedErr < 0.05 && r > STAGE_R * 1.02 && r < STAGE_R * 1.15) {
        col = float3(0.92, 0.88, 0.72);
    }
    return col;
}

/// 偏光 3D 模式下的视域背景：正交偏光时背景同步消光（偏光仪暗域），
/// 平行偏光时全亮；宝石之外的区域不再显示明亮环境，避免与「均质全暗」混淆。
/// 正交 3D 再叠加可见载物台圆盘（polarStageDisc）。
float3 polarBackground(constant Uniforms &u, float gemR, float2 uv,
                       float3 rayOrigin, float3 rayDirection) {
    float3 color = sceneBackground(u, uv, rayOrigin, rayDirection);
    if (int(round(u.polarParams.x)) == POLAR_MODE_CROSSED3D) {
        float malus = cos(u.polarParams.y);
        color *= 0.04 + 0.96 * malus * malus;
        color = polarStageDisc(u, gemR, rayOrigin, rayDirection, color);
    }
    return color;
}

/// 背景线性辐射度 → 显示色：与宝石命中像素同一套 exposure + ACES + gamma 2.2
///（见 spectralToDisplay / CPU 参考 finalize）。miss 分支若直接返回线性值，
/// 背景与宝石就处在不同编码空间——渐变与阴影在 gamma 前后差一个数量级，
/// Metal↔CPU diff harness 会把这种偏差整幅咬出来。
float3 backgroundToDisplay(constant Uniforms &u, float3 linear) {
    float3 c = max(linear, float3(0.0)) * exp2(u.cameraForwardExposure.w);
    c = clamp((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14), 0.0, 1.0);
    return pow(max(c, float3(0.0)), float3(1.0 / 2.2));
}

fragment float4 optics_fragment(VertexOut in [[stage_in]],
                                constant Uniforms &u [[buffer(1)]],
                                constant float4 *planeTable [[buffer(2)]],
                                constant float4 *spectral [[buffer(3)]],
                                constant float4 *rotTable [[buffer(4)]],
                                constant float4 *meshParams [[buffer(7)]],
                                constant float4 *meshNodes [[buffer(8)]],
                                constant float4 *meshTris [[buffer(9)]],
                                constant float4 *meshEdges [[buffer(15)]]) {
    // 与 Web 版相同的焦平面映射（含宽高比、焦点偏移与平移）。
    float2 centered = (in.uv - 0.5) * 2.0;
    centered.x *= u.resolution.x / max(u.resolution.y, 1.0);
    centered.x += u.cameraUpFocal.w;   // focalOffset
    centered -= u.panParams.xy;        // pan
    float3 rayOrigin = u.cameraPositionScale.xyz;
    float3 rayDirection = normalize(
        u.cameraForwardExposure.xyz
        + u.cameraRightEnvRot.xyz * centered.x * u.cameraPositionScale.w
        + u.cameraUpFocal.xyz * centered.y * u.cameraPositionScale.w
    );

    float ior = u.bodyColorIor.w;
    float dispersion = u.observerDispersion.w;
    float redIor = max(1.001, ior - dispersion * 0.48);
    float blueIor = ior + dispersion * 0.52;

    // 31 bin 出射光谱；三族波长共享几何、各自一个 IOR（B/G 色散）
    thread float spectralRadiance[SPEC_BINS];
    for (int b = 0; b < SPEC_BINS; b += 1) spectralRadiance[b] = 0.0;
    bool missed = false;
    bool polarMode = int(round(u.polarParams.x)) == POLAR_MODE_CROSSED3D;
    // 网格模式（凹面支持）：meshParams = (meshMode, nodeCount, 0, 0)，
    // 由 OpticsRenderer 在 OpticsRenderState.mesh 非 nil 时写入。
    // 网格与偏光 3D 可叠加：traceGemPolar 在 meshNodeCount>0 时把几何求交
    // 换成 BVH（原矿晶形），Jones 演算照旧。锥光干涉图仍是理想平行晶片模型，
    // 与几何无关（独立 fragment，不经此分支）。
    bool meshMode = meshParams[0].x > 0.5 && meshParams[0].y > 0.5;
    int meshNodeCount = meshMode ? int(meshParams[0].y + 0.5) : 0;
    // 偏光载物台圆盘的基准半径：网格吃 meshParams.z（顶点最大模长），
    // 凸体琢型给固定兜底（与 0.34 取景尺度相称）
    float stageGemR = meshMode ? max(meshParams[0].z, 0.6) : 1.35;
    // 族划分：B 400–490（10 bin）/ G 500–570（8 bin）/ R 580–700（13 bin）
    if (polarMode) {
        traceGemPolar(u, planeTable, spectral, rotTable, rayOrigin, rayDirection,
                      blueIor, 0, 10, spectralRadiance, missed,
                      meshNodeCount, meshNodes, meshTris);
        if (missed) return float4(backgroundToDisplay(
            u, polarBackground(u, stageGemR, in.uv, rayOrigin, rayDirection)), 1.0);
        bool missedG2 = false, missedR2 = false;
        traceGemPolar(u, planeTable, spectral, rotTable, rayOrigin, rayDirection,
                      ior, 10, 8, spectralRadiance, missedG2,
                      meshNodeCount, meshNodes, meshTris);
        traceGemPolar(u, planeTable, spectral, rotTable, rayOrigin, rayDirection,
                      redIor, 18, 13, spectralRadiance, missedR2,
                      meshNodeCount, meshNodes, meshTris);
    } else if (meshMode) {
        // 磨砂遮罩强度：0=通透（默认），1=磨砂。搭在 bodyColorIor.x——
        // 体色升级为光谱吸收后 xyz 三个槽位一直空着，不值得为它扩 320B 布局
        // （布局断言与 CPU 参考 harness 都押在这个数上）
        float frost = clamp(u.bodyColorIor.x, 0.0, 1.0);
        // —— 白模（教学晶体模型）：不透光、无折射，首命中直接漫反射着色 ——
        // 搭在停用的 absorptionAxisX 槽位：xyz = 白模反照率（主题色），w = 开关。
        // 全 shader 无处读 u.absorptionAxis*（RGB 回退路径已停用），复用零风险；
        // polar 路径同样不读，因此互斥成立。开白模时 frost 无意义（白模无体色可加深）。
        float whiteFlag = u.absorptionAxisX.w;
        if (whiteFlag > 0.5) {
            float wmT;
            float3 wmN;
            if (intersectMeshBVH(int(meshParams[0].y + 0.5), meshNodes, meshTris,
                                 rayOrigin, rayDirection, wmT, wmN)) {
                // 背面命中（射线从缝隙进入等）把法向翻向相机，避免暗面
                float3 nn = dot(wmN, rayDirection) < 0.0 ? wmN : -wmN;
                // 双灯漫反射 + 高光 + 边缘光：与晶胞模式同一套灯向，
                // 读起来是"同一间摄影棚里的教学模型"，而不是另一套渲染。
                // 2026-10-03 用户反馈米勒指数白模太暗淡：抬双灯权重与提亮
                // 底座（0.5+diff*0.82 → 0.60+diff*0.95），亮面截断在 1.0。
                float3 key = normalize(float3(0.55, 0.35, 0.76));
                float3 fill = normalize(float3(-0.6, -0.2, -0.5));
                float diff = max(dot(nn, key), 0.0) * 0.80
                           + max(dot(nn, fill), 0.0) * 0.26;
                float3 half0 = normalize(key - rayDirection);
                float spec = pow(max(dot(nn, half0), 0.0), 40.0) * 0.22;
                float rim = pow(1.0 - max(dot(nn, -rayDirection), 0.0), 3.0) * 0.16;
                float3 albedo = clamp(u.absorptionAxisX.xyz, float3(0.0), float3(1.0));
                float3 wmColor = albedo * (0.60 + diff * 0.95) + spec + rim;
                // 黑色晶棱**实线**（几何级）：CPU 预计算的真晶棱表
                // （扇形三角化对角线两侧法向一致被邻接判定排除，只留真二面角
                // 棱），逐像素对命中点做点-线段距离，与像素投影像尺寸比较得
                // ~1px 最细抗锯齿实线（2026-10-03 用户两轮反馈：先是导数法
                // 1px 抖成虚线，改几何实线 2px 后要求压回最细）。
                float3 hitP = rayOrigin + rayDirection * wmT;
                float pixWorld = wmT * (length(dfdx(rayDirection))
                                      + length(dfdy(rayDirection)));
                float dEdge = 1e19;
                int edgeCount = int(meshParams[0].w + 0.5);
                for (int e = 0; e < edgeCount; e += 1) {
                    float3 ea = meshEdges[e * 2].xyz;
                    float3 eab = meshEdges[e * 2 + 1].xyz - ea;
                    float tt = clamp(dot(hitP - ea, eab) / max(dot(eab, eab), 1e-12),
                                     0.0, 1.0);
                    dEdge = min(dEdge, distance(hitP, ea + eab * tt));
                }
                float edgeLine = 1.0 - smoothstep(0.25 * pixWorld, 0.85 * pixWorld, dEdge);
                // 棱线调灰（2026-10-03 用户反馈黑色仍太显眼）：中灰压暗即可，
                // 不吃掉晶面本身的明度对比
                wmColor = mix(wmColor, float3(0.42, 0.42, 0.44), edgeLine);
                return float4(clamp(wmColor, 0.0, 1.0), 1.0);
            }
            return float4(backgroundToDisplay(
                u, polarBackground(u, stageGemR, in.uv, rayOrigin, rayDirection)), 1.0);
        }
        // 凹面走网格路径：三条 IOR 族照常共享几何、合并为完整 31 bin 光谱
        traceMeshSpectral(u, spectral, int(meshParams[0].y + 0.5), meshNodes, meshTris,
                          rayOrigin, rayDirection, blueIor, frost, 0, 10, spectralRadiance, missed);
        if (missed) return float4(backgroundToDisplay(
            u, polarBackground(u, stageGemR, in.uv, rayOrigin, rayDirection)), 1.0);
        traceMeshSpectral(u, spectral, int(meshParams[0].y + 0.5), meshNodes, meshTris,
                          rayOrigin, rayDirection, ior, frost, 10, 8, spectralRadiance, missed);
        traceMeshSpectral(u, spectral, int(meshParams[0].y + 0.5), meshNodes, meshTris,
                          rayOrigin, rayDirection, redIor, frost, 18, 13, spectralRadiance, missed);
    } else {
        traceGemSpectral(u, planeTable, spectral, rayOrigin, rayDirection,
                         blueIor, 0, 10, spectralRadiance, missed);
        if (missed) {
            return float4(backgroundToDisplay(
                u, polarBackground(u, stageGemR, in.uv, rayOrigin, rayDirection)), 1.0);
        }
        bool missedG = false;
        bool missedR = false;
        traceGemSpectral(u, planeTable, spectral, rayOrigin, rayDirection,
                         ior, 10, 8, spectralRadiance, missedG);
        traceGemSpectral(u, planeTable, spectral, rayOrigin, rayDirection,
                         redIor, 18, 13, spectralRadiance, missedR);
    }

    float3 color;
    if (u.flags.w == SODIUM_ILLUM_INDEX) {
        // 钠光单色照明：不做色适应，按 589.3nm CIE 色度（Y 归一）映射明视觉亮度
        float Y = 0.0;
        for (int b = 0; b < SPEC_BINS; b += 1) Y += kCMF[b].y * spectralRadiance[b];
        color = kXYZToRGB * float3(1.325, 1.0, 0.0026) * Y;
    } else {
        // CIE 1931 三刺激积分（照明体白场归一 = 完全色适应）
        int whiteBase = illumBase(u);
        float X = 0.0, Y = 0.0, Z = 0.0;
        float Xw = 0.0, Yw = 0.0, Zw = 0.0;
        for (int b = 0; b < SPEC_BINS; b += 1) {
            float s = specValue(spectral, whiteBase, b);
            X += kCMF[b].x * spectralRadiance[b];
            Y += kCMF[b].y * spectralRadiance[b];
            Z += kCMF[b].z * spectralRadiance[b];
            Xw += kCMF[b].x * s;
            Yw += kCMF[b].y * s;
            Zw += kCMF[b].z * s;
        }
        float3 xyz = float3(X / max(Xw, 1e-6), Y / max(Yw, 1e-6), Z / max(Zw, 1e-6));
        color = kXYZToRGB * xyz;
    }
    color = max(color, float3(0.0));
    color *= exp2(u.cameraForwardExposure.w);
    // ACES filmic 近似
    color = clamp((color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14), 0.0, 1.0);
    color = pow(max(color, float3(0.0)), float3(1.0 / 2.2));
    return float4(color, 1.0);
}

// MARK: - 量子实验室（单光子的完整历史）
//
//  与上面的路径追踪共用相机/几何/光谱/色调映射，只多两件事：
//   - 晶体按半透明（"X 光"）呈现，好让体内的光子轨迹看得见；
//   - 轨迹本身由 CPU 侧的 PhotonSimulator 算好（段缓冲上传），
//     着色器只负责画：o/e 两条相干支路（各自衰减、带相位行波）、
//     吸收闪光、声子涟漪、亚稳态脉冲、自发辐射的新光子。
//  分工是刻意的：**物理在可断言的 Swift 层，着色器只做表现**。

#define QUANTUM_MAX_SEGMENTS 128

/// 轨迹段：a→b 一条发光胶囊；kind 决定配色与是否虚化
struct QuantumSegment {
    float4 a;    // xyz 起点，w 类型（0 入射 / 1 o 支路 / 2 e 支路 / 3 幽灵分支 / 4 发射 / 5 反射·散射标记）
    float4 b;    // xyz 终点，w 强度
    float4 tint; // rgb 颜色，w 线宽(px)
    float4 mod;  // x 行波相位密度(周期/px)，y 虚线开关，z 辉光半径(px)，w 备用
};

struct QuantumUniforms {
    float4 q0;      // x 动画进度 0...1，y 时间(秒)，z 段数，w 标记数
    float4 q1;      // x 波长(nm)，y 显示幽灵分支，z 辉光总强度，w 晶体不透明度
    // 自发辐射的显示色：必须跟着**发射波长**走（红宝石 R 线是深红、钻石 N3 是蓝），
    // 不能写死成红色 —— 否则"蓝荧光"看起来像红荧光。
    float4 emissionTint;
    float4 mk[4];   // 标记：xyz 位置，w 类型（1 吸收闪光 / 2 声子涟漪 / 3 亚稳态 / 4 发射 / 5 热扩散）
    float4 mp[4];   // 标记参数：x 半径(px)，y 强度，z 相位/进度，w 备用
};

/// 31 bin 辐射度 → 线性 sRGB（与 optics_fragment 末段同口径：白场归一 + ACES + gamma 2.2）
float3 quantumToneMap(constant Uniforms &u, constant float4 *spectral, thread float *radiance) {
    float3 color;
    if (u.flags.w == SODIUM_ILLUM_INDEX) {
        float Y = 0.0;
        for (int b = 0; b < SPEC_BINS; b += 1) Y += kCMF[b].y * radiance[b];
        color = kXYZToRGB * float3(1.325, 1.0, 0.0026) * Y;
    } else {
        int whiteBase = illumBase(u);
        float X = 0.0, Y = 0.0, Z = 0.0;
        float Xw = 0.0, Yw = 0.0, Zw = 0.0;
        for (int b = 0; b < SPEC_BINS; b += 1) {
            float s = specValue(spectral, whiteBase, b);
            X += kCMF[b].x * radiance[b];
            Y += kCMF[b].y * radiance[b];
            Z += kCMF[b].z * radiance[b];
            Xw += kCMF[b].x * s;
            Yw += kCMF[b].y * s;
            Zw += kCMF[b].z * s;
        }
        color = kXYZToRGB * float3(X / max(Xw, 1e-6), Y / max(Yw, 1e-6), Z / max(Zw, 1e-6));
    }
    color = max(color, float3(0.0));
    color *= exp2(u.cameraForwardExposure.w);
    color = clamp((color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14), 0.0, 1.0);
    return pow(max(color, float3(0.0)), float3(1.0 / 2.2));
}

/// 世界点 → 屏幕像素坐标（与 optics_fragment 的射线构造严格互逆：
/// centered.x = dot(v,right)/(dot(v,forward)·scale)，再反解 uv）
bool quantumProject(constant Uniforms &u, float3 p, thread float2 &pixel) {
    float3 v = p - u.cameraPositionScale.xyz;
    float depth = dot(v, u.cameraForwardExposure.xyz);
    if (depth <= 0.01) return false;
    float scale = max(u.cameraPositionScale.w, 1e-6);
    float cx = dot(v, u.cameraRightEnvRot.xyz) / (depth * scale);
    float cy = dot(v, u.cameraUpFocal.xyz) / (depth * scale);
    float aspect = u.resolution.x / max(u.resolution.y, 1.0);
    float uvx = 0.5 + (cx - u.cameraUpFocal.w + u.panParams.x) / (2.0 * aspect);
    float uvy = 0.5 + (cy + u.panParams.y) / 2.0;
    pixel = float2(uvx * u.resolution.x, uvy * u.resolution.y);
    return true;
}

fragment float4 optics_quantum_fragment(VertexOut in [[stage_in]],
                                        constant Uniforms &u [[buffer(1)]],
                                        constant float4 *planeTable [[buffer(2)]],
                                        constant float4 *spectral [[buffer(3)]],
                                        constant QuantumUniforms &q [[buffer(5)]],
                                        constant QuantumSegment *segs [[buffer(6)]],
                                        constant float4 *meshParams [[buffer(7)]],
                                        constant float4 *meshNodes [[buffer(8)]],
                                        constant float4 *meshTris [[buffer(9)]]) {
    float2 centered = (in.uv - 0.5) * 2.0;
    centered.x *= u.resolution.x / max(u.resolution.y, 1.0);
    centered.x += u.cameraUpFocal.w;
    centered -= u.panParams.xy;
    float3 rayOrigin = u.cameraPositionScale.xyz;
    float3 rayDirection = normalize(
        u.cameraForwardExposure.xyz
        + u.cameraRightEnvRot.xyz * centered.x * u.cameraPositionScale.w
        + u.cameraUpFocal.xyz * centered.y * u.cameraPositionScale.w
    );

    // —— 背景 + 半透明晶体（体色仍由 31 bin 光谱路径追踪给出）——
    // 原矿晶形（meshParams 开启）时衬底走网格路径：frost 传 0——量子视图的
    // 晶体本来就是半透明"X 光"呈现，磨砂面散射增强没有意义。
    bool meshMode = meshParams[0].x > 0.5 && meshParams[0].y > 0.5;
    // 背景先转显示色（exposure+ACES+gamma）：下面的体色 quantumToneMap、
    // Fresnel 高光与轨迹辉光都在显示空间叠加，背景不能停在线性空间。
    float3 color = backgroundToDisplay(
        u, sceneBackground(u, in.uv, rayOrigin, rayDirection));
    thread float spectralRadiance[SPEC_BINS];
    for (int b = 0; b < SPEC_BINS; b += 1) spectralRadiance[b] = 0.0;
    bool missed = false;
    if (meshMode) {
        traceMeshSpectral(u, spectral, int(meshParams[0].y + 0.5), meshNodes, meshTris,
                          rayOrigin, rayDirection, u.bodyColorIor.w, 0.0,
                          0, SPEC_BINS, spectralRadiance, missed);
    } else {
        traceGemSpectral(u, planeTable, spectral, rayOrigin, rayDirection,
                         u.bodyColorIor.w, 0, SPEC_BINS, spectralRadiance, missed);
    }
    if (!missed) {
        float3 body = quantumToneMap(u, spectral, spectralRadiance);
        float alpha = clamp(q.q1.w, 0.0, 1.0);
        color = mix(color, body, alpha);
        // 晶面棱边：沿视线方向的镜面环境反射（玻璃光泽）
        float3 entryN = float3(0.0, 0.0, 1.0);
        float nearT = 0.0, farT = 0.0;
        bool edgeHit = meshMode
            ? intersectMeshBVH(int(meshParams[0].y + 0.5), meshNodes, meshTris,
                               rayOrigin, rayDirection, nearT, entryN)
            : intersectConvex(u, planeTable, rayOrigin, rayDirection, nearT, farT, entryN);
        if (edgeHit && nearT > 0.0) {
            float fres = dielectricFresnel(dot(-rayDirection, entryN), 1.0, u.bodyColorIor.w);
            color += fres * 0.26 * float3(1.0);
        }
    }

    // —— 光子轨迹 ——
    float2 pixel = in.uv * u.resolution;
    int segCount = int(min(q.q0.z, float(QUANTUM_MAX_SEGMENTS)));
    float showGhost = q.q1.y;
    float progress = clamp(q.q0.x, 0.0, 1.0);
    float time = q.q0.y;
    float glowGain = q.q1.z;

    for (int i = 0; i < segCount; i += 1) {
        QuantumSegment s = segs[i];
        int kind = int(round(s.a.w));
        if (kind == 3 && showGhost < 0.5) continue;
        float2 pa, pb;
        if (!quantumProject(u, s.a.xyz, pa) || !quantumProject(u, s.b.xyz, pb)) continue;
        float2 ba = pb - pa;
        float len2 = max(dot(ba, ba), 1e-6);
        float h = clamp(dot(pixel - pa, ba) / len2, 0.0, 1.0);
        float d = length(pixel - pa - ba * h);
        float width = max(s.tint.w, 0.6);
        float core = exp(-(d * d) / max(width * width, 0.25));
        float halo = exp(-(d * d) / max(s.mod.z * s.mod.z, 1.0)) * 0.45;
        float amp = s.b.w;
        // o/e 支路：沿段的行波显示相位累积（两模相位速度不同 → 视觉上错开）
        if (s.mod.x > 0.0) {
            float ph = 6.2831853 * (length(pa + ba * h) * s.mod.x) - time * 4.0;
            amp *= 0.55 + 0.45 * cos(ph);
        }
        if (s.mod.y > 0.5) {
            // 幽灵分支：虚线（未被探测器选中的概率幅分支）
            amp *= step(0.45, fract(length(pa + ba * h) * 0.06));
        }
        // 进度截断：动画只画到当前进度所在的那一段
        float3 tint = s.tint.rgb;
        float3 add = tint * (core + halo) * amp * glowGain;
        color += add;
        // 端点小球：让"光子在哪"看得出来
        float dEnd = min(length(pixel - pa), length(pixel - pb));
        color += tint * exp(-(dEnd * dEnd) / max(width * width * 2.25, 1.0)) * amp * 0.35 * glowGain;
    }

    // —— 事件标记 ——
    int markerCount = int(min(q.q0.w, 4.0));
    for (int m = 0; m < markerCount; m += 1) {
        float4 mk = q.mk[m];
        float4 mp = q.mp[m];
        int kind = int(round(mk.w));
        float2 c;
        if (!quantumProject(u, mk.xyz, c)) continue;
        float r = length(pixel - c);
        float radius = max(mp.x, 1.0);
        float amp = mp.y;
        if (kind == 1) {
            // 吸收闪光：一个亮核 + 一圈迅速扩散的激波
            color += float3(1.0, 0.86, 0.55) * exp(-(r * r) / max(radius * radius, 1.0)) * amp;
            float wave = exp(-pow(r - radius * 1.8, 2.0) / max(radius * 0.9, 4.0));
            color += float3(1.0, 0.72, 0.36) * wave * amp * 0.6;
        } else if (kind == 2) {
            // 声子涟漪：同心环，随时间向外扩散并衰减
            float phase = fract(mp.z);
            float ringR = radius * (0.35 + 1.65 * phase);
            float ring = exp(-pow(r - ringR, 2.0) / max(radius * 0.35, 4.0));
            color += float3(1.0, 0.62, 0.28) * ring * amp * (1.0 - phase);
        } else if (kind == 3) {
            // 亚稳态：缓慢脉冲（毫秒级停留的可视化）
            float pulse = 0.55 + 0.45 * sin(time * 5.0);
            color += float3(0.55, 0.95, 1.0) * exp(-(r * r) / max(radius * radius, 1.0)) * amp * pulse;
        } else if (kind == 4) {
            // 自发辐射：颜色取发射波长（q.emissionTint）
            color += q.emissionTint.rgb * exp(-(r * r) / max(radius * radius, 1.0)) * amp;
        } else if (kind == 5) {
            // 热扩散：一团慢慢淡开的暖色雾
            float fade = clamp(1.0 - mp.z, 0.0, 1.0);
            color += float3(0.95, 0.45, 0.22) * exp(-(r * r) / max(radius * radius * 2.5, 1.0)) * amp * fade;
        }
    }

    // 进度未到的部分整体压暗（视觉上的"故事讲到哪了"）
    color = mix(color * 0.82, color, clamp(progress * 1.5, 0.0, 1.0));
    return float4(min(color, float3(1.0)), 1.0);
}

// MARK: - 网格光学路径（凹面支持）
//
// 对齐上游 OpenGemCutting 2026-09-21 vgpu 后端（opticsShader.wgsl 的
// intersectMesh / traceMeshGem）：在凸体半空间路径之外的第二条几何路径，
// 任意闭合三角网格——保留凹槽、孔洞与分离组件，**永不做凸包**。
// 几何打包（节点 3 行 / 三角形 4 行）与 OpticsMesh.swift 逐字段一致；
// 遍历凭据（预序 + 逃生链接）两侧共用。
//
// 与凸体路径的本质差异：凸体内"光程"是单一区间，单条前向路径即可；
// 凹几何下"体内"是区间的并集，反射/折射分支必须各自重新求实际边界
// （空腔、气隙自然出现），所以走显式路径分裂栈而不是单一透射循环。

/// 无栈 BVH 遍历最近命中（Möller–Trumbore）。
/// 不做背面剔除——凹面光学出射与空腔需要每个边界的两侧；
/// 加了剔除的话内部出射会漏掉远壁，整颗晶体变黑洞。
bool intersectMeshBVH(int nodeCount,
                      constant float4 *nodeTable,
                      constant float4 *triTable,
                      float3 origin, float3 direction,
                      thread float &hitT, thread float3 &hitNormal) {
    hitT = 1e19;
    hitNormal = float3(0.0, 0.0, 1.0);
    int node = 0;
    while (node < nodeCount) {
        float4 mnRow = nodeTable[node * 3];       // (min.xyz, escape)
        float4 mxRow = nodeTable[node * 3 + 1];   // (max.xyz, start)
        float4 countRow = nodeTable[node * 3 + 2]; // (count, 0, 0, 0)
        // slab 包围盒（方向分量近零走轴区间判断，避免除零）
        float nearT = 0.0;
        float farT = hitT;
        bool overlap = true;
        for (int axis = 0; axis < 3; axis += 1) {
            float o = origin[axis], d = direction[axis];
            if (fabs(d) < 1e-12) {
                if (o < mnRow[axis] || o > mxRow[axis]) { overlap = false; break; }
            } else {
                float a = (mnRow[axis] - o) / d;
                float b = (mxRow[axis] - o) / d;
                nearT = max(nearT, min(a, b));
                farT = min(farT, max(a, b));
                if (nearT > farT) { overlap = false; break; }
            }
        }
        if (!overlap) { node = int(mnRow.w); continue; }   // 逃生链接：整段跳过子树
        int start = int(mxRow.w);
        int count = int(countRow.x);
        for (int index = start; index < start + count; index += 1) {
            float3 a = triTable[index * 4].xyz;
            float3 ab = triTable[index * 4 + 1].xyz;
            float3 ac = triTable[index * 4 + 2].xyz;
            float3 p = cross(direction, ac);
            float det = dot(ab, p);
            if (fabs(det) < 1e-12) continue;
            float3 rel = origin - a;
            float uu = dot(rel, p) / det;
            if (uu < -1e-6 || uu > 1.000001) continue;
            float3 q = cross(rel, ab);
            float vv = dot(direction, q) / det;
            if (vv < -1e-6 || uu + vv > 1.000001) continue;
            float distance = dot(ac, q) / det;
            if (distance > 1e-7 && distance < hitT) {
                hitT = distance;
                hitNormal = triTable[index * 4 + 3].xyz;
            }
        }
        node += 1;
    }
    return hitT < 1e18;
}

/// 网格路径分裂栈槽位（上游 9 槽 / 511 步迭代上限）。
/// throughput 按家族内最宽 bin 数（红族 13）预留。
struct MeshPath {
    float3 origin;
    float3 direction;
    float throughput[13];
    int depth;
    bool inside;
    bool first;   // 主射线标志：未命中 = 整像素走背景
};

/// 磨砂哈希：命中点 → [0,1) 伪随机。取世界系位置做种子——
/// 表面微结构在空间中固定，静止画面零时域噪声，转动时闪点跟着表面走
float frostHash1(float3 p) {
    return fract(sin(dot(p, float3(127.1, 311.7, 74.7))) * 43758.5453);
}

/// 网格版单族波长追踪：traceMeshGem（WGSL）的光谱升级版。
/// 与 traceGemSpectral 同口径——逐 bin 透射率、方向加权三主轴吸收、
/// 环境亮度出射；差异在几何求交与路径分裂（凹几何必须分支重探边界）。
/// 能量预算（depth ≥ maxBounces）内的未解决路径直接丢弃，
/// 不伪造"穿过另一组件"的环境出射。
///
/// frost > 0 为磨砂遮罩：每个命中点施加确定性微法向扰动（同一表面点
/// 一条微法线，反射/折射分支共用——物理上就是同一块微表面）。相邻像素
/// 命中不同微面 → 透射方向被打散，路径分裂栈天然把它们积分成乳白磨砂；
/// 同时吸收 ×(1+0.8·frost) 加深体色，呼应"磨砂更浑浊"的观感。
void traceMeshSpectral(constant Uniforms &u,
                       constant float4 *spectral,
                       int nodeCount,
                       constant float4 *nodeTable,
                       constant float4 *triTable,
                       float3 rayOrigin, float3 rayDirection,
                       float ior, float frost, int binBegin, int binCount,
                       thread float *radiance,
                       thread bool &missed) {
    thread MeshPath stack[9];
    int pending = 1;
    stack[0].origin = rayOrigin;
    stack[0].direction = rayDirection;
    for (int b = 0; b < binCount; b += 1) stack[0].throughput[b] = 1.0;
    stack[0].depth = 0;
    stack[0].inside = false;
    stack[0].first = true;

    int iBase = illumBase(u);
    for (int iter = 0; iter < 511 && pending > 0; iter += 1) {
        pending -= 1;
        MeshPath path = stack[pending];
        float hitT;
        float3 hitNormal;
        bool hit = intersectMeshBVH(nodeCount, nodeTable, triTable,
                                    path.origin, path.direction, hitT, hitNormal);
        if (!hit) {
            if (path.first) { missed = true; return; }
            float lum = environmentLuminance(u, path.direction);
            for (int b = 0; b < binCount; b += 1) {
                int bin = binBegin + b;
                radiance[bin] += path.throughput[b] * lum * specValue(spectral, iBase, bin);
            }
            continue;
        }
        path.first = false;
        // 能量预算：预算内未解决 = 丢弃（与上游 CPU 参考同语义）
        if (path.depth >= u.flags.y) continue;

        float3 hitPoint = path.origin + path.direction * hitT;
        // 体内段：该光段方向上的逐波长有效吸收（局部系对角张量；非偏振两偏振各半）。
        // 磨砂时吸收加深（×1+0.8·frost）——磨砂面散射损耗的体感就是"更浑浊"
        if (path.inside) {
            float3 w = 0.5 * (float3(1.0) - path.direction * path.direction);
            float absorbBoost = 1.0 + frost * 0.8;
            for (int b = 0; b < binCount; b += 1) {
                int bin = binBegin + b;
                float segmentAbsorption =
                    w.x * specValue(spectral, SPEC_AXIS_X_BASE, bin)
                  + w.y * specValue(spectral, SPEC_AXIS_Y_BASE, bin)
                  + w.z * specValue(spectral, SPEC_AXIS_Z_BASE, bin);
                path.throughput[b] *= exp(-max(segmentAbsorption, 0.0) * absorbBoost * hitT);
            }
        }

        bool entering = dot(path.direction, hitNormal) < 0.0;
        float3 incidentNormal = entering ? hitNormal : -hitNormal;
        // 磨砂微法向：扰动上限 ~29°（0.55 半径），远小于 90°——不会把
        // entering/inside 侧别算翻（侧别判定用未扰动几何法向，见上一行）
        if (frost > 0.0) {
            float3 j = float3(frostHash1(hitPoint + float3(1.7, 9.3, 4.1)),
                              frostHash1(hitPoint + float3(9.3, 4.1, 1.7)),
                              frostHash1(hitPoint + float3(4.1, 1.7, 9.3))) * 2.0 - 1.0;
            incidentNormal = normalize(incidentNormal + j * (frost * 0.55));
        }
        float fresnel = dielectricFresnel(dot(-path.direction, incidentNormal),
                                          entering ? 1.0 : ior,
                                          entering ? ior : 1.0);
        float maxThroughput = 0.0;
        for (int b = 0; b < binCount; b += 1) {
            maxThroughput = max(maxThroughput, path.throughput[b]);
        }
        // 反射分支：留在界面的来侧（进入→外部反射；出射→体内继续反弹）
        float reflectWeight = maxThroughput * fresnel;
        if (reflectWeight >= 0.002 && pending < 9) {
            MeshPath rp = path;
            rp.origin = hitPoint + incidentNormal * 2e-6;
            rp.direction = reflect(path.direction, incidentNormal);
            for (int b = 0; b < binCount; b += 1) rp.throughput[b] *= fresnel;
            rp.depth = path.depth + 1;
            rp.inside = !entering;
            rp.first = false;
            stack[pending] = rp;
            pending += 1;
        }
        // 折射分支：穿过界面（TIR 时 refract 返回零向量，无分支）
        float3 exitDirection = refract(path.direction, incidentNormal,
                                       entering ? 1.0 / ior : ior);
        float transmitWeight = maxThroughput * (1.0 - fresnel);
        if (dot(exitDirection, exitDirection) > 1e-7
            && transmitWeight >= 0.002 && pending < 9) {
            MeshPath tp = path;
            tp.origin = hitPoint - incidentNormal * 2e-6;
            tp.direction = exitDirection;
            for (int b = 0; b < binCount; b += 1) tp.throughput[b] *= (1.0 - fresnel);
            tp.depth = path.depth + 1;
            tp.inside = entering;
            tp.first = false;
            stack[pending] = tp;
            pending += 1;
        }
    }
    missed = false;
}

// MARK: - 晶胞模式（原子球 / 化学键 / 晶胞框 · 解析求交）
//
// 笔记晶胞查看器的几何在凸体/网格路径之外：原子是分散的 CPK 色球，
// 化学键与晶胞框是细胶囊。它们不参与折射路径追踪（不透明、无色散），
// 只需"全局最近命中"的解析求交 + 简单摄影棚双灯着色；
// 但射线生成、平移与背景复用同一套 Uniforms——仍然只有一套渲染系统。
//
// buffer(10) cellParams = (原子数, 胶囊数, 0, 0)，由 OpticsRenderer 写入；
// buffer(11) 原子表：每原子 2×float4 —— (pos.xyz, radius) (rgb, 0)；
// buffer(12) 胶囊表：每胶囊 3×float4 —— (a.xyz, radius) (b.xyz, 0) (rgb, 0)。
// 布局与 Swift 侧 OpticsCellState 逐字段对应，改任何一边都要同步另一边。

#define CELL_MAX_ATOMS 512
#define CELL_MAX_CAPSULES 1024

/// ray-sphere：最近正 t；无命中 -1
float cellSphereHit(float3 origin, float3 rd, float3 center, float radius) {
    float3 oc = origin - center;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - radius * radius;
    float h = b * b - c;
    if (h < 0.0) return -1.0;
    float t = -b - sqrt(h);
    if (t > 1e-6) return t;
    t = -b + sqrt(h);
    return t > 1e-6 ? t : -1.0;
}

/// ray-capsule（圆柱身 + 半球端帽）；命中时给出表面法向
float cellCapsuleHit(float3 origin, float3 rd, float3 a, float3 b, float radius,
                     thread float3 &normal) {
    float3 ba = b - a;
    float3 oa = origin - a;
    float baba = dot(ba, ba);
    float bard = dot(ba, rd);
    float baod = dot(ba, oa);
    float rdoa = dot(rd, oa);
    float oaod = dot(oa, oa);
    float aa = baba - bard * bard;
    float bb = baba * rdoa - baod * bard;
    float cc = baba * oaod - baod * baod - radius * radius * baba;
    float h = bb * bb - aa * cc;
    if (h >= 0.0 && aa > 1e-12) {
        float t = (-bb - sqrt(h)) / aa;
        float y = baod + t * bard;
        if (y > 0.0 && y < baba && t > 1e-6) {
            normal = normalize(oa + t * rd - ba * (y / baba));
            return t;
        }
        // 端帽：退化成球求交（球心取更近的一端）
        float3 oc = (y <= 0.0) ? oa : (origin - b);
        bb = dot(rd, oc);
        cc = dot(oc, oc) - radius * radius;
        h = bb * bb - cc;
        if (h > 0.0) {
            t = -bb - sqrt(h);
            if (t > 1e-6) { normal = normalize(oc + t * rd); return t; }
        }
    }
    return -1.0;
}

fragment float4 optics_cell_fragment(VertexOut in [[stage_in]],
                                     constant Uniforms &u [[buffer(1)]],
                                     constant float4 *cellParams [[buffer(10)]],
                                     constant float4 *atomTable [[buffer(11)]],
                                     constant float4 *capsuleTable [[buffer(12)]]) {
    // 与 optics_fragment 相同的焦平面映射（宽高比、焦点偏移、平移）
    float2 centered = (in.uv - 0.5) * 2.0;
    centered.x *= u.resolution.x / max(u.resolution.y, 1.0);
    centered.x += u.cameraUpFocal.w;
    centered -= u.panParams.xy;
    float3 rayOrigin = u.cameraPositionScale.xyz;
    float3 rd = normalize(
        u.cameraForwardExposure.xyz
        + u.cameraRightEnvRot.xyz * centered.x * u.cameraPositionScale.w
        + u.cameraUpFocal.xyz * centered.y * u.cameraPositionScale.w
    );

    int atomCount = int(cellParams[0].x + 0.5);
    int capCount = int(cellParams[0].y + 0.5);

    float bestT = 1e19;
    float3 bestNormal = float3(0.0, 0.0, 1.0);
    float3 bestColor = float3(1.0);
    for (int i = 0; i < CELL_MAX_ATOMS; i += 1) {
        if (i >= atomCount) break;
        float4 pr = atomTable[i * 2];
        float t = cellSphereHit(rayOrigin, rd, pr.xyz, pr.w);
        if (t > 0.0 && t < bestT) {
            bestT = t;
            bestNormal = normalize(rayOrigin + rd * t - pr.xyz);
            bestColor = atomTable[i * 2 + 1].xyz;
        }
    }
    for (int i = 0; i < CELL_MAX_CAPSULES; i += 1) {
        if (i >= capCount) break;
        float4 pa = capsuleTable[i * 3];
        float4 pb = capsuleTable[i * 3 + 1];
        float3 capNormal;
        float t = cellCapsuleHit(rayOrigin, rd, pa.xyz, pb.xyz, pa.w, capNormal);
        // 注意：不能用 bestNormal 直接收出参——求交函数在 t 并非更优时也会写它，
        // 会把上一个更优命中的法向悄悄覆盖掉（不报错，只画错光影）
        if (t > 0.0 && t < bestT) {
            bestT = t;
            bestNormal = capNormal;
            bestColor = capsuleTable[i * 3 + 2].xyz;
        }
    }

    if (bestT >= 1e18) {
        return float4(sceneBackground(u, in.uv, rayOrigin, rd), 1.0);
    }

    // 显示空间双灯着色（base 已是 sRGB；与 sceneBackground 同一口径，不再走 gamma）
    float3 key = normalize(float3(0.55, 0.35, 0.76));
    float3 fill = normalize(float3(-0.6, -0.2, -0.5));
    float diff = max(dot(bestNormal, key), 0.0) * 0.72
               + max(dot(bestNormal, fill), 0.0) * 0.22;
    float3 half0 = normalize(key - rd);
    float spec = pow(max(dot(bestNormal, half0), 0.0), 48.0) * 0.5;
    float rim = pow(1.0 - max(dot(bestNormal, -rd), 0.0), 3.0) * 0.18;
    float3 color = bestColor * (0.34 + diff) + spec + rim * bestColor;
    return float4(clamp(color, 0.0, 1.0), 1.0);
}

// MARK: - 光轴指示层（紫色虚线 · 量子光线同款发光细线，叠加在主结果之上）
//
// 画法对齐 optics_quantum_fragment 的轨迹段：端点投到屏幕（quantumProject，
// 与射线构造严格互逆），核心线 ~1.4px + 柔和辉光，线宽恒定不随缩放加粗。
// 重合的轴直接叠加发光（一轴晶光轴与 c 重合即如此，不做并排偏移）。
// 几何/配色由 Swift 侧 CrystalMarkup 统一构造，笔记 / 光学实验室米勒指数 /
// 偏光实验室 3D / 量子实验室四处共用同一份虚线。
//
// 遮挡：panParams.w = occludeAxes 为真时，被晶体挡住的轴段不发光（axial
// 点对相机做一次 BVH 求交，命中在点之前即遮挡）。2026-10 起四个调用方
//（笔记晶轴、光学实验室米勒指数白模、偏光 3D、量子）一律恒显 X 光虚线：
// 二轴晶两根光轴共四个方向，遮挡会把背对镜头的一根整根吞掉。遮挡代码保留
// 作能力开关，当前无调用方开启。
//
// 近裁剪面：胶囊端点转到相机身后时不能整条丢弃（旋转临界角会闪烁/缺失）——
// 把身后端点裁到 depth=0.01 近平面上，只画仍可见的那段。
//
// buffer(13) axisParams = (线段数, 全局不透明度, 0, 0)
// buffer(14) 轴表：每根 3×float4 —— (a.xyz, 线宽px) (b.xyz, 0) (rgb, alpha)
// 与 Swift 侧 OpticsAxisState 逐字段对应，改任何一边都要同步另一边。

// 与 Swift 侧 OpticsAxisState.maxCapsules 一致（切磨工作台刀口描边需要 ~160+）
#define AXIS_MAX_CAPSULES 256

fragment float4 optics_axis_fragment(VertexOut in [[stage_in]],
                                     constant Uniforms &u [[buffer(1)]],
                                     constant float4 *axisParams [[buffer(13)]],
                                     constant float4 *axisTable [[buffer(14)]],
                                     constant float4 *meshParams [[buffer(7)]],
                                     constant float4 *meshNodes [[buffer(8)]],
                                     constant float4 *meshTris [[buffer(9)]]) {
    float2 pixel = in.uv * u.resolution;
    int count = int(axisParams[0].x + 0.5);
    int nodeCount = int(meshParams[0].y + 0.5);
    float3 camPos = u.cameraPositionScale.xyz;
    float3 camFwd = u.cameraForwardExposure.xyz;

    float3 add = float3(0.0);
    for (int i = 0; i < AXIS_MAX_CAPSULES; i += 1) {
        if (i >= count) break;
        float3 pa3 = axisTable[i * 3].xyz;
        float3 pb3 = axisTable[i * 3 + 1].xyz;
        float3 tint = axisTable[i * 3 + 2].xyz;
        float width = max(axisTable[i * 3].w, 0.8);
        // 近裁剪面裁剪：深度 ≤ 0.01 的端点挪到近平面交点，保证旋转到轴
        // 几乎指向相机时胶囊只是缩短、不会整条消失（旧实现任一端点投影
        // 失败即 continue，临界角下虚线成段闪烁缺失）。
        float da = dot(pa3 - camPos, camFwd);
        float db = dot(pb3 - camPos, camFwd);
        if (da <= 0.01 && db <= 0.01) continue;
        if (da <= 0.01) {
            pa3 += (pb3 - pa3) * clamp((0.01 - da) / (db - da), 0.0, 1.0);
        } else if (db <= 0.01) {
            pb3 += (pa3 - pb3) * clamp((0.01 - db) / (da - db), 0.0, 1.0);
        }
        float2 pa, pb;
        if (!quantumProject(u, pa3, pa) || !quantumProject(u, pb3, pb)) continue;
        // 辉光半径 ~9px：离线段超过它的像素连求交都不做（全屏 × 64 线段
        // 的 BVH 遮挡是本 pass 唯一贵的东西，先用屏幕距离把 99% 像素筛掉）
        float2 ba = pb - pa;
        float h = clamp(dot(pixel - pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
        float d = length(pixel - pa - ba * h);
        if (d > 9.0) continue;
        // 遮挡开关：仅当渲染状态显式请求（panParams.w = occludeAxes）时遮挡。
        // 白模型下 occludeAxes=true → 光轴进入晶体内部被不透明晶体挡住、只露
        // 两端，与偏光实验室紫轴同机制；晶体外轴段恒显。
        if (u.panParams.w > 0.5 && nodeCount > 0) {
            float3 p3 = mix(pa3, pb3, h);
            // 过晶体中心、背离相机的远半侧轴段必被不透明实体挡住：凸晶体下整段
            // 都在实体之后，BVH 仅能吞正对相机那侧的内部，薄/长晶体上背侧轴会
            // 探出轮廓外形成「断开」；直接按中心面裁掉远半段，得到「光轴从晶面
            // 出来、内部被挡」的连贯观感（与偏光实验室一致）。近半侧外部段恒显。
            if (dot(p3, camPos) < 0.0) { continue; }
            float3 toP = p3 - camPos;
            float distP = length(toP);
            float occlT;
            float3 occlN;
            if (distP > 1e-5 &&
                intersectMeshBVH(nodeCount, meshNodes, meshTris,
                                 camPos, toP / distP, occlT, occlN) &&
                occlT < distP - 0.004) {
                continue;
            }
        }
        float core = exp(-(d * d) / max(width * width, 0.25));
        float halo = exp(-(d * d) / 9.0) * 0.42;
        add += tint * (core + halo) * axisParams[0].y;
    }
    if (add.r + add.g + add.b <= 0.0) {
        return float4(0.0);   // 未命中：alpha 0，叠加后不改动主结果像素
    }
    // rgb 按 add 装载、alpha 取分量最大值：辉光尾部 alpha 小 → 与主结果
    // 平滑混合；线芯重合处（光轴×c）亮度叠加、rgb 戳到 1 → 读作同一条亮线
    float a = clamp(max(add.r, max(add.g, add.b)), 0.0, 1.0);
    return float4(clamp(add, 0.0, 1.0), a);
}

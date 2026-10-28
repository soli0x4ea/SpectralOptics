# 许可范围与迁移说明 / Licensing transition

> 存档说明：本文件取自上游 https://github.com/yuyou-dev/OpenGemCutting/blob/main/docs/legal/LICENSING.md
> （抓取日期 2026-10-28）。仅作许可证据存档，非本仓库的许可条款；本仓库许可见根目录 LICENSE 与 THIRD_PARTY_NOTICES.md。

---

状态：自 **v2.1.1（2026-10-14）** 起采用本许可迁移方案；准确源码边界以公开 `v2.1.1` 标签所指提交为准。既有 MIT 授权保持有效。本页是说明；正式英文条款见根目录 LICENSE。

## 新条款的目标

维护者有权授权、且首次随新条款发布的原创内容采用 **OpenGemCutting Personal Non-Commercial License 1.0**。允许个人非商业自用、学习、自己设备上的运行、备份和私下修改；禁止商用、软件二次分发、公开镜像、打包出售、对外托管及 SaaS/API 服务。公司内部使用、接单设计、设计／加工用于出售的宝石也属于商业使用，须另行取得书面许可。该商业使用口径及本次许可发布已由维护者确认。

使用者仍拥有自己的原创设计和输出；个人非商业分享设计结果不等于分发软件。软件许可不替代宝石切型来源本身的权利。该许可属于"源码可见、限制使用"，不是 OSI 定义的开源许可；项目名称 OpenGemCutting 不代表新内容仍按 MIT 授权。

## 不能被新条款收回的范围

1. **既有 MIT 发布**：截至迁移前公开 main `e8cfa6c906001c3a28fc6f786a437cbbf3a07e2e`（v2.1.0 发布合并）以及此前按 MIT 提供的内容，继续保留原有授权。旧文本完整保存于 [LEGACY_MIT_LICENSE.txt](LEGACY_MIT_LICENSE.txt)。相同代码换一个许可证文件不能消除旧授权；新限制主要约束以后首次按新条款发布的原创增量。
2. **外部贡献**：公开历史中 EthanSMC 的 `fb9b7fb`、`293a9b9` 涉及 WebGPU、调度、着色器、测试和文档，按当时 MIT 提供。保留作者与许可，不宣称取得其重新授权或独占权。其他已按 MIT 发布的项目内容同样保留原授权。
3. **独立模块**：网格内核、Earcut、实验室 MIT 包、格式桥接来源、字体和 npm 依赖等保留其自有许可证；不改 vendor 的 LICENSE，不给 MIT／ISC／LGPL／OFL 材料附加替代限制。详见 [第三方声明](THIRD_PARTY_NOTICES.md)。p5.js 等 LGPL 组件所要求的使用者权利继续保留；未来分发构建仍须核对相应许可义务。
4. **GitHub 平台**：公共仓库的站内查看和 fork 受 GitHub 条款约束，无法承诺绝对禁止 fork。这里不擅自更改仓库可见性、移除旧标签或关闭现有发布；转私有也不能收回已授予的 MIT 权利。

## 后续贡献与发布

缺陷报告、原创说明与设计建议仍可提交。新代码贡献暂不按"Fork 后即可提交代码"办理；先由维护者就准确代码范围确认贡献者有权授权，并取得发布所需书面许可。旧贡献不会因本页被视为同意新条款。Companion 的贡献提交流程应遵守 [贡献说明](../../.github/CONTRIBUTING.md)，既有 MIT fork 的权利保留。

发布前应由熟悉适用法域的软件许可专业人士复核：旧 MIT 与新原创内容的边界、外部贡献、LGPL 构建交付、商业使用定义及贡献条款。本次 v2.1.1 是许可与文档发布，不修改应用功能，也不追溯禁止旧版本商用／分发。发布记录列明验证结果；维护者确认不等于已经完成法律专业审查。

参考依据：[MIT 原文](https://opensource.org/license/mit)、[OSI 开源定义](https://opensource.org/osd)、[GitHub 仓库许可说明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)、[GitHub Terms of Service](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service)。

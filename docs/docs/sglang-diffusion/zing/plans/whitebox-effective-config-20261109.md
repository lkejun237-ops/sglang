# Whitebox 配置与运维入口核查 — 2026-11-09

本记录服务于 PR #647 的源码清理，不是上线验收。所有线上观测均为只读；没有加载GPU模型、
重启服务或改平台配置。[结构化证据](../../../../../benchmark/whitebox/history/20261109-release-observations.json)
包含完整SHA、观测时间、版本接口来源与证据等级。[当前操作手册](whitebox-manual-releases.mdx)为命令入口。

## 冻结版本和可比边界

- main：`3a9236633cc35bfe618af8caa13c16180d294717`。
- candidate实现：`8c9ace34a6ce30fea2ac5957514420053fb7747b`；其后独立审查另修复准备器Decoder准入（只接受当前原生服务支持的lynn_streaming）；CLI/Qwen表值不变，新准入单独补测。
- SDK：`689e81f718e22065877de9af589e6496c712f979`，dev中的准确pin；3个capture/producer/session测试文件40项通过。
- main与candidate均执行各自真实 `ServerArgs.add_cli_args`；Qwen六项执行原生env解析函数。
  `None`不填成0。main不识别候选新增缓存CLI，因此没有拿候选argv强行当main启动参数。
- candidate以仓库脱敏bindings示例生成bundle/commands，验证公共行为；**没有把示例称为现网接线**。
  真实私有bindings、GPU逐rank硬件和初始化后最终配置本轮未采齐；同模型/硬件的自动调整结果为unknown。
- pipeline的main值为冻结版本dataclass默认；候选值来自prepare_assets输出合同，不是已加载模型观测。

## 当前可核实的运行身份

| 环境 | Gateway自报source SHA | SDK SHA | 证据和限制 |
| --- | --- | --- | --- |
| prod-shanghai | `b94033f08df7caf4bf017712b12fdd0c126dfa1f` | `689e81f718e22065877de9af589e6496c712f979` | 2026-11-09T16:53:44.465114+08:00，[版本API](https://whitebox-sh.loopit.me/whitebox/v1/whitebox/version)；不是全进程/镜像审计 |
| prod-thailand | `2e0833d3e251341d5ef83110aaec9f3e40e5dce1` | `e65413e08443dd2f8991090e445a1805a494ef2c` | 2026-11-09T16:53:45.277542+08:00，[版本API](https://whitebox-th.loopit.me/whitebox/v1/whitebox/version)；不是全进程/镜像审计 |
| test-shanghai-copy | `af9ea3bf31268c9dfd8e4d1bc2b51636e46ec132` | `689e81f718e22065877de9af589e6496c712f979` | 2026-11-09T16:53:45.405581+08:00，[版本API](https://shcopy.world-test-platform.loopit.com.cn/whitebox/v1/whitebox/version)；不是全进程/镜像审计 |

这三条在线版本仍自报历史 `white_box` 来源，不能因新发布政策改成main而重标旧镜像。
泰国平台service/lane与Pod容器的image引用一致：Gateway/Coordinator为
`sha256:51ed8cab94deb3917c49a1d17342599dc94a63fb049f7972ee25c0952aa0a715`，Room为
`sha256:61e8bcf98f98d67f748f153d06f2a9a7f87450f6c9e3df18a999cfd712cdd2d9`。
API未提供本轮可核验的runtime imageID、进程argv或加载后模型配置，不能把image引用等同于这些证据。
上海CPU/两地GPU/副本GPU的实际imageID、逐rank硬件、模型初始化配置仍为unknown。

## 配置对照

下表的candidate列上海/泰国相同；地址/硬件要求/缓存路径不同另列于bundle。
CPU解析测试覆盖0/false关闭值及相反父环境，不只比较JSON或命令字符串。

| component/key | live value/evidence | frozen-main default | candidate effective（CPU阶段） | source/precedence | semantic/perf/memory/lifecycle effect | verification status |
| --- | --- | --- | --- | --- | --- | --- |
| DiT / zing_source_work_quantum | unknown | `1` | `1` | 显式CLI优先；未指定值继续由模型/自动调整决定 | 调度粒度与队列/吞吐 | CPU实际解析；GPU未验证 |
| DiT / zing_source_prompt_joint | unknown | `true` | `false` | 显式CLI优先；未指定值继续由模型/自动调整决定 | Prompt联合计算路径；可能影响耗时与数值 | CPU实际解析；GPU未验证 |
| DiT / zing_source_encoder_mode | unknown | 未指定（依赖后续模型/硬件处理） | `"overlap"` | 显式CLI优先；未指定值继续由模型/自动调整决定 | 编码/DiT重叠与峰值资源 | CPU实际解析；GPU未验证 |
| DiT / zing_source_encoder_prefetch | unknown | 未指定（依赖后续模型/硬件处理） | `true` | 显式CLI优先；未指定值继续由模型/自动调整决定 | 预取与在途内存 | CPU实际解析；GPU未验证 |
| DiT / zing_source_async_output | unknown | `true` | `true` | 显式CLI优先；未指定值继续由模型/自动调整决定 | 输出排队/尾部排空 | CPU实际解析；GPU未验证 |
| DiT / zing_source_decoder_overlap | unknown | 未指定（依赖后续模型/硬件处理） | `true` | 显式CLI优先；未指定值继续由模型/自动调整决定 | 解码重叠/资源峰值 | CPU实际解析；GPU未验证 |
| DiT / zing_source_defer_final | unknown | `true` | `true` | 显式CLI优先；未指定值继续由模型/自动调整决定 | KV提交边界与时序 | CPU实际解析；GPU未验证 |
| DiT / mem_fraction_static | unknown | 未指定（依赖后续模型/硬件处理） | `0.8` | 显式CLI优先；未指定值继续由模型/自动调整决定 | 静态KV预算；实际每rank容量待GPU计算 | CPU实际解析；GPU未验证 |
| DiT / sp_degree | unknown | 未指定（依赖后续模型/硬件处理） | `2` | 显式CLI优先；未指定值继续由模型/自动调整决定 | 通信拓扑/并行开销 | CPU实际解析；GPU未验证 |
| DiT / num_gpus | unknown | `1` | `2` | 显式CLI优先；未指定值继续由模型/自动调整决定 | 可见DiT rank数，不是整个四卡组 | CPU实际解析；GPU未验证 |
| DiT / zing_initial_block_count | unknown | `3` | `3` | 显式CLI优先；未指定值继续由模型/自动调整决定 | 首段质量与启动延迟 | CPU实际解析；GPU未验证 |
| DiT / zing_denoising_step_list | unknown | 未指定（依赖后续模型/硬件处理） | `[1000, 500]` | 显式CLI优先；未指定值继续由模型/自动调整决定 | 稳态采样成本/画质 | CPU实际解析；GPU未验证 |
| DiT / zing_initial_denoising_step_list | unknown | 未指定（依赖后续模型/硬件处理） | `[1000, 750, 500, 250]` | 显式CLI优先；未指定值继续由模型/自动调整决定 | 初始采样成本/画质 | CPU实际解析；GPU未验证 |
| DiT / zing_reference_max_images | unknown | 未指定（依赖后续模型/硬件处理） | `21` | 显式CLI优先；未指定值继续由模型/自动调整决定 | 输入上限与条件内存 | CPU实际解析；GPU未验证 |
| DiT / realtime_session_lease_ttl_s | unknown | `60.0` | `1200.0` | 显式CLI优先；未指定值继续由模型/自动调整决定 | 租约/清理边界 | CPU实际解析；GPU未验证 |
| DiT / realtime_session_idle_timeout_s | unknown | `60.0` | `1200.0` | 显式CLI优先；未指定值继续由模型/自动调整决定 | 闲置收尾边界 | CPU实际解析；GPU未验证 |
| DiT / realtime_session_max_lifetime_s | unknown | `600.0` | `7200.0` | 显式CLI优先；未指定值继续由模型/自动调整决定 | 底层会话上限，区别于每玩家TTL | CPU实际解析；GPU未验证 |
| DiT / zing_block0_cache_host_gb | unknown | 未指定（依赖后续模型/硬件处理） | `416.0` | 显式CLI优先；未指定值继续由模型/自动调整决定 | Host缓存容量 | CPU实际解析；GPU未验证 |
| DiT / zing_block0_cache_disk_gb | unknown | 未指定（依赖后续模型/硬件处理） | `700.0` | 显式CLI优先；未指定值继续由模型/自动调整决定 | 磁盘缓存容量 | CPU实际解析；GPU未验证 |
| DiT / zing_block0_cache_gc | unknown | `false` | `true` | 显式CLI优先；未指定值继续由模型/自动调整决定 | 后台回收 | CPU实际解析；GPU未验证 |
| DiT / zing_block0_cache_min_free_gb | unknown | 该版本解析器未提供 | `128.0` | 显式CLI优先；未指定值继续由模型/自动调整决定 | 磁盘余量防护 | CPU实际解析；GPU未验证 |
| Encoder/Conditioning / Qwen hf_rmsnorm | unknown | `true` | `false` | 公共env→原生Qwen解析函数 | kernel/FP8数值路径、速度与显存；六项不是无影响的格式差异 | CPU解析；GPU未验证 |
| Encoder/Conditioning / Qwen fused_mrope | unknown | `true` | `false` | 公共env→原生Qwen解析函数 | kernel/FP8数值路径、速度与显存；六项不是无影响的格式差异 | CPU解析；GPU未验证 |
| Encoder/Conditioning / Qwen fused_gate_up | unknown | `true` | `false` | 公共env→原生Qwen解析函数 | kernel/FP8数值路径、速度与显存；六项不是无影响的格式差异 | CPU解析；GPU未验证 |
| Encoder/Conditioning / Qwen text_fa4 | unknown | `true` | `false` | 公共env→原生Qwen解析函数 | kernel/FP8数值路径、速度与显存；六项不是无影响的格式差异 | CPU解析；GPU未验证 |
| Encoder/Conditioning / Qwen vision_fa4 | unknown | `true` | `false` | 公共env→原生Qwen解析函数 | kernel/FP8数值路径、速度与显存；六项不是无影响的格式差异 | CPU解析；GPU未验证 |
| Encoder/Conditioning / Qwen fp8 | unknown | `true` | `false` | 公共env→原生Qwen解析函数 | kernel/FP8数值路径、速度与显存；六项不是无影响的格式差异 | CPU解析；GPU未验证 |
| pipeline / window,sink | unknown | 21,2（dataclass） | 6,0 | 公共policy→prepared profile及KV pool同时绑定 | KV容量、上下文保留与行为 | CPU资产校验；GPU未验证 |
| pipeline / AdaLN | unknown | weights_free（dataclass） | resident | 公共policy→pipeline；禁止地域覆盖 | 常驻权重/算力和显存路径不同 | GPU速度/显存未知 |
| model / checkpoint | unknown | 依赖所选model | 89c3a67d…a1d731；完整值见common.json | prepared zing_config与公共policy一致 | 模型身份，不能代替全部权重hash | CPU元数据校验；实际权重未读取 |
| Decoder / type,weights,architecture | unknown | 依赖所选资产 | 四卡入口固定lynn_streaming/bf16并冻结资产SHA | prepared pipeline/receipt；两地必须相同资产合同 | 数值、画质、解码内存与时序 | 模板/拒绝测试；GPU未验证 |
| model / dimensions | unknown | 模型/请求决定 | 832×480 | common.json；输入与目标尺寸分开 | token/KV/显存与画质 | 配置校验；736未验证 |
| Gateway/Room/Coordinator / player TTL | unknown（未读进程env） | 组件默认，未等同于model lifetime | 1800秒 | 公共policy→角色env | 单玩家寿命；不是GameRoom统一寿命 | CPU配置；线上未应用 |
| model group / layout | unknown | 由入口与硬件决定 | E1+Conditioning+SP2+D1；单玩家 | 完整组launcher映射4卡；standalone另走native入口 | 不把rank/service数当玩家容量 | CPU命令；真实GPU未运行 |

当前相对旧地区配置的候选变化还包括上海AdaLN table→resident、媒体预算回公共值、泰国每玩家
7200→1800秒。它们满足统一公共策略要求，但可能改变显存、播放和生命周期；未实测不能写无退化。
角色池模板不得被解释为已有Encoder/Decoder也切换了这些值。

## Jenkins、Grafana与采集状态

- Jenkins真实base：[jenkins.ops.loopit.me](https://jenkins.ops.loopit.me/)。
  [Room Job](https://jenkins.ops.loopit.me/job/loopit/job/world-model/job/whitebox-th-room/)及
  [最近成功构建43](https://jenkins.ops.loopit.me/job/loopit/job/world-model/job/whitebox-th-room/43/)
  的JSON API已读。该构建源SHA为 `d76297aa4e6bca2ce6413cc8aa4be0d2e6c45152`，另有SDK检出
  `d40a219e14813b62c564c7322af0325f367271eb`；它不等于当前Room运行镜像。
- 平台历史buildUrl含旧的编码分号，本记录使用Jenkins Job自身返回的实际URL。
  平台历史“生效中”记录image与当前service/Pod image不一致，因此历史标签不能替代现场身份。
- Control Job为 `loopit/world-model/whitebox-th-control`。两个Job的`config.xml`均HTTP403；
  **安装中的pipeline来源/SHA未核实**，不能声称它已经更新为本PR的Jenkinsfile。
- Grafana真实base：[log.ops.loopit.me](https://log.ops.loopit.me/)，health JSON HTTP200。
  Prometheus datasource UID=`prometheus`，类型`prometheus`，默认datasource，已通过API核实。
  当前可见范围搜索Whitebox返回空；看板UID与provisioning/import证据为unknown，不等于全局不存在。
- 泰国Gateway、Coordinator、Room以及monitor-agent default lane读取到`businessMonitoring.metricsPort=0`。
  因而这些lane没有通过该配置启用组件业务metrics；其他采集路径需另证，不以仓库Grafana JSON代替。
- 仓库的 `benchmark/whitebox/grafana-dashboard.json` 保留为可维护定义；启用看板、ServiceMonitor或告警
  是独立运维变更，本次没有执行。Prometheus查询 `up{namespace="whitebox-playtest"}` 成功但返回0条序列；这不是组件healthy证明，也不能证明不存在其他采集路径。

## 下一次现场采集须补齐

用对应环境权限只读采集每个组件的实际imageID、进程argv与白名单env，保存bundle/pipeline/profile
哈希及初始化后最终配置。逐rank记录GPU UUID/capability/显存、模型/Decoder身份与自动调整日志。
Coordinator/Room与模型池Coordinator分别记录，不能用Gateway一次HTTP200替代四组件。
之后用真实私有bindings重生成candidate，对照相同模型与硬件，填写上表unknown；不需要为采集重启服务。

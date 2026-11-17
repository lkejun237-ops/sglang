# 原始需求与当前实现的接口对照

适用基线：2026-11-12 的 SGLang main `f73646b56cac1242041ecc0a88d32a8813668a10`。
运行时仍以所选源码的 CLI/校验代码为准，旧 PR 的 schema/示例不构成当前接口合同。

## 为什么维护在 SGLang

[infra-skills #17](https://github.com/seedleap/infra-skills/issues/17) 原先规划跨仓库 skill，
Antigravity「FR17-SGLang Release And Deploy Skill」最后对齐的是 #655 `c163f655e6`。
后续 [#657](https://github.com/seedleap/sglang/pull/657) 已合并编排器和同名仓库自有 skill。
该 skill 随 launcher/spec/state 变化，使用仓库内相对入口；infra-skills 继续提供可跨项目复用的镜像、迁移、checkpoint 能力。
不再建立第二份部署 skill 或重复底层脚本；原始 Issue 的四个缺口按下表处理。

| 原始反馈 | 当前代码与 skill 应如何处理 |
| --- | --- |
| 模型池配置与密钥过滤 | `launch.py` 的 worker_config、secret_files 和 SECRET_ENV，`orchestrate.py attach` 已提供受控路径。密钥从平台挂载的文件注入，回执排除Secret环境键；接线不等于发布Coordinator。 |
| 回执持久化与墙钟阶段计时 | 编排器写 state 的 created_at、deployment/history 的 created_at/updated_at，并收集运行回执；原始 launch receipt 仍在解包目录旁，没有通用 receipt-output 或 started_at/ready_at/finished_at 字段。归档原始回执，不伪造字段；必要时在外层另记录观测时间/测量口径。 |
| assets.json 生成与全量hash耗时 | 当前命令叫 `assets`，不是 hash-assets。生成后再核验；不能把清单生成冒称原始训练导出 provenance 已验证。原始 receipt 没有独立 asset_verify_duration_sec，缺项标未采集，不能拿整体启动耗时填充。 |
| 固定站点与SP2边界 | 当前 target/site 参数化，支持测试与生产环境的H100/H200与B300角色；具体profile/拓扑按当前校验器准入。多service不等于多SP rank，不能为了736静默增卡或将480结果外推。 |

## 资产、凭据与来源

由资产准备者确认模型和外部组件实际目录、独立角色pipeline、AdaLN manifest/权重/schedule/CUDA provenance，
部署准备者据此冻结 spec、镜像 digest、asset_receipt_sha256 和可追溯源码 SHA。资产/策略变动后重新生成清单和计划。
用当前 config 的 required 字段准备 bootstrap 配置；CLI 会先 load_config，不支持凭空省略必填字段。

```bash
# 准备者已有合法role config；为外部组件逐项追加 --extra-file。
python3 deploy/sglang/launch.py assets --config /private/deploy/dit.json \
  --extra-file /verified/lynn/model.safetensors --output /private/deploy/assets.json
# 将生成的路径/hash写回role config并冻结；不带output只核验。
python3 deploy/sglang/launch.py assets --config /private/deploy/dit.json
```

正式交付前先 fetch origin/main，用 `git merge-base --is-ancestor "$SOURCE_SHA" origin/main` 核验来源；
开发验证的任务分支也必须已推送并核对远端SHA，单独标注开发状态。使用完整40位commit，禁止从脏工作区补丁或镜像旧源码拼装。
源码交付使用 `source.prepare`/`orchestrate up` 的不可变归档与目标端校验，不重写OSS上传/解包工具。

不要复制凭据到聊天或shell参数。Secret路径与环境名来自当前 worker_config/secret_files，执行机从已有授权文件提供密钥；
不得从Worker下载密钥、关闭签名鉴权或在skill示例写明文 export TOKEN。缺权限或跨节点网络不通就报告具体阻塞。

## 验证证据的版本边界

#657 的上海480实测、受控回滚和浏览器记录在根目录 `benchmark/whitebox/sglang_shanghai_copy_20261110.md`，
属于其冻结源码、模型、输入和环境；不是任意新源码/1102/736/B300的统一通过证明。
当前技能文档可用离线parser与既有合同回归验证，不需要为文档补测申请GPU或改线上服务。
模型ready、通信组selftest、真实出帧、池注册与浏览器业务E2E各有单独结果；释放未知也不能报告验收通过。

上海副本支持由获准的 RAM 用户通过 `deploy/sglang/ram_publish.py request` 发起一次已自测版本的挂载。发布作业以自己的原生身份执行，命令、权限范围和回执含义见 `deploy/sglang/ORCHESTRATION.md` 的“上海测试池由 RAM 用户发起挂载”。自测、挂载及玩家出帧分别核验。

# GPU 任务归属记录

核验日期：2026-11-06。供 Codex 查询 H100/H200、B300 任务时识别实际提交账号。
人员展示统一使用两字称呼；`dev_api` 保留平台账号名。

## 稳定账号映射

| 称呼 | PAI / RAM Username | 查询任务用的 UserId | 证据 |
| --- | --- | --- | --- |
| 克君 | `y0lo87u8c1jb` | `205206788242572972` | 云 SSO 实名目录 + RAM 用户目录 |
| dev_api | `dev_api` | `1728199219479827` | 上海与泰国工作区成员目录 + 历史任务的 UserId/Username |
| 豪泽 | `afr8g73djy7u` | `206388188177887498` | 云 SSO 实名目录 + RAM 用户目录 |
| 天鹏 | `s8m6s1z0w3kn` | `201607488177885763` | 云 SSO 实名目录 + RAM 用户目录 |
| 英伦 | `xqgugl1ub66i` | `203386289444729276` | 云 SSO 实名目录 + RAM 用户目录 |
| 泽阳 | `bk2i7voy1091` | `208673589353288819` | 云 SSO 实名目录 + RAM 用户目录 |
| 逸文 | `0u1xq4yio4jj` | `204082489356491299` | 云 SSO 实名目录 + RAM 用户目录 |
| 生栋 | `uno822gr1qt6` | `207692188177887224` | 云 SSO 实名目录 + RAM 用户目录 |
| 夏腾 | `naozw93w7p9j` | `208194088177885318` | 云 SSO 实名目录 + RAM 用户目录 |

上述为已核验映射。英伦、泽阳、逸文此前仅按工作内容推定，本次通过实名目录确认。
本表只保留运维相关账号；未列入的 UserId 标为“未知”，查询目录核实后再补充。

`dev_api` 是独立平台账号，也是上述工作区的 owner；工作区 owner、任务的 TenantId
或本机 CLI profile 名都不能代替任务提交人的 UserId。即使业务由克君负责，
也应分别显示“提交账号：dev_api；业务负责人：克君”，不能把两个账号合并。
本机查询默认使用已授权的 `shengdong` profile，不要求切换到同名 `dev_api` profile。

## 归属识别规则

1. 查机器前先读[测试机器与访问方式](benchmark/AI_INFRA_TEST_MACHINES.md)，
   确认 region、workspace、ResourceId；硬件按实际资源核验，不能仅凭任务名判断。
2. 对 PAI-DLC 任务，以 `GetJob.UserId` / `ListJobs.Jobs[].UserId` 精确匹配上表，
   输出称呼、Username、UserId 和精确 Job ID。Username 可能为空，不影响数字 ID 匹配。
3. 任务名、代码分支、checkpoint 和用户明确指认可作为业务负责人线索，
   必须与实际提交账号分别记录；代提交不能按任务名称归户。
4. 工作区成员列表不等于全部历史提交者列表；不能因成员列表里没有某人而排除其历史任务。
5. H100 独立 K3s 节点没有 PAI-DLC Job ID。记录 namespace、Pod/Job 名称与 UID，
   只有创建者元数据或审计记录明确携带可关联的账号身份时，才使用本表归属。
   `root`、Kubernetes ownerReferences 和工作负载名称均不能证明实际提交人；
   缺少证据就标为“提交人未知”，不得套用 dev_api 或克君。
6. 任务状态、占卡数及当前运行 Job ID 必须实时查询，不作为长期身份映射保存。
   本记录不授权停止、修改或删除任何任务。

## 按账号查询任务

以下为只读查询示例；先按当前机器入口选择 region/workspace。H200 常用
`cn-shanghai` / `1453237` / `quota3l8bnv9a35n`；B300 常用
`ap-southeast-7` / `384` / `quota1betfyi1ilu`。CPU 辅助任务另查泰国 workspace `385`。
其他配额不能直接标为同一种硬件。

```sh
TASK_OWNER_SESSION_ID=$(uuidgen | tr '[:upper:]' '[:lower:]')
TASK_OWNER_UA="AlibabaCloud-Agent-Skills/alibabacloud-pai-dlc-job/$TASK_OWNER_SESSION_ID"
TASK_OWNER_UID=205206788242572972  # 克君；查询其他账号时替换为表中 UserId

aliyun pai-dlc ListJobs --profile shengdong --region cn-shanghai \
  --WorkspaceId 1453237 --ShowOwn false \
  --UserIdForFilter "$TASK_OWNER_UID" \
  --StartTime 2026-01-01T00:00:00Z --PageNumber 1 --PageSize 100 \
  --user-agent "$TASK_OWNER_UA" \
  --cli-query '{TotalCount:TotalCount,Jobs:Jobs[].{JobId:JobId,Name:DisplayName,UserId:UserId,Username:Username,Status:Status,ResourceId:ResourceId}}'
```

示例显式指定历史起点；实际按需要设置时间范围。未指定起点时默认只查最近 7 天，
不能将该结果当作全部任务。按 `TotalCount` 继续递增 `PageNumber` 直到读完所有页；
不要只列第一页，也不要只查 Running 就宣称账号没有历史任务。
若只看某硬件配额，可额外传 `--ResourceId` 精确筛选。

## 映射核验依据与复查

2026-11-06 只读查询以下 API，将云 SSO 的实名与 Username 对齐，再以同一
Username 关联 RAM 数字 UserId；`dev_api` 则直接从工作区成员及历史任务核验。
这使人员映射独立于任务用途和命名习惯。

- `cloudsso ListUsers`：`cn-shanghai`，DirectoryId `d-00abarvwb464`；
  核对 `DisplayName`、`UserName`，本次 `IsTruncated=false`。
- `ram ListUsers`：核对同一 `UserName` 对应的数字 `UserId`，本次 `IsTruncated=false`。
- `aiworkspace ListMembers`：上海 workspace `1453237`、泰国 workspace `384`；
  `dev_api` 的 `UserId=1728199219479827`、`AccountType=1`。
- `pai-dlc ListJobs`：上述两个 GPU 工作区，分别按克君、dev_api 的 UserId 分页查询，
  核对 `JobId`、`UserId`、`Username`。原始响应和任务状态快照不入库。

云 SSO 的 `u-...` UserId 与 RAM/PAI 数字 UserId 属于不同标识空间，
`UserIdForFilter` 必须使用本表数字 ID。复查 API 时处理全部分页，
仅选择身份相关字段；不要打印或归档完整作业环境、启动命令或镜像认证字段。

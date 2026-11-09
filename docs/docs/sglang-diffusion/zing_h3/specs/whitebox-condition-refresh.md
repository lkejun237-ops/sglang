# 服务端周期 Condition 刷新与 GPU Prefix 复用

这是实现说明，不是线上/GPU质量验收记录。无需新增客户端接口，不修改网页。
现有 `prompt.update` 的内容、版本、源边界和回执协议继续有效。

## 相同条件更新

当前已生效的完整 Prompt 未改变、请求未显式替换参考图，且模型已经应用的
条件事件与源会话当前事件一致时，复用该会话 GPU 上的静态 Prefix KV。
显式 `references` 更新（包括空数组）保守走原准备/准入/缓存恢复路径；
相同文字不允许覆盖参考图变更或尚未生效的候选。独立 Encoder/DiT 路径使用
原内容缓存键校验完整条件，并经 SP rank 一致性投票选择复用路径。

复用保留静态 KV 的原存储地址、字节、模型签名和分片布局；清除动态历史及
其位置/Graph派生状态，在原安全边界以最近五个生成 latent 重建 clean target
历史。不会重新生成这些 latent，不改变全局输出序号/RNG进度/Sink开关，
不开新Slot。当前结构中的合法 deferred-final source/target 对随旧历史一起
丢弃，不能在未完成 source/step0 事务中执行重置。

该路径跳过条件重新编码、L1/L2 lookup及静态Prefix H2D恢复；历史重建仍需计算。
原生计数 `gpu_prefix_reuses` 表示实际完成的复用；Block0回执使用 `cache_result=bypass`
及内部 `lookup.reuse_source=resident_gpu_prefix`，相关准备/恢复阶段标记skipped。
它不是L1 hit，更不能当作零耗时。真实条件变化仍使用原路径，支持原缓存行为。

## 纯服务端动态配置

运维在 Room 的已有运行配置目录管理：

```text
$WHITEBOX_ROOM_SPOOL/.runtime/condition-refresh.json
```

```json
{"interval_seconds": 5}
```

- `0` 关闭；正数设置刷新间隔，支持小数，最大86400秒。
- 新进程缺少有效文件默认关闭。运行中每秒最多读取一次；非法值/暂时缺失保留
  上次有效值并记录警告。建议原子替换文件，避免半写入。
- 改间隔重新计时；设0取消尚未派发的自动刷新。已派发操作安全结算后停止后续
  周期。修改无需重启，对该Room现有与新模型段生效。
- 各段独立计时；从真实模型输出开始。使用Host开场确认的游戏还必须已完成
  对应生命周期的确认。输入不用于自动延长周期，也不会被刷新伪装为活动。
- 间隔指上次条件更新/自动刷新生效后等待多久；完整clip边界、重建和后续RGB
  仍有开销，不承诺严格每5.000秒生效。

Room只在生成进展、条件已生效且原控制边界空闲时提交同一Prompt；参考图省略，
继续使用当前集合。真实用户条件更新取消尚未派发的自动更新；已派发的按原串行
边界结束。Sink、条件切换、Prefill、退出/断线、暂停或不支持clip barrier时不额外
入队。不补发错过的周期，不允许并发自动刷新，不暂停共享World或占额外Slot。

## 状态与恢复

自动更新的模型版本照常递增，沿原wire event_id和输出prompt_version确认。
每个Segment只保留一条内部周期操作及计数，不消耗1024条用户操作名额；两小时
1440次不会线性增长操作表。原业务Condition/lifecycle和历史操作证据保留，
#602开场确认不会被当作新事件重放。普通状态/帧版本仍如实反映模型执行版本；
不新增公开自动刷新控制消息，不向World重新提交一次性事件。

原生accepted、kv_ready、首RGB仍是不同阶段。未确认执行结果保持原pending；
错误后暂停该配置版本下的周期，运维可设0再启用。原模型段重建失败仍按既有
失败清理规则处理，不吞错继续生成，不自动重启模型，不伪造Slot空闲。

日志 `condition_refresh_policy`、`condition_refresh_enqueued`、`condition_refresh`
记录有效配置与执行版本/状态。内部Journal字段不会直接暴露为客户端控制参数。

## 验证边界

本地CPU小模型检查包含 eager/Graph路径模拟、split/unsplit、deferred-final、
引用图条件、输出数值对照、Prefix存储地址/字节保持、相同与不同条件/多rank
决策，以及真实Room代码通过受控Worker进行周期更新、热停用、按键和释放。
虚拟两小时周期检查操作状态有界。CPU Graph模拟不是CUDA Graph/SP GPU验收。

未证明真实checkpoint画质、抗漂移收益、FPS或线上延迟；这些需要后续授权的
同输入GPU/真实试玩对照。本实现不恢复此前已暂停的构建部署。

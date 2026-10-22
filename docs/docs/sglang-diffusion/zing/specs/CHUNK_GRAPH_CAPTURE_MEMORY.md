# Zing整块Graph capture的KV恢复边界

固定前向计划从host持有的历史长度记录`write_start`，不读取GPU上的索引张量。
每个前向仅在`[write_start, write_start + clean_rows)`写入权威raw K，
在`[write_start, write_start + rows)`写入V。该范围之外的历史K/V只读。
前缀计划的clean_rows等于其全部行数；active计划不写raw K。

Graph capture的eager预热会改写这些行。按整块中所有前向分别求K、V写入范围的
最小包围区间，只备份这些区间在各层的视图。预热后和capture结束/失败时恢复，
随后正式replay只执行一次更新。未写入的历史与容量尾部不需要复制；rotated K
不是权威存储，由每次执行在读取前重建，保持原有不备份的规则。

`write_start` 是当前固定前向计划的 host 元数据，与生成 `write_rows` 的历史长度
同源；它不增加会话状态。恢复边界不能靠读取 GPU 索引推导，否则会在 capture
准备阶段引入同步。区间取最小包围范围而非维护零散区间表，以覆盖同一整块内各次
前向写入；中间可能包含只读行，但不会遗漏声明范围内的写入。

备份是capture期间的临时显存，不改变启动时的池容量、admission、窗口、精度、
会话所有权或取消/清理行为。未来若固定前向增加其他K/V写入，必须同步扩大其
host计划声明的写入范围与恢复测试。

CPU记录后端检查完整模型输出/cache逐位一致和失败回滚；真实CUDA测试检查
eager预热、capture、replay恰好更新一次且历史保持不变。
测试与归档校验入口位于
[验证工具](../../../../../benchmark/zing_h3_causal/lynn_decoder_attention/README.md)；
实测显存、完整模型吞吐及当前版本验收状态位于
[PR #356](https://github.com/seedleap/sglang/pull/356)。

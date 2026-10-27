# Prompt 切换对齐 codec 周期

连续source与普通block推理沿用现有prompt更新入口，无新增开关。候选仍可随时接收；已有五个历史latent，且下一待生成的全局latent起点为5的倍数时，才实际采用。

## 行为与实现

当前codec每17帧产生五个latent，相位为0..4；DiT初始1/2/2，后续3/2。切换会把最近五个target latent重编号为本地0..4，再从本地5开始新块。若在全局43直接切换，历史38..42和新目标43的局部相位会偏离继续运行的codec。

修改后，在43收到候选，先用旧prompt完成43、44；到45才重建40..44，新prompt生成45..47与48..49。Decoder历史、绝对计数与相位继续运行。2+3仍可属于一次跨周期dispatch，先完成旧2，再开始新3。

实现集中在`SourceReadySession`：

- `prompt_switch_ready`使用已提交的全局游标，普通block入口也只在安全起点报告prompt更新。
- `_next_model_chunk`在有候选时先排空ready队列；未到周期边界继续原计划，到边界才选择新epoch的本地chunk3，避免lookahead提前排入跨边界旧块。
- `_activate_pending_prompt`核对真实`ready.chunk.start`，不只看会被重置的本地`model_chunk.start`。

保留最新候选覆盖旧候选的语义。EOS不足完整周期时，旧prompt完成现有尾帧，不补帧或重复输出。历史仍为最近五个target latent；source-history、reference缩放、采样、dispatch合并目标和Decoder协议不属于本修复。

四卡验证还发现独立Decoder输出身份的既存问题：后续命令只携带channel，输出却仍从命令取session，导致第一帧失败。`decoder_worker._output_identity`改用open保存且已通过channel校验的owner；不改变协议、相位或张量。对应控制循环回归使用真实身份组装函数，避免mock掩盖缺失字段。

## 决策与代价

约束epoch生效位置比重置Decoder相位更小，保留既有codec状态和输出时间线，不重排张量或复制KV。新增条件只读CPU游标，不增加GPU同步或编码。有待生效候选时暂停跨块预取，可能影响这段吞吐，尚未测量。候选可能等待下个尚未开始的周期；17帧约0.71秒是24 FPS下的视频时间，不是控制消息到生效的墙钟延迟保证。

沿用目标分支`844fdf42b077`的ready队列、prefix重建和分布式投票入口，以及现有diffusion pytest风格；参见[贡献指南](../../../developer_guide/contribution_guide.mdx)。这里是seedleap派生仓库的Zing契约，不是上游SGLang的通用限制。上游review重点应是默认生效时机变化、队列排空和分布式路径证据；不保证社区接受或画质改善。

## 验证与review

2026-10-26，本分支在基线`844fdf42b077`上的实现以本地WSL、Python 3.11、PyTorch 2.8.0+cpu、pytest 8.4.2验证，共401个不同CPU用例最终通过，4个CUDA命名用例未选择。其中：

- `test_zing_h3_chunks.py`与`test_zing_h3_delayed_eviction_prompt.py`：72通过。覆盖中途更新继续旧2、候选阻止跨边界lookahead、serial/cross_chunk到10才采用新3、EOS9不强制切换、最新候选、reference替换和普通block状态。
- 相关joint、delayed eviction、ready prepare/route、vote phase、Graph与model回归：初次323通过、6处旧切换预期失败；更新夹具与预期后相关7项通过，再重跑完整Graph CPU文件97通过。Graph使用CPU记录后端，不是真实CUDA Graph。
- Ruff 0.15.1检查/格式、isort 7.0.0、codespell 2.4.1、CPU测试inventory、两个仓库测试注册/lint检查及`git diff --check`通过。

各文件以`python -m pytest -q <测试文件>`执行，Graph/扩展组使用`-k 'not cuda'`。环境为`PYTHONPATH=python`、`SGLANG_LIGHTWEIGHT_RUNTIME=1`、`SGLANG_DIFFUSION_PLATFORM_OVERRIDE=cpu`、`HF_HUB_OFFLINE=1`，OMP/MKL/OPENBLAS线程数均为1；另设`SGLANG_DIFFUSION_PERF_BASELINE_PLATFORM=h100`仅用于避免公共测试配置导入时探测CUDA，不代表使用H100或运行性能测试。

同日，以运行时代码`02a234e7e633`在阿里云上海H100目标节点（实际设备L20Z/SM90，每卡81,559 MiB）完成三例四卡推理：GPU0为Qwen+Encoder，GPU1/2为DiT SP2/Ulysses2，GPU3为独立Lynn Decoder。三例均在全局latent 38收到候选，两个rank都先用旧prompt完成`[38,40)`，到40才采用新prompt，后续为`[40,43)`、`[43,45)`。客户端报告首个新prompt输出帧编号136（从0开始，视频时间约5.67秒）；三段均完整输出360帧并通过视频全量解码检查。

测试为832×480、24 FPS节奏输入、seed42、w6/sink3、block_relative/gap12、after_step0、prefix5、cross_chunk。客户端在输入128帧处等待只读观测确认已提交latent 38，再发送更新并继续输入；没有强制修改服务端生效位置或Decoder相位。权重使用节点缓存step2500，初始三块四步、后续两步。weights-free AdaLN静态表实际加载10项、242,242,560字节，checkpoint校验通过，投影参数存储为0；启动计数不代表完整请求命中统计。KV池为单槽、prefix上限12,288行，实际槽容量24,160行。

按性能基线配置FA4、RoPE融合/grouped/chunk reuse、stable shapes、编译AdaLN、FA4 metadata reuse、SP CUDA Graph和progressive encoder tiling，HISTORY_ROPE_PLAN=0；本次未采集Graph replay等逐项命中计数，不将开关值视为命中证据。硬件专用fixed FA4/current RoPE-KV关闭；remote Qwen使用BF16/SDPA，其六项优化开关均关闭以匹配可验证的服务profile。未启用reference/source缩放。w6、步数、checkpoint及Qwen配置是本次配方，不能与其他历史实验直接归因比较。

整轮每秒采样的NVML显存峰值为GPU0/1/2/3分别62,645 / 75,487 / 74,703 / 21,497 MiB；它是整卡采样值，不是组件allocator峰值。本次每例仅推理一次，没有性能测速、736档或修复前后同条件画质A/B，因此尚不能认定闪烁消失或吞吐改善。独立Decoder身份修复另有12项CPU回归通过。视频、配置及验证结果链接集中在[PR #430](https://github.com/seedleap/sglang/pull/430)，实验脚本和原始产物不进入仓库。

阅读顺序：本说明 → `incremental.py`的ready条件、块规划、生效函数 → `test_zing_h3_chunks.py`周期边界回归 → `test_zing_h3_delayed_eviction_prompt.py`普通block入口检查。重点核对全局游标与本地坐标，以及ready队列是否可能绕过边界。

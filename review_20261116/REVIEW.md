# 全双工 RAG 研究与 minicpm_rag_test 审查

日期：2026-11-16。范围：当前工作区中的实际文件（该目录目前未纳入 Git 跟踪），不是仅审查 main 提交。业务代码未修改。外部论文和官方仓库通过用户指定代理读取。

后续官方源码核查补充：Hugging Face `openbmb/MiniCPM-o-4_5` 的当前 `MiniCPMODuplex.streaming_prefill()` **已有 `text_list` 参数**，包含 TEXT 模式和直接文本 KV prefill。先前调研未覆盖到这一入口，不能将“未找到完整官方实时 RAG 方案”理解为“官方全双工没有文本注入接口”。本地 `MiniCPMO45/modeling_minicpmo_unified.py` 未包含该参数，存在版本/适配差异。源码和版本信息保存在 `official_reference/`。模型接口能力与实时调度、资料遵循训练和 Realtime 服务协议是否暴露该字段需分别判断。

## 结论

1. 已有直接相关研究和公开实现。MoshiRAG 是异步检索与压缩参考条件注入；Context Spanning 是会话中的原始文本 span prefill，最接近本项目的文本 KV 注入路线。
2. 本地确实实现了中途文本 KV 注入，不是仅更新系统提示；但系统目前是使用 duplex 模型的分段 RAG POC，回答期间没有持续处理用户音频，也不逐 unit 输出回答。
3. 正常路径的 22 个测试通过；另外 5 个审查探针复现了现有缺陷。这不能作为完整全双工正确性的证明。
4. 优先修复模型操作的所有权、任务取消和会话隔离，再设计逐 unit 音频调度。直接增加异步任务并发访问同一个 KV cache 会放大问题。

## 研究依据

### R1：MoshiRAG

- 论文：*MoshiRAG: Asynchronous Knowledge Retrieval for Full-Duplex Speech Language Models*，arXiv:2604.12928。
- 首次提交 2026-04-14；核查的 v3 为 2026-05-12。
- 原文：`https://arxiv.org/html/2604.12928v3`
- 官方代码：`https://github.com/kyutai-labs/moshi-rag`
- 摘要页：`https://arxiv.org/abs/2604.12928`
- 系统持续听说，模型预测检索触发信号，异步后端返回参考文本；参考编码器压缩文本，将条件表示与输入 embedding 相加，在多个时间步注入。
- 官方发布的是经过微调的 RAG 模型；可替换检索后端，不等于给任意原始语音模型接检索即可免训练使用。
- 论文附录比较了插入式和相加式注入；受控实验中的插入式结果较好，但受序列长度成本限制。该实验将参考资料预先注入，不能据此证明未训练的 MiniCPM 会话中途注入可靠。
- 对本项目的价值：学习异步触发、检索时延与关键事实输出时刻的配合、持续输入输出调度；模型结构和权重不能直接迁移到 MiniCPM。
- 本地核查材料：`moshirag.txt`、`moshirag_repo.txt`。

### R2：Context Spanning

- 论文：*Context Spanning: A Communication Framework for Full-Duplex Speech Models and External LLM Backends*，arXiv:2609.33443，2026-09-27 提交。
- 原文：`https://arxiv.org/html/2609.33443v1`
- 官方代码：`https://github.com/mindlogic-ai/ContextSpanning`
- 官方权重：`https://huggingface.co/mindlogicinc/context-spanning-7b`
- 基于 PersonaPlex/Moshi 系列；检索结果以 `<sos>...<eos>` 文本 span 在会话中直接 prefill，更新 KV 后继续解码，同时处理音频流和时间索引。
- 这与本地“中途把文本 embedding 写入 KV”最接近；区别在于它有专门的 span 协议、帧/位置处理及训练。本地是 `<unit> + 资料文本`，没有相应训练。
- 论文 §4.1 报告使用约 2,100 小时合成双通道数据并微调 PersonaPlex 的可训练参数，训练包含检索触发、延迟到达的 span 和拒答场景。
- 论文与仓库延迟表也有边界：600-frame span 的 prefill 加解码约 86 ms，超过其 80 ms 帧预算，不能把“单次 prefill”视为任意长度均满足实时要求。
- 对本项目的价值：优先借鉴协议、时序和训练数据设计；其权重和具体音频 token 填充规则不能直接套用 MiniCPM。
- 本地核查材料：`context_spanning.txt`、`context_repo.txt`。

### R3：ReFrame

- 论文：*ReFrame: Silence-Aware Compute Reclamation for Full-Duplex Speech Models*，arXiv:2611.01835。
- 首次提交 2026-11-01；核查的 v2 为 2026-11-09。
- 原文：`https://arxiv.org/html/2611.01835v2`
- 研究静默/监听帧的计算节省，将释放的时间预算分配给可恢复的检索及 LLM worker，并通过 Moshi 条件接口注入及时到达的参考资料。
- 与本项目相关的是调度预算、请求截止时间与迟到结果处理，不是 MiniCPM 原生 RAG 注入方法。本次没有确认到作者提供的代码仓库。
- 本地核查材料：`reframe.txt`。

### R4：MiniCPM 官方边界

- 官方 Issue：`https://github.com/OpenBMB/MiniCPM-V/issues/1110`
- Contributor 回复（2026-06-08）：`https://github.com/OpenBMB/MiniCPM-V/issues/1110#issuecomment-4645839682`
- 普通调用可由业务检索后拼接 messages；边全双工对话边检索并注入当前回答，公开回复要求业务侧工程编排，提醒其可能影响连续交互体验。
- 本次用 GitHub API 检索该仓库同时包含 RAG 和 duplex 的 issue，返回该条。这个查询不是完整的代码/分支穷举，不能作为“所有版本都绝无支持”的证明。
- 论文 R1–R3 均未在本次查阅的正文中发现 MiniCPM 实验，不能把其他模型上的效果当作 MiniCPM 的效果。
- 本地检索记录：`minicpm_search.txt`。

## 当前链路确认

```text
麦克风（浏览器按 1 秒发送）
  → VAD 判断整段结束
  → Whisper 分段 ASR
  → 字符 TF-IDF 检索
  → 构建证据文本
  → <unit> + tokenizer.encode(prompt) → decoder.feed → KV
  → 强制首 token 为 speak
  → 最多 8 个生成/续答 unit
  → 拼完全部文本和 PCM 后一次返回浏览器
```

`knowledge_injection.py:61` 的输入最终确实进入 `decoder.feed()`，且 `finalize_unit()` 会补 unit 结束并登记滑窗。这条底层注入路径成立。但强制 speak 仅控制发言，不保证模型学会资料语义、引用归属或拒答。

## 已确认的问题

### P1-1：回答期间停止消费模型音频，输出也不是逐 unit 流式

- `server.py:325`：`query_task` 未结束就直接退出 `drain_model_audio()`。
- `runtime.py:200`：循环生成全部回答 unit，将所有音频拼起来后返回；`server.py:248` 等待整个调用结束才发答案。
- `static/app.js:270`：输入按 16,000 样本发送，约 1 秒一个包。
- 后果：模型在生成回答时无法根据持续输入及时改变行为；新输入积压，随后补处理可能进一步延迟 WebSocket 中的打断消息。接收音频包或浏览器播放期间继续录音，不等于模型持续全双工。
- 复现：`test_audio_backlog_is_not_drained_while_query_active`。
- 建议：单一模型 worker 串行维护 KV，按 unit 交替执行音频 prefill / 生成 / finalize；每 unit 立即向客户端发送结果。接收、VAD 和检索在模型 worker 外运行，通过队列提交命令。真实 unit 耗时和积压必须有指标。

### P1-2：取消 asyncio task 后，模型线程仍运行并可与新操作重叠

- `runtime.py:247` 使用 `asyncio.to_thread(self._answer_sync)`。
- `server.py:205` 新问题会取消旧 task，却不等待旧线程结束；取消也会释放该 task 持有的 asyncio lock。
- 注入走 `engine.py:86`，未持有 `server.py` 的 `model_lock`。新轮注入可与旧轮生成/finalize 同时操作一个 KV、pending logits 和 TTS 状态。
- `server.py:339` 的 interrupt 设置 break，但模型 `streaming_generate()` 的 token 循环及 `_answer_sync()` 的续答循环没有读取该 break 信号。设置事件不构成停止生成。
- 连接关闭同样取消 task 后直接 cleanup，可能与尚未退出的模型线程重叠。
- 复现：`test_cancel_releases_async_lock_while_model_thread_still_runs`，旧生成被受控阻塞后，取消 task，新注入仍在旧生成结束前执行。
- 建议：所有模型状态变更通过单一 worker；任务取消使用 generation/turn ID 及协作式停止，在安全边界 finalize 或回滚，并确认 worker 完成后再接受后续变更。不要仅将 asyncio lock 包在 to_thread 外层。

### P1-3：多个连接共用同一个有状态 runtime，锁却按连接创建

- `server.py:362` 在 app lifespan 创建一个 runtime；`server.py:405` 将其传给每个 WebSocket。
- `server.py:125` 每个连接自己的 lock 不能串行化跨连接操作。
- `runtime.py:126` 已有 `_view` 时 prepare 直接返回，新连接没有实际重置会话或应用新 prompt。
- `server.py:203` 任一连接退出就关闭共享 runtime。
- 后果：不同连接混用 KV/证据/TTS 状态；一个连接断开可清理其他连接正在使用的会话。即使只允许一个用户，先完成回答后正常重连也会遇到重新加载模型等生命周期成本。
- 复现：`test_prepare_existing_view_does_not_reset_session` 确认已有 view 时新 prepare 没有调用 view.prepare。跨连接共享由代码对象传递确认，未用真实 GPU 跑并发破坏实验。
- 建议：先明确单会话占用策略并拒绝第二个活跃会话；模型加载与会话创建/重置/清理分离。要支持多会话则隔离完整 KV 和 TTS 状态，并由全局调度器管理模型。

### P2-1：gate 在注入完成前就 ready，回答后无法复位

- `engine.py:78` 在 await 注入前调用 `mark_injected()`。
- `gate.py:46` 直接设 ANSWER_READY；注入失败也无回滚。
- `gate.py:53` 的 `consume_answer_ready()` 没有被服务调用；`finish_speaking()` 仅处理 SPEAKING，因此正常完成后仍停在 ANSWER_READY。
- 注入中发生 interrupt 后，pipeline 没有再次验证 turn，仍返回 `stale=False`。
- 目前服务多数控制依赖 query_task，掩盖了 gate 错误；一旦按 gate 实现真正双工调度，就可能提前放行。
- 复现：`test_answer_completion_leaves_gate_answer_ready`、`test_interrupt_during_injection_returns_nonstale_result`。
- 建议：RETRIEVING → INJECTING →（worker 确认注入且 turn 有效）ANSWER_READY → SPEAKING → LISTENING，定义失败和取消出口。状态校验本身不能撤销已经写入 KV 的旧资料，应与 worker 版本控制一起修复。

### P2-2：用户开始说话时没有自动撤销回答/检索

- `server.py:306` 附近 VAD 开始说话只发送状态；直到完整新语段结束，才执行 `start_query()`。
- 前端 `vad.state` 不停止播放；新 text.query 也不主动停止上一段已排队音频。
- 目前明确有效的是按钮停止浏览器播放，不代表模型即时停止，也不等于自然插话打断。
- 建议：VAD start / 模型交互信号触发 turn 失效和播放取消；输出携带 response_id，浏览器拒绝播放已取消轮次。避免将所有短 backchannel 都当作强打断，需要有明确策略。

### P2-3：音频时间索引和 synthetic knowledge unit 混用

- `knowledge_injection.py:78` 每次文本注入和续答都增加 `audio_chunk_idx`。
- `modeling_minicpmo_unified.py:4928` 后续真实音频的首包补齐、chunk metadata、prefix frames 和消费量依赖该索引；`current_time` 也使用该值。
- 多次续答没有真实用户音频，却增加音频时间；文本提问先于首个麦克风包时还会提前跳过 `audio_chunk_idx == 0` 的处理分支。
- 索引混用由代码确认；是否导致特定输入的首包失败、音频错位或实际质量下降，本次未加载真实模型测量。
- 建议：拆分 unit 序号、真实输入音频 chunk 索引和 wall-clock 时间，保持流式音频处理器的首包契约。

## 需要真实模型补测的风险

1. **语音是否完整表达文本。** knowledge unit 将文本上限提高到 96 tokens / 220 chars，但 TTS 仍是最多 26 audio tokens 每 unit（`modeling_minicpmo_unified.py:5292`）。不能仅靠“文本包含六百、audio 非空”证明完整播报。需将输出音频独立 ASR 回转，并验证末尾数字、否定词和结论。
2. **人为结束与模型 turn 结束不一致。** `should_continue_knowledge_answer()` 可在出现标点或达到 8 units 时停止，即使 `end_of_turn=False`。下一轮 force-listen 可能关闭 turn，但服务已经发 answer.done，TTS 和未播内容是否被截断需检查。
3. **未经训练的资料协议。** 当前 prompt 没有独立可信指令/不可信证据边界，资料中的指令和模型原有知识可能干扰回答；force-speak 不能解决。应测试冲突资料、拒答、跨轮更新、精确数字和证据归属。
4. **证据长期保留及预算。** 多个 continuation 重复注入资料，占用 KV 和 prefill 时间；滑窗只把生成文本记为保留文本，资料是否在长会话中仍可用需测试。`RAG_MAX_EVIDENCE_CHARS` 定义为 900，但 pipeline 未传入该值，实际使用 `build_evidence_card()` 默认 1200 chars。还需按 token 和时间控制预算。
5. **检索/ASR 边界。** 字符 TF-IDF 是可接受的离线基线，但对同义表达、ASR 错字、短问题和多轮省略未验证；空 ASR 会得到“无依据”卡片并强制回答，宜跳过空语段。不要先把这些效果问题归因于 KV 注入。

## 建议实施顺序

1. 会话占用与 worker 所有权：先确保没有并发改同一份 KV/TTS；修复关闭和取消等待。
2. turn/response ID 贯穿 ASR、检索、注入、生成和播放；定义过期与失败处理。
3. 回答逐 unit 输出；输入持续处理；VAD/打断控制不被模型调用阻塞；音频队列有时限与积压指标。
4. 分离真实音频时间与知识 unit 索引；修复 gate。
5. 测量未训练 MiniCPM 的质量上限，决定是否采用参考 Context Spanning 的 span 训练。训练研究可尝试 LoRA 等较小实验，但它在此模型上的效果尚未验证，不能承诺免训练或小成本即可达到论文效果。

建议评估至少覆盖：

- 资料更新导致答案与模型记忆冲突（600→437）、否定条件、长数字和资料不足；
- 连续多轮同主题不同资料，以及滑窗前后；
- 检索/注入/LLM/TTS/finalize/浏览器播放各阶段插话；
- 新问题、断线、重连、第二连接和慢检索；
- 文本事实正确率、音频回转事实正确率、TTFA、插话停止时延、输入 backlog age、过期结果丢弃数、KV 增长和每次注入耗时。

## 验证结果与边界

现有测试：

```bash
cd /root/paddlejob/workspace/env_run/avatar_full/minicpm_rag_test
../.venv/bin/python -m pytest tests -q
```

结果：22 passed in 3.12s。

新增审查探针：

```bash
../.venv/bin/python -m pytest review_20261116/test_review_reproductions.py -q
```

结果：5 passed in 0.75s。**这些断言用于确认现存缺陷行为，测试通过不表示问题已修复；修复后应将期望改成正确行为。**

本次没有启动 9B 模型、没有重新跑浏览器端测试，没有验证论文性能数字。既有 README 的真实模型通过记录仅作为历史背景，不作为本次真实运行证据。

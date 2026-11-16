最终报告：REVIEW.md。
研究：MoshiRAG 异步参考 embedding 相加，需要经过训练的模型；Context Spanning 实时文本 span prefill，专门训练，最接近当前路线；ReFrame 是推理预算与参考注入调度。没有确认 MiniCPM 官方提供现成全双工RAG。
本地：真实KV注入路径成立，但整体为分段硬门禁；确认线程取消不停止、共享有状态runtime、gate时序/复位问题、回答期音频积压；音频索引混用和TTS完整性需要真实模型补测。

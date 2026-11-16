核查时间：2026-11-16
官方仓库：openbmb/MiniCPM-o-4_5（Hugging Face）
元数据 revision：fb09628e98b1f55d8d5b863e0eccfcdd11bff499
来源：`https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/modeling_minicpmo.py`
本地文件为本次读取 main 的原文快照。
关键更正：MiniCPMODuplex.streaming_prefill 已包含 text_list（第2781行），支持 TEXT 模式和直接文本KV prefill（第3116行）。本地 unified 适配不包含此入口。模型层文本入口不等于服务协议、训练保证和完整RAG方案。

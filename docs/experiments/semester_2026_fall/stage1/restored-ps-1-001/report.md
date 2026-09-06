# restored-ps-1-001

状态：通过

Single-process wall time includes model/data construction and 20-step history (or configured steps); PS elapsed time also includes spawn, IPC and cleanup. These are one-run demonstration timings, not a scaling benchmark.

已将每次更新与单进程 FP64 MBGD 对照，包含第 1 步和第 20 步；参数最大绝对差为 0.

Worker 不直接更新权重；server 按各 shard 的实际样本数对梯度加权平均。Launcher 清理结果和各进程退出码记录在 results.json。

本目录仅保留摘要报告；实验原始文件已按交付清理策略删除。

# performance-001

状态：通过

所有输入、权重和偏置均来自相同的 FP32 fixture；误差使用独立的 FP64 循环参考实现计算，相对误差分母为 max(abs(reference), 1e-8)。

GPU event intervals cover the whole operator call and may include GPU idle time while Python submits work. These are not kernel-only measurements. Profiler timings are excluded.

本目录仅保留摘要报告；实验原始文件已按交付清理策略删除。

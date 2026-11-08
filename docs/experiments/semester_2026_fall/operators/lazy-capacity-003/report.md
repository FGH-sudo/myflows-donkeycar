# WDDM 显存驻留边界复测

另用 cifar stem、w16 逐步增大 batch，在首次出现非本地显存后按整数二分。容量试验先上传固定设备批次并同步，再记录 WDDM 的进程 Local Usage 和 Non Local Usage。容许相对该基准增加 1 MiB 的计数波动。
- before 的最大驻留 batch 为 37，相邻的 38 已超出。
- after 的最大驻留 batch 为 46，相邻的 47 已超出。

原始测量与协议保存在本目录的 JSON 文件中。

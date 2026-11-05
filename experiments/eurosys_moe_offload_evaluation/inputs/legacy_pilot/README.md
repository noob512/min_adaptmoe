# Legacy pilot inputs

这些文件是旧 runner 仍需读取的不可变 token fixture、calibration statistics 和固定
placement plan。它们从历史 `artifacts/` 中迁出，是输入而非论文实验结果。

它们不得用于 EuroSys 正式性能数据。正式运行必须依据新 protocol 重新划分
calibration/validation/evaluation split，重新生成专家统计和计划，并写入相邻的
`../generated/` 目录。

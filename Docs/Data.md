# TORCH.UTILS.DATA
支持以下功能
- 映射格式，迭代格式的数据集
- 定义数据的装载顺序
- 自动分批
- 多进程数据载入

## Dataset Types
- map-style
- iterable-style
  - 是iterableDataset的子类
  - 实现__iter__()方法
 
## 关闭自动批导入数据
- batch_size, batch_sampler都是None即关闭

##collate_fn功能
- 当自动批导入关闭
  - 仅仅将numpy数组转换为tensors
- 当自动批导入打开
  - 数据聚集成batch，导出
  
## 多进程数据导入
python存在GIL（Global Interpreter Lock），pytorch采用多进程的方式

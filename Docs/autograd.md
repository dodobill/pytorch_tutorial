# Torch.autograd
## 叶子节点
- a = torch.rand(10, requires_grad=True)
  - 创建时申明requires_grad=true，为叶子
- b = torch.rand(10, requires_grad=true).cuda()
  - 跟踪梯度，通过cuda方法生成，故不是叶子
- d = torch.rand(10).cuda()
  - 不跟踪梯度，使用cuda仍是叶子
## 反向传播
- backward()
  - 计算当前张量相关叶子的梯度
- register_hook(lambda x:f(x))
  - 对最后的grad进行变换，将变化注册到叶子上
## Profiler
- **torch.autograd.profiler.profile**
  - 计算每一个方法的时空开销
## Anomaly detection
- **torch.autograd.detect_anomaly**
  - 当有反向传播方法出错时，爆出错误

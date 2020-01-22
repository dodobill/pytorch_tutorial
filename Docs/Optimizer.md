# Optimizer
## 优化器进行一次更新
1. 某些优化算法需要多次评估模型，需要传递一个closure给函数optimizer(closure)
```python
for input, target in dataset:
    def closure():
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        return loss
    optimizer.step(closure)
```
2. **torch.optim.Adadelta**
    - 见论文 An Adaptive Learning Rate Method <https://arxiv.org/abs/1212.5701>
 
3. **torch.optim.Adagrad**
    - <http://jmlr.org/papers/v12/duchi11a.html>
    
4. **torch.optim.Adam -->AdamW, Adamax**
    - <https://arxiv.org/abs/1412.6980>
    - <https://arxiv.org/abs/1711.05101>
    - <https://arxiv.org/abs/1412.6980>
    
5. **torch.optim.ASGD**
    - 均值随机梯度下降
    - <http://dl.acm.org/citation.cfm?id=131098>
    
## 如何调整学习率
1. torch.optim.lr_scheduler包含了需索调整学习率的方法

2. **LambdaLR**
    - 学习率随训练epoch而变化

3. **StepLR**
    - 学习率随训练epoch而变化
    
4. **ReduceLROnPlateau**
    - 当模型停止变优后，减少学习率

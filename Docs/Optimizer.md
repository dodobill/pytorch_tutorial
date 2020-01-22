# Optimizer
## 优化器进行一次更新
- 某些优化算法需要多次评估模型，需要传递一个closure给函数optimizer(closure)
```
for input, target in dataset:
    def closure():
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        return loss
    optimizer.step(closure)
```


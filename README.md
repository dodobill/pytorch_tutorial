# pytorch_tutorial
## TORCH
### Tensors
1. torch.is_tensor(obj) -->is_storage, is_floating_point
    - 返回真，如果为tensor类型
    1. 参数
        - obejct类
2. torch.set_default_dtype(d) -->get_default_dtype
    - 修改默认的数值类型
    - 通常默认的类型为torch.float32
    1. 参数
        - torch.dtype
    2. 例子
        - ```python
        torch.tensor([1.2,3]).dtype
        torch.set_default_dtype(torch.float64)
        ```

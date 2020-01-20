# pytorch_tutorial
## TORCH
### Tensors
1. **torch.is_tensor(obj) -->is_storage, is_floating_point**
    - 返回真，如果为tensor类型
    1. 参数
        - obejct类
        
2. **torch.set_default_dtype(d) -->get_default_dtype**
    - 修改默认的数值类型
    - 通常默认的类型为torch.float32
    1. 参数
        - torch.dtype
    2. 例子
        ```python
        torch.tensor([1.2,3]).dtype
        torch.set_default_dtype(torch.float64)
        ```
        
3. **torch.numel(input)**
    - 返回tensor中的元素个数
    
4. **torch.set_flush_denormal(mode)**
    - 是否将非正规浮点数归0
    
5. **torch.tensor()**
    - 当x是tensor类型，使用torch.tensor(x)等价于x.clone().detach()
    
6. **torch.sparse_coo_tensor()**
    - 构造COO稀疏矩阵
    1. 参数
        - indices 当前坐标对应的数值索引
        - values 索引对应的值
        - size 矩阵的大小
        
7. **torch.as_tensor()**
    - 避免复制
    
8. **torch.as_strided()**
    - 按步长展示矩阵
    
9. **torch.from_numpy(ndarray)**
    - 以numpy数组创建tensor
    - tensor与array共享内存
    
10. **torch.zeros() -->ones**
    - 零矩阵
    
11. **torch.arange() -->range, linspace, logspace**
    - 生成间隔数 [start,end)

12. **torch.eye() -->empty, empty_like**
    - 生成对角线全为1的矩阵

13. **torch.full() -->full_like**
    - 矩阵全是相同的值

14. **torch.quantize_per_tensor() -->per_channel**
    - 离散化函数

###indexing, slicing, joining, mutating ops
1. **torch.cat()**
    - 在某一维度将tensor拼接起来
    
2. **torch.chunk()**
    - 切分为多个块
    
3. **torch.index_select()**
    - 从当前矩阵中选择元素
    - 不共享
   

# pytorch_tutorial
## TORCH
### *Tensors*
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
    - 生成间隔数 \[start,end)

12. **torch.eye() -->empty, empty_like**
    - 生成对角线全为1的矩阵

13. **torch.full() -->full_like**
    - 矩阵全是相同的值

14. **torch.quantize_per_tensor() -->per_channel**
    - 离散化函数

### *indexing, slicing, joining, mutating ops*
1. **torch.cat()**
    - 在某一维度将tensor拼接起来
 
2. **torch.chunk()**
    - 切分为多个块
   
3. **torch.index_select()**
    - 从当前矩阵中选择元素
    - 不共享
   
4. **torch.masked_select()**
    - 通过bool矩阵，取出矩阵中的元素
    - 不共享

5. **torch.nonzero()**
    - 给出矩阵中非零值的坐标
    1. as_tuple=False
        - 矩阵中每一行代表一个坐标
    2. as_tuple=True
        - 多个一维度数组组成的元组，分别代表各维度的下标哥

6. **torch.reshape()**
    - 可能返回视图，也可能返回副本
    
7. **torch.split()**
    - 切分
 
8. **torch.squeeze() -->unsqueeze**
    - 去除值为1的维度
    - (Ax1xBxCx1xD)-->(AxBxCxD)
    - 共享内存
    
9. **torch.stack()**
    - 在新维度上拼接张量（维度增加）
    - 所有张量结构相同
    
10. **torch.t()**
    - 转置

11. **torch.take()**
    - 取输入矩阵对应位置的元素
    - 索引将输入矩阵看成一维
    - 结果的结构与索引矩阵相同
    
12. **torch.unbind()**
    - 沿着指定维度均分矩阵
    - 结果保存在元组中
    
13. **torch.where(condition,x,y)**
    - if condtion x else y
    - x,y结构相同
     
### *Generators*
    





















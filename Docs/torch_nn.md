# TORCH.NN
## Parameters
1. **CLASS torch.nn.Parameter**
  - 为Tensor的子类
  - 当在module中增加parameter属性时，自动添加到网络的参数中
  - 可通过module.parameters()查看
  
## Containers
### *Module*
1. **CLASS torch.nn.Module**
  - 自定义的模版类必须继承该类
  - module可以嵌套，其他module可以作为当前module的属性
  - add_module(name,module)
    - 将子模版加入当前模版
  - apply(fun)
    - 将某一方法应用到所有子模版中（常用于初始化）
  - children()
    - 返回所有子模版
  - cpu（）
    - 将模型所有参数转移到cpu中
  - double（）
    - 将所有单精度转为双精度
  - eval（）
    - 设置模型为评估态
  - extra_repr()
    - 重写可自定义模型打印信息
  - *forward(input)*
    - 定义计算方式
  - load_state_dict()
    - 复制参数到当前模型
  - parameters()：参与反向传播的 buffer()：不参与反向传播的参数
    - 返回模型的所有参数
    - 主要作为参数传递给优化器
  - register_backward_hook(hook) -->register_forward_hook()
    - 注册一个反向传播钩子
  - register_parameter()
    - 将参数加入到模型当中
  - state_dict()
    - 返回模型中所有的parameter和buffer
    
### *Sequential*
1. 按顺序将模型传入构造器

### *ModuleList -->ModuleDcit*
1. self.linears = nn.ModuleList([nn.Linear(10,10) for i in range(10)])

### *ConvXd -->ConTransposeXd*
1. 卷积
2. 反卷积

### *Padding*

### *激活函数*

### *MultiheadAttention*
  - 谷歌论文，多头注意力机制
  - MultiHead(Q,K,V) = Concat(head1,...,headh)W whereheadi = Attention(QWi,KWi,VWi)
  
### *Normalization layers*
1. BatchNorm1d -->2d, 3d

### *RNN*
1. **torch.nn.LSTM**
  - (input_size,hidden_size,num_layers)
  
### *LSTMCell -->RNN*
cell一次仅计算序列中的一个，而完整的循环神经网络自动完成连续传递

### *Tranformer layers*
1. **CLASS torch.nn.Transformer()**
  - 见谷歌的论文Attention Is All You Need
  
2. **CLASS torch.nn.TransformerEncoder() -->TransformerEncoderLayer()**
  - transformer结构的encoder层
  
3. **nn.Linear()**
  - 线性层
  
### *Dropout Layers*
- dropout, dropout2d, dropout3d

### *Embedding*
- 存放词嵌入
- 参数
  - num-embeddings：字典大小
  - embedding-dim：词嵌入维度
  - padding-idx：如果给出，则使用该索引作为padding向量，该向量的梯度始终为0
  - max_norm：所有向量的范数不能大于改值
  - scale_grad_by_freq：根据该词语出现的频次，反比例缩放更新梯度

### *Loss function*
1. **L1Loss**
  - 输入和目标差值绝对值的平均或总和
  
2. **MSELoss**
  - 输入和目标差值平方的平均或总和
  
3. **CrossEntropyLoss**
  - 交叉熵
  - loss(x, class) = -log(exp(x\[class])/sum(exp(xi)))
  
4. **CTCLoss**
  - 不要求结果一一对应真确，序列正确即可
  - 常用于语音识别
  - 论文见<https://www.cs.toronto.edu/~graves/icml_2006.pdf>
  - 配合循环神经网络使用效果更佳
  
5. **NLLLoss**
  - 对于C个个类的分类问题特别有用
  
6. **KLDivLoss**
  - L = {l1,...ln}, ln = yn*(logyn-xn)
  - 对L求和或者均值
  
7. **SmoothL1Loss**
  - 损失值计算更加平滑
  - 分差值绝对值大于1和小于1的情况
  
8. **MultiMarginLoss**
  - 优化多分类问题
  - 结果是batch_size个概率分布, 目标是一个一维张量表示索引
  
9. **PixelShuffle**
  - 变化矩阵，*实现亚卷积*
  
### *Utilities*
1. **clip_grad_norm_ --> value**
  - 修剪梯度的范数
  - 修剪梯度的值
  

  
  









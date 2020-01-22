# Creating TorchScript Code
- 将模型从python代码转换为torchscript，可以在c++的环境下运行
- 应用于实际的开发场景中
- 可通过torch.jit.script直接书写torchscript
- 可通过trace的方式，从python自动生成torchscript

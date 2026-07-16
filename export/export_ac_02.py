import torch

# 加载第一个 JIT 模型
model1 = torch.jit.load("../logs/CAS02_LB_ppo/0_exported/policies/policy_1.pt")
model1.eval()

# 创建第一个模型的示例输入（根据模型需求调整大小和类型）
example_input1 = torch.randn(1,705)  # 假设这是适合第一个模型的输入


# 导出第一个模型
torch.onnx.export(model1,               # 第二个 JIT 模型
                  example_input1,       # 第二个模型的示例输入
                  "02_policy.onnx",    # 第二个模型的输出文件名
                  export_params=True,   # 导出模型的参数
                  opset_version=11,     # ONNX opset 版本
                  do_constant_folding=True,  # 优化常量折叠
                  input_names=['input'],    # 输入名
                  output_names=['output'],  # 输出名
                  )
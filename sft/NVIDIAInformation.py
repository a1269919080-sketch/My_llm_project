import torch

print("="*40)
print("🔍 PyTorch 核心信息")
print("="*40)

# 1. 查看 PyTorch 版本号
print(f"PyTorch 版本: {torch.__version__}")

# 2. 查看 PyTorch 编译时使用的 CUDA 版本
# 如果显示 None，说明你装的是纯 CPU 版本（无法使用显卡）
print(f"PyTorch 关联的 CUDA 版本: {torch.version.cuda}")

# 3. 最关键：检查显卡是否可用
# 如果返回 True，恭喜你，显卡驱动和 PyTorch 完美匹配！
is_available = torch.cuda.is_available()
print(f"显卡是否可用: {is_available}")

if is_available:
    print("="*40)
    print("✅ 显卡详细信息")
    print("="*40)
    # 4. 查看显卡名称（确认是不是你的 5060）
    print(f"显卡名称: {torch.cuda.get_device_name(0)}")
    
    # 5. 查看显卡算力架构
    # RTX 5060 应该是 sm_120 (Compute Capability 12.0)
    props = torch.cuda.get_device_properties(0)
    print(f"显卡算力: {props.major}.{props.minor}")
else:
    print("❌ 警告：PyTorch 无法访问显卡！请检查驱动或重装支持 CUDA 的 PyTorch。")

import torch
import matplotlib.pyplot as plt

orgin_path = "debug.pt"
target_path = "debug2.pt"
origin = torch.load(orgin_path)
target = torch.load(target_path)

# 展平并转 numpy
o = origin.flatten().cpu().numpy()
t = target.flatten().cpu().numpy()

# 统计
same = (o == t).sum()
total = len(o)
print(f"总元素数: {total}")
print(f"相同: {same} ({same/total*100:.2f}%)")
print(f"不同: {total - same} ({(total-same)/total*100:.2f}%)")
print(f"最大绝对误差: {abs(o - t).max():.6e}")
print(f"平均绝对误差: {abs(o - t).mean():.6e}")

# 折线图：取前 1000 个点对比
plt.figure(figsize=(12, 5))
plt.plot(o[:1000], label='origin', alpha=0.7)
plt.plot(t[:1000], label='target', alpha=0.7, linestyle='--')
plt.xlabel('index')
plt.ylabel('value')
plt.legend()
plt.title('前1000个元素对比')
plt.show()

# 误差折线图
plt.figure(figsize=(12, 4))
plt.plot(abs(o - t)[:1000], color='red', alpha=0.6)
plt.xlabel('index')
plt.ylabel('|diff|')
plt.title('绝对误差 (前1000个)')
plt.show()
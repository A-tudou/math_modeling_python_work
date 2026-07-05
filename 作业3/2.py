import numpy as np
import matplotlib.pyplot as plt

# ========== 中文乱码修复 ==========
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ========== 基础参数 ==========
demand = [40, 60, 80]   # 1/2/3季度交货需求
a_base = 50
b_base = 0.2
c_base = 4
max_prod = 100          # 单季度最大产能

# ---------------------- 函数：给定a,b,c，计算最小总费用与最优生产库存方案 ----------------------
def calculate_min_cost(a, b, c):
    min_total_cost = float('inf')
    best_x1 = 0
    best_I1 = 0

    # 第三季度成本函数
    def stage3(I2):
        x3 = max(demand[2] - I2, 0)
        x3 = min(x3, max_prod)
        I3 = I2 + x3 - demand[2]
        cost = a * x3 + b * (x3 ** 2) + c * I3
        return cost, x3, I3

    # 第二季度成本函数
    def stage2(I1):
        min_c2 = float('inf')
        best_x2 = 0
        best_I2 = 0
        for x2 in range(0, max_prod + 1):
            I2 = I1 + x2 - demand[1]
            if I2 < 0:
                continue
            c3, _, _ = stage3(I2)
            total = a * x2 + b * (x2 ** 2) + c * I2 + c3
            if total < min_c2:
                min_c2 = total
                best_x2 = x2
                best_I2 = I2
        return min_c2, best_x2, best_I2

    # 遍历一季度所有可行产量
    for x1 in range(0, max_prod + 1):
        I1 = x1 - demand[0]
        if I1 < 0:
            continue
        cost2, _, _ = stage2(I1)
        total = a * x1 + b * (x1 ** 2) + c * I1 + cost2
        if total < min_total_cost:
            min_total_cost = total
            best_x1 = x1
            best_I1 = I1

    # 回溯完整方案
    _, best_x2, best_I2 = stage2(best_I1)
    _, best_x3, best_I3 = stage3(best_I2)
    return min_total_cost, [best_x1, best_x2, best_x3], [best_I1, best_I2, best_I3]

# ========== (1) 求解基准参数下最优生产计划 ==========
min_cost, x_opt, I_opt = calculate_min_cost(a_base, b_base, c_base)
print("===== (1) 基准参数下最优生产库存方案 =====")
print(f"季度产量：一季度{x_opt[0]}台，二季度{x_opt[1]}台，三季度{x_opt[2]}台")
print(f"季末库存：一季度{I_opt[0]}台，二季度{I_opt[1]}台，三季度{I_opt[2]}台")
print(f"满足全部交货需求，最小总费用：{min_cost:.2f} 元")

# ========== (2) 生成灵敏度绘图数据：a、b、c分别变化对总费用的影响 ==========
# 1. 参数a（单位固定生产成本）变化
a_range = np.linspace(30, 80, 25)
cost_a = []
for ai in a_range:
    tc, _, _ = calculate_min_cost(ai, b_base, c_base)
    cost_a.append(tc)

# 2. 参数b（二次生产成本系数）变化
b_range = np.linspace(0.05, 0.5, 25)
cost_b = []
for bi in b_range:
    tc, _, _ = calculate_min_cost(a_base, bi, c_base)
    cost_b.append(tc)

# 3. 参数c（单位季度存储费）变化
c_range = np.linspace(0, 12, 25)
cost_c = []
for ci in c_range:
    tc, _, _ = calculate_min_cost(a_base, b_base, ci)
    cost_c.append(tc)

# ========== 绘制三张子图，分别展示a、b、c对总费用的影响 ==========
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), dpi=100)

# 图1：a与总费用
ax1.plot(a_range, cost_a, color="#d62728", linewidth=2)
ax1.set_xlabel("单位固定生产成本 a", fontsize=11)
ax1.set_ylabel("最小总费用（元）", fontsize=11)
ax1.set_title("参数a变化对总费用的影响", fontsize=13)
ax1.grid(alpha=0.3)

# 图2：b与总费用
ax2.plot(b_range, cost_b, color="#2ca02c", linewidth=2)
ax2.set_xlabel("二次生产成本系数 b", fontsize=11)
ax2.set_ylabel("最小总费用（元）", fontsize=11)
ax2.set_title("参数b变化对总费用的影响", fontsize=13)
ax2.grid(alpha=0.3)

# 图3：c与总费用
ax3.plot(c_range, cost_c, color="#1f77b4", linewidth=2)
ax3.set_xlabel("单位季度存储费 c", fontsize=11)
ax3.set_ylabel("最小总费用（元）", fontsize=11)
ax3.set_title("参数c变化对总费用的影响", fontsize=13)
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# 参数
r = np.array([0.05, 0.28, 0.21, 0.23, 0.25])
q = np.array([0.00, 0.025, 0.015, 0.055, 0.026])
p = np.array([0.00, 0.01, 0.02, 0.045, 0.065])
M = 1.0

risk_list = []
profit_list = []
k_range = np.linspace(0.05, 0.27, 80)

for k in k_range:
    # 变量顺序：[x0,x1,x2,x3,x4, V]，目标min V
    c = np.array([0, 0, 0, 0, 0, 1])

    # 不等式约束 A_ub @ x <= b_ub
    A_ub = []
    b_ub = []
    # q_i * x_i - V <=0
    for i in range(1, 5):
        row = [0] * 6
        row[i] = q[i]
        row[5] = -1
        A_ub.append(row)
        b_ub.append(0)
    # -(r-p)x <= -kM  收益约束 >=kM
    row_profit = [- (r[i] - p[i]) for i in range(5)] + [0]
    A_ub.append(row_profit)
    b_ub.append(-k * M)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # 等式约束 A_eq @ x = b_eq
    A_eq = [[(1 + p[i]) for i in range(5)] + [0]]
    b_eq = np.array([M])

    # 变量上下限 x0~x4>=0, V>=0
    bounds = [(0, None) for _ in range(6)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.success:
        risk_list.append(res.x[5])
        profit_list.append(k)

# 绘图
plt.rcParams["font.family"] = ["SimHei"]
plt.figure(figsize=(10, 6))
plt.plot(profit_list, risk_list, lw=2.5, color="#2E86AB", label="风险-收益有效前沿")
plt.scatter(profit_list, risk_list, s=10, c="#A23B72", alpha=0.7)
plt.xlabel("最低期望收益率 k")
plt.ylabel("最小整体风险 V")
plt.title("模型二：固定收益极小化风险 风险收益关系曲线")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

print(f"起点 k={profit_list[0]:.4f}, 风险V={risk_list[0]:.6f}")
print(f"终点 k={profit_list[-1]:.4f}, 风险V={risk_list[-1]:.6f}")
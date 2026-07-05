import numpy as np
import matplotlib.pyplot as plt

# ---------------------- 中文乱码修复 ----------------------
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------- 基础参数 ----------------------
price = np.array([20, 25, 30])
earn = np.array([5, 8, 10])
sigma_single = np.array([2, 6, 10])
rho = np.array([
    [1, 5/24, -0.5],
    [5/24, 1, -0.25],
    [-0.5, -0.25, 1]
])
cov_matrix = np.outer(sigma_single, sigma_single) * rho
total_fund = 500000
min_return_rate = 0.2

# 单只股票回报率
single_rate = earn / price
print("单只股票回报率：")
print(f"A: {single_rate[0]:.2%}, B: {single_rate[1]:.2%}, C: {single_rate[2]:.2%}")

# 1. 最优投资方案（全仓A）
x_opt = np.array([total_fund / price[0], 0, 0])
total_cost = price @ x_opt
total_earn = earn @ x_opt
real_rate = total_earn / total_cost
port_variance = x_opt.T @ cov_matrix @ x_opt
port_std = np.sqrt(port_variance)

print("\n===== (1) 最优低风险投资方案 =====")
print(f"A股：{x_opt[0]:.0f} 股，B股：0 股，C股：0 股")
print(f"投入资金：{total_cost:.0f} 元，年收益：{total_earn:.0f} 元")
print(f"回报率：{real_rate:.2%}，组合总风险标准差：{port_std:.2f}")

# ---------------------- 2. 遍历权重生成全部混合组合（单位资金，解决散点扎堆） ----------------------
risk_list = []
ret_list = []
# 单独记录三个单资产点
single_risk = []
single_ret = []
for i in range(3):
    w = np.zeros(3)
    w[i] = 1
    r = w @ (earn / price)
    var = w.T @ cov_matrix @ w
    std = np.sqrt(var)
    single_risk.append(std)
    single_ret.append(r)

# 遍历全部混合权重，步长缩小到0.005，采样更多组合
step = 0.005
for w1 in np.arange(0, 1 + step, step):
    for w2 in np.arange(0, 1 - w1 + step, step):
        w3 = 1 - w1 - w2
        w = np.array([w1, w2, w3])
        r = w @ (earn / price)
        var = w.T @ cov_matrix @ w
        std = np.sqrt(var)
        if r >= min_return_rate:
            risk_list.append(std)
            ret_list.append(r)

# ---------------------- 绘图（分层展示，区分混合组合+单资产） ----------------------
plt.figure(figsize=(12,7), dpi=110)
# 所有混合投资组合（浅蓝色小点）
plt.scatter(risk_list, ret_list, c="#639bf0", s=3, alpha=0.5, label="A/B/C混合投资组合")
# 三个单股票资产点
plt.scatter(single_risk[0], single_ret[0], c="red", s=130, zorder=10, label="全仓A（最优低风险点）")
plt.scatter(single_risk[1], single_ret[1], c="orange", s=100, zorder=9, label="全仓B")
plt.scatter(single_risk[2], single_ret[2], c="darkgreen", s=100, zorder=9, label="全仓C（最高收益）")

plt.xlabel("单位资金投资风险（收益标准差）", fontsize=12)
plt.ylabel("年度投资回报率", fontsize=12)
plt.title("投资回报率与风险关系图（马科维茨可行集）", fontsize=14)
plt.legend(loc="upper left")
plt.grid(alpha=0.3)
plt.show()
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize
from scipy.integrate import quad

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)



# 一、核心几何函数

def elliptical_segment_area(h, a, b):

    h = np.asarray(h, dtype=float)
    ratio = np.clip((b - h) / b, -1.0, 1.0)
    theta = np.arccos(ratio)
    return a * b * (theta - np.sin(theta) * np.cos(theta))


def circular_segment_area(h, R):
    h = np.asarray(h, dtype=float)
    R = np.asarray(R, dtype=float)
    ratio = np.clip((R - h) / R, -1.0, 1.0)
    theta = np.arccos(ratio)
    return R * R * (theta - np.sin(theta) * np.cos(theta))

# 二、问题一：小椭圆型储油罐

def small_tank_volume_no_shift(h, a, b, L):
    S = elliptical_segment_area(h, a, b)
    return S * L / 1000.0


def small_tank_volume_shift(h, a, b, L, alpha_deg, n_seg=2000):
    alpha = np.deg2rad(alpha_deg)
    x = np.linspace(-L / 2.0, L / 2.0, n_seg)
    dx = x[1] - x[0]
    hx = h - x * np.tan(alpha)
    S = elliptical_segment_area(hx, a, b)
    return np.sum(S) * dx / 1000.0


def fit_ellipse_params(data_no_in, init_oil=262.0):
    a0, b0, L0 = 40.5, 25.0, 240.0

    def residuals(p):
        a, b, L = p
        if a <= 0 or b <= 0 or L <= 0:
            return np.full(data_no_in.shape[0], 1e6)
        h_cm = data_no_in[:, 1] / 10.0  # mm -> cm
        V_model = small_tank_volume_no_shift(h_cm, a, b, L)
        V_exp = init_oil + data_no_in[:, 0]
        return V_model - V_exp

    res = least_squares(residuals, x0=[a0, b0, L0],
                        bounds=([30, 15, 200], [60, 40, 300]))
    return res.x


def generate_capacity_table_small(a, b, L, alpha_deg=None, h_max_mm=None,
                                  step_mm=10):
    if h_max_mm is None:
        h_max_mm = int(2 * b * 10)
    h_mm = np.arange(0, h_max_mm + step_mm, step_mm)
    h_cm = h_mm / 10.0
    if alpha_deg is None:
        V = np.array([small_tank_volume_no_shift(h, a, b, L) for h in h_cm])
        tag = "无变位"
    else:
        V = np.array([small_tank_volume_shift(h, a, b, L, alpha_deg)
                      for h in h_cm])
        tag = f"变位α={alpha_deg}°"
    df = pd.DataFrame({"油位高度(mm)": h_mm,
                       "理论储油量(L)": np.round(V, 3)})
    df.attrs["tag"] = tag
    return df

# 三、问题二：实际储油罐（圆柱 + 两端球冠）

R_BIG = 150.0
L1_BIG = 800.0
r_CAP = 163.0
h0_CAP = 38.0
d_CAP = np.sqrt(r_CAP ** 2 - R_BIG ** 2)  # 球冠几何参数 d = √(r²-R²)


def cap_section_radius(x_local, side="left"):
    if side == "left":
        rho2 = r_CAP ** 2 - (d_CAP - x_local) ** 2
    else:
        rho2 = r_CAP ** 2 - (d_CAP - (h0_CAP - x_local)) ** 2
    rho2 = np.clip(rho2, 0.0, R_BIG ** 2)
    return np.sqrt(rho2)


def big_tank_volume(h_cm, alpha_deg, beta_deg, n_cyl=2000, n_cap=500):
    alpha = np.deg2rad(alpha_deg)
    beta = np.deg2rad(beta_deg)
    h_eff = h_cm * np.cos(beta)

    # ---------- 圆柱段 ----------
    x_cyl = np.linspace(-L1_BIG / 2.0, L1_BIG / 2.0, n_cyl)
    dx_cyl = x_cyl[1] - x_cyl[0]
    hx_cyl = h_eff - x_cyl * np.sin(alpha)
    S_cyl = circular_segment_area(hx_cyl, R_BIG)
    V_cyl = np.sum(S_cyl) * dx_cyl

    # ---------- 左球冠段 ----------
    # 球冠局部坐标 x_local 对应整体轴向位置 x = -L1/2 - (h0 - x_local)
    x_cap = np.linspace(0, h0_CAP, n_cap)
    dx_cap = x_cap[1] - x_cap[0]
    x_left_axis = -L1_BIG / 2.0 - (h0_CAP - x_cap)
    hx_left = h_eff - x_left_axis * np.sin(alpha)
    rho_left = cap_section_radius(x_cap, side="left")
    S_left = circular_segment_area(hx_left, rho_left)
    V_left = np.sum(S_left) * dx_cap

    # ---------- 右球冠段 ----------
    x_right_axis = L1_BIG / 2.0 + x_cap
    hx_right = h_eff - x_right_axis * np.sin(alpha)
    rho_right = cap_section_radius(x_cap, side="right")
    S_right = circular_segment_area(hx_right, rho_right)
    V_right = np.sum(S_right) * dx_cap

    return (V_cyl + V_left + V_right) / 1000.0


def inverse_parameters(data_out, alpha0=2.0, beta0=4.0):
    h_mm = data_out[:, 0]
    V_exp = data_out[:, 1]
    dV_exp = np.diff(V_exp)  # 出油后容积变化（负数）

    def objective(p):
        alpha_deg, beta_deg = p
        h_cm = h_mm / 10.0
        V_model = np.array([big_tank_volume(h, alpha_deg, beta_deg)
                            for h in h_cm])
        dV_model = np.diff(V_model)
        return np.sum((dV_model - dV_exp) ** 2)

    bounds = [(0.0, 10.0), (0.0, 10.0)]
    res = minimize(objective, x0=[alpha0, beta0], method='SLSQP',
                   bounds=bounds,
                   options={'ftol': 1e-8, 'maxiter': 500})
    return res.x


def generate_capacity_table_big(alpha_deg, beta_deg, h_max_mm=3000,
                               step_mm=100):
    h_mm = np.arange(0, h_max_mm + step_mm, step_mm)
    V = np.array([big_tank_volume(h / 10.0, alpha_deg, beta_deg) for h in h_mm])
    df = pd.DataFrame({"油位高度(mm)": h_mm,
                      "理论储油量(L)": np.round(V, 3)})
    df.attrs["tag"] = f"变位α={alpha_deg:.3f}°, β={beta_deg:.3f}°"
    return df


# 四、数据读取

def read_xls_safe(filepath, sheet_name=0):
    """安全读取 xls，失败返回 None。"""
    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name, engine='xlrd',
                           header=None)
        return df
    except Exception as e:
        print(f"[警告] 读取文件失败: {filepath}\n       错误信息: {e}")
        return None


def load_problem1_data(filepath="问题A附件1：实验采集数据表.xls"):
    if not os.path.exists(filepath):
        print(f"[提示] 未找到数据文件 {filepath}，将使用示例数据进行演示。")
        return None
    sheets = ['no_in', 'no_out', 'shift_in', 'shift_out']
    data = {}
    for i, key in enumerate(sheets):
        df = read_xls_safe(filepath, sheet_name=i)
        if df is not None:
            df = df.dropna(how='all').reset_index(drop=True)
            data[key] = df.values
        else:
            data[key] = None
    return data


def load_problem2_data(filepath="问题A附件2：实际采集数据表.xls"):
    if not os.path.exists(filepath):
        print(f"[提示] 未找到数据文件 {filepath}，将使用示例数据进行演示。")
        return None
    df = read_xls_safe(filepath, sheet_name=0)
    if df is None:
        return None
    df = df.dropna(how='all').reset_index(drop=True)
    return df.values



# 绘图函数

def plot_problem1(df_no, df_shift, data_dict):
    # ---------- 图1 ----------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_no["油位高度(mm)"], df_no["理论储油量(L)"],
            'b-', label='无变位罐容表', linewidth=2)
    ax.plot(df_shift["油位高度(mm)"], df_shift["理论储油量(L)"],
            'r--', label=f'变位罐容表 (α=4.1°)', linewidth=2)
    if data_dict is not None:
        if data_dict.get('no_in') is not None:
            d = data_dict['no_in']
            ax.scatter(d[:, 1], 262 + d[:, 0], c='k', s=20,
                       label='无变位进油实验', alpha=0.6)
        if data_dict.get('shift_in') is not None:
            d = data_dict['shift_in']
            ax.scatter(d[:, 1], 262 + d[:, 0], c='g', s=20,
                       label='变位进油实验', alpha=0.6)
    ax.set_xlabel('油位高度 (mm)')
    ax.set_ylabel('储油量 (L)')
    ax.set_title('图1：问题一 罐容表对比曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "problem1_fig1_capacity_compare.png"),
                dpi=150)
    plt.close(fig)

    # ---------- 图2：误差分析 ----------
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    a_fit = df_no.attrs.get("a", 40.5)
    b_fit = df_no.attrs.get("b", 25.0)
    L_fit = df_no.attrs.get("L", 240.0)
    alpha = 4.1

    for ax, key, title in zip(axes, ['shift_in', 'shift_out'],
                              ['变位进油误差', '变位出油误差']):
        if data_dict is not None and data_dict.get(key) is not None:
            d = data_dict[key]
            h_cm = d[:, 1] / 10.0
            V_model = np.array([small_tank_volume_shift(h, a_fit, b_fit,
                                                        L_fit, alpha)
                                for h in h_cm])
            if key == 'shift_in':
                V_exp = 262.0 + d[:, 0]
            else:
                V_exp = V_model.max() - d[:, 0]
            mask = V_exp > 0
            err = np.abs((V_model[mask] - V_exp[mask]) / V_exp[mask]) * 100
            ax.plot(d[mask, 1], err, 'b.-', markersize=5)
            ax.set_title(f'图2-{title}')
            ax.set_xlabel('油位高度 (mm)')
            ax.set_ylabel('相对误差 (%)')
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(f'图2-{title}（无数据）')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "problem1_fig2_error.png"), dpi=150)
    plt.close(fig)

    # ---------- 图3：差异对比 ----------
    fig, ax = plt.subplots(figsize=(10, 6))
    h = df_no["油位高度(mm)"].values
    diff = df_no["理论储油量(L)"].values - df_shift["理论储油量(L)"].values
    ax.plot(h, diff, 'm-', linewidth=2)
    ax.set_xlabel('油位高度 (mm)')
    ax.set_ylabel('储油量差异 (L)')
    ax.set_title('图3：无变位 - 变位 罐容表差异对比')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "problem1_fig3_diff.png"), dpi=150)
    plt.close(fig)

    print("[问题一] 图表已保存至 output/ 目录。")


def plot_problem2(df_cap, alpha, beta, data_arr):
    """问题二绘图：图1 罐容表+散点；图2 理论vs实测(y=x)；图3 误差(2x1)；图4 三维曲面。"""
    h_table = df_cap["油位高度(mm)"].values
    V_table = df_cap["理论储油量(L)"].values

    if data_arr is not None:
        h_obs = data_arr[:, 4]
        V_obs = data_arr[:, 5]
        V_model_obs = np.array([big_tank_volume(h / 10.0, alpha, beta)
                                for h in h_obs])

    # ---------- 图1 ----------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(h_table, V_table, 'b-', linewidth=2, label='变位后罐容表')
    if data_arr is not None:
        ax.scatter(h_obs, V_obs, c='r', s=15, alpha=0.5, label='实测数据')
    ax.set_xlabel('油位高度 (mm)')
    ax.set_ylabel('储油量 (L)')
    ax.set_title(f'图1：问题二 罐容表 (α={alpha:.3f}°, β={beta:.3f}°)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "problem2_fig1_capacity.png"), dpi=150)
    plt.close(fig)

    # ---------- 图2 ----------
    if data_arr is not None:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(V_obs, V_model_obs, c='b', s=15, alpha=0.5)
        vmin = min(V_obs.min(), V_model_obs.min())
        vmax = max(V_obs.max(), V_model_obs.max())
        ax.plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=2, label='y=x')
        ax.set_xlabel('实测值 (L)')
        ax.set_ylabel('理论值 (L)')
        ax.set_title('图2：理论值 vs 实测值')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "problem2_fig2_scatter.png"),
                    dpi=150)
        plt.close(fig)

    # ---------- 图3 ----------
    if data_arr is not None:
        rel_err = np.abs((V_model_obs - V_obs) / V_obs) * 100
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        axes[0].plot(h_obs, rel_err, 'b.-', markersize=4)
        axes[0].set_xlabel('油位高度 (mm)')
        axes[0].set_ylabel('相对误差 (%)')
        axes[0].set_title('图3：误差分布曲线')
        axes[0].grid(True, alpha=0.3)
        axes[1].hist(rel_err, bins=30, color='steelblue', edgecolor='black')
        axes[1].set_xlabel('相对误差 (%)')
        axes[1].set_ylabel('频数')
        axes[1].set_title('误差直方图')
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "problem2_fig3_error.png"), dpi=150)
        plt.close(fig)

    # ---------- 图4：三维曲面（β 固定）----------
    fig = plt.figure(figsize=(10, 7))
    ax3d = fig.add_subplot(111, projection='3d')
    h_grid = np.linspace(50, 2900, 30)
    a_grid = np.linspace(0, 8, 20)
    H, A = np.meshgrid(h_grid, a_grid)
    V_grid = np.zeros_like(H)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            V_grid[i, j] = big_tank_volume(H[i, j] / 10.0, A[i, j], beta)
    surf = ax3d.plot_surface(H, A, V_grid, cmap='viridis', alpha=0.8)
    ax3d.set_xlabel('油位高度 (mm)')
    ax3d.set_ylabel('α (度)')
    ax3d.set_zlabel('储油量 (L)')
    ax3d.set_title('图4：V(h, α) 三维曲面（β 固定）')
    fig.colorbar(surf, shrink=0.5, aspect=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "problem2_fig4_surface.png"), dpi=150)
    plt.close(fig)

    print("[问题二] 图表已保存至 output/ 目录。")


# 灵敏度分析

def sensitivity_analysis(alpha_inv, beta_inv, h_fixed_mm=1500):
    h_cm = h_fixed_mm / 10.0
    alpha_range = np.linspace(0, 8, 50)
    beta_range = np.linspace(0, 8, 50)

    V_alpha = np.array([big_tank_volume(h_cm, a, beta_inv)
                        for a in alpha_range])
    V_beta = np.array([big_tank_volume(h_cm, alpha_inv, b)
                       for b in beta_range])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(alpha_range, V_alpha, 'b-', linewidth=2)
    axes[0].axvline(alpha_inv, color='r', linestyle='--',
                   label=f'反演值 α={alpha_inv:.3f}°')
    axes[0].set_xlabel('α (度)')
    axes[0].set_ylabel('储油量 (L)')
    axes[0].set_title(f'α 灵敏度曲线（β={beta_inv:.3f}°, h={h_fixed_mm}mm）')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(beta_range, V_beta, 'r-', linewidth=2)
    axes[1].axvline(beta_inv, color='b', linestyle='--',
                   label=f'反演值 β={beta_inv:.3f}°')
    axes[1].set_xlabel('β (度)')
    axes[1].set_ylabel('储油量 (L)')
    axes[1].set_title(f'β 灵敏度曲线（α={alpha_inv:.3f}°, h={h_fixed_mm}mm）')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "sensitivity_analysis.png"), dpi=150)
    plt.close(fig)

    V_base = big_tank_volume(h_cm, alpha_inv, beta_inv)
    V_a_p = big_tank_volume(h_cm, alpha_inv + 1, beta_inv)
    V_a_m = big_tank_volume(h_cm, alpha_inv - 1, beta_inv)
    V_b_p = big_tank_volume(h_cm, alpha_inv, beta_inv + 1)
    V_b_m = big_tank_volume(h_cm, alpha_inv, beta_inv - 1)

    print("\n========== 灵敏度分析 ==========")
    print(f"基准: α={alpha_inv:.4f}°, β={beta_inv:.4f}°, "
          f"h={h_fixed_mm}mm → V={V_base:.3f} L")
    print(f"α+1° → V={V_a_p:.3f} L (Δ={V_a_p-V_base:+.3f} L)")
    print(f"α-1° → V={V_a_m:.3f} L (Δ={V_a_m-V_base:+.3f} L)")
    print(f"β+1° → V={V_b_p:.3f} L (Δ={V_b_p-V_base:+.3f} L)")
    print(f"β-1° → V={V_b_m:.3f} L (Δ={V_b_m-V_base:+.3f} L)")
    da = abs(V_a_p - V_a_m) / 2
    db = abs(V_b_p - V_b_m) / 2
    if da > db:
        print(f"结论: α 的影响更大（|ΔV_α|={da:.3f} > |ΔV_β|={db:.3f}）")
    else:
        print(f"结论: β 的影响更大（|ΔV_β|={db:.3f} > |ΔV_α|={da:.3f}）")
    print("[灵敏度分析] 图表已保存至 output/sensitivity_analysis.png")


# 主程序

def main_problem1():
    print("=" * 70)
    print("问题一：小椭圆型储油罐（两端平头椭圆柱体）")
    print("=" * 70)

    data_dict = load_problem1_data()
    if data_dict is None:
        print("[演示模式] 使用示例数据。")
        data_dict = generate_synthetic_data_problem1()

    print("\n--- 参数拟合 ---")
    a, b, L = fit_ellipse_params(data_dict['no_in'])
    print(f"拟合参数: a={a:.4f} cm, b={b:.4f} cm, L={L:.4f} cm")

    print("\n--- 生成罐容表 ---")
    df_no = generate_capacity_table_small(a, b, L, alpha_deg=None,
                                          h_max_mm=int(2 * b * 10),
                                          step_mm=10)
    df_no.attrs["a"], df_no.attrs["b"], df_no.attrs["L"] = a, b, L
    df_shift = generate_capacity_table_small(a, b, L, alpha_deg=4.1,
                                             h_max_mm=int(2 * b * 10),
                                             step_mm=10)

    df_no.to_csv(os.path.join(OUTPUT_DIR, "problem1_capacity_no_shift.csv"),
                 index=False, encoding='utf-8-sig')
    df_shift.to_csv(os.path.join(OUTPUT_DIR, "problem1_capacity_shift.csv"),
                    index=False, encoding='utf-8-sig')
    print("罐容表已保存: problem1_capacity_no_shift.csv, "
          "problem1_capacity_shift.csv")
    print("\n无变位罐容表（前 10 行）:")
    print(df_no.head(10).to_string(index=False))
    print("\n变位罐容表（前 10 行）:")
    print(df_shift.head(10).to_string(index=False))

    print("\n--- 模型验证 ---")
    alpha = 4.1
    a_fit, b_fit, L_fit = a, b, L
    for key in ['shift_in', 'shift_out']:
        d = data_dict[key]
        h_cm = d[:, 1] / 10.0
        V_model = np.array([small_tank_volume_shift(h, a_fit, b_fit, L_fit,
                                                    alpha) for h in h_cm])
        if key == 'shift_in':
            V_exp = 262.0 + d[:, 0]
        else:
            V_exp = V_model.max() - d[:, 0]
        mask = V_exp > 0
        rel_err = np.abs((V_model[mask] - V_exp[mask]) / V_exp[mask]) * 100
        print(f"{key}: 平均相对误差={rel_err.mean():.4f}%, "
              f"最大相对误差={rel_err.max():.4f}%")

    plot_problem1(df_no, df_shift, data_dict)


def main_problem2():
    print("\n" + "=" * 70)
    print("问题二：实际储油罐（圆柱 + 两端球冠形封头）")
    print("=" * 70)

    data_arr = load_problem2_data()
    if data_arr is None:
        print("[演示模式] 使用示例数据。")
        data_arr = generate_synthetic_data_problem2()

    # 筛选出油数据
    out_mask = data_arr[:, 3] > 0
    data_out = data_arr[out_mask]
    if data_out.size == 0:
        # 无明确出油列，直接使用所有数据
        data_out = np.column_stack([
            data_arr[:, 4],  # 油高
            data_arr[:, 5],  # 容积
            np.zeros(data_arr.shape[0])
        ])
    else:
        data_out = np.column_stack([
            data_out[:, 4], data_out[:, 5], data_out[:, 3]
        ])

    print("\n--- 参数反演 ---")
    alpha_inv, beta_inv = inverse_parameters(data_out, alpha0=2.0, beta0=4.0)
    print(f"反演结果: α={alpha_inv:.4f}°, β={beta_inv:.4f}°")

    print("\n--- 生成罐容表 ---")
    df_cap = generate_capacity_table_big(alpha_inv, beta_inv, h_max_mm=3000,
                                         step_mm=100)
    df_cap.to_csv(os.path.join(OUTPUT_DIR, "problem2_capacity_table.csv"),
                  index=False, encoding='utf-8-sig')
    print("罐容表已保存: problem2_capacity_table.csv")
    print("\n变位罐容表（前 10 行）:")
    print(df_cap.head(10).to_string(index=False))

    print("\n--- 模型验证 ---")
    h_obs = data_arr[:, 4]
    V_obs = data_arr[:, 5]
    V_model_obs = np.array([big_tank_volume(h / 10.0, alpha_inv, beta_inv)
                            for h in h_obs])
    mask = V_obs > 0
    rel_err = np.abs((V_model_obs[mask] - V_obs[mask]) / V_obs[mask]) * 100
    print(f"平均相对误差={rel_err.mean():.4f}%")
    print(f"最大相对误差={rel_err.max():.4f}%")
    print(f"误差<1% 占比: {np.mean(rel_err < 1) * 100:.2f}%")
    print(f"误差<2% 占比: {np.mean(rel_err < 2) * 100:.2f}%")
    print(f"误差<3% 占比: {np.mean(rel_err < 3) * 100:.2f}%")

    plot_problem2(df_cap, alpha_inv, beta_inv, data_arr)

    return alpha_inv, beta_inv


def main():
    print("储油罐变位识别与罐容表标定")
    print(f"输出目录: {os.path.abspath(OUTPUT_DIR)}\n")

    main_problem1()

    alpha_inv, beta_inv = main_problem2()

    print("\n" + "=" * 70)
    print("附加任务：灵敏度分析")
    print("=" * 70)
    sensitivity_analysis(alpha_inv, beta_inv, h_fixed_mm=1500)

    print("\n全部任务完成！所有结果已保存至 output/ 目录。")


if __name__ == "__main__":
    main()

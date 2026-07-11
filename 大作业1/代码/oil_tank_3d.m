% 储油罐三维结构可视化
% 结构：中间圆柱体 + 两端球冠形封头
% 坐标系：原点O为圆柱段中心轴线中点，x轴为轴线方向，z轴竖直向上
% 总长度 L = L1 + 2*h0，球冠与圆柱在 x = ±L1/2 处相切过渡

clear; close all;

%% ========== 参数设置 ==========
R  = 1.5;       % 圆柱体半径
L1 = 8;         % 圆柱段长度
h0 = 1.2;       % 单个球冠高度

% 根据相切条件计算球冠曲率半径 r
% 相切条件：R^2 = 2*r*h0 - h0^2  →  r = (R^2 + h0^2) / (2*h0)
r = (R^2 + h0^2) / (2 * h0);

fprintf('========== 储油罐参数 ==========\n');
fprintf('  圆柱半径       R  = %.2f\n', R);
fprintf('  圆柱段长度     L1 = %.2f\n', L1);
fprintf('  球冠高度       h0 = %.2f\n', h0);
fprintf('  球冠曲率半径   r  = %.2f\n', r);
fprintf('  储油罐总长度   L  = %.2f\n', L1 + 2*h0);
fprintf('================================\n');

%% ========== 网格分辨率 ==========
n_theta = 60;    % 周向网格数
n_x     = 40;    % 圆柱轴向网格数
n_phi   = 30;    % 球冠极角方向网格数

%% ========== 1. 圆柱体部分 ==========
% 范围：x ∈ [-L1/2, L1/2]，截面圆半径 R
theta = linspace(0, 2*pi, n_theta);
x_cyl = linspace(-L1/2, L1/2, n_x);

[Theta, X_cyl] = meshgrid(theta, x_cyl);   % X_cyl: 40×60, Theta: 40×60
Y_cyl = R * cos(Theta);
Z_cyl = R * sin(Theta);

%% ========== 2. 右端球冠 ==========
% 范围：x ∈ [L1/2, L1/2 + h0]
% 球心坐标：x_c = L1/2 + h0 - r  （球冠高度 h0 = 球心到基面距离 + r）
x_c_right = L1/2 + h0 - r;

% 极角范围：φ ∈ [0, φ_max]，其中 φ_max = acos((r - h0)/r)
phi_max = acos((r - h0) / r);
phi_r   = linspace(0, phi_max, n_phi);
theta_cap = linspace(0, 2*pi, n_theta);

[Phi_r, Theta_r] = meshgrid(phi_r, theta_cap);

X_right = x_c_right + r * cos(Phi_r);
Y_right = r * sin(Phi_r) .* cos(Theta_r);
Z_right = r * sin(Phi_r) .* sin(Theta_r);

%% ========== 3. 左端球冠 ==========
% 范围：x ∈ [-L1/2 - h0, -L1/2]
% 球心坐标：x_c = -L1/2 - h0 + r
x_c_left = -L1/2 - h0 + r;

% 极角范围：φ ∈ [π - φ_max, π]
phi_l = linspace(pi - phi_max, pi, n_phi);

[Phi_l, Theta_l] = meshgrid(phi_l, theta_cap);

X_left = x_c_left + r * cos(Phi_l);
Y_left = r * sin(Phi_l) .* cos(Theta_l);
Z_left = r * sin(Phi_l) .* sin(Theta_l);

%% ========== 4. 三维绘图 ==========
figure('Position', [100, 100, 1000, 650]);
hold on;

% --- 圆柱体 ---
surf(X_cyl, Y_cyl, Z_cyl, ...
    'FaceColor', [0.25, 0.55, 0.85], ...
    'FaceAlpha', 0.75, ...
    'EdgeColor', 'none', ...
    'DisplayName', '圆柱段');

% --- 右球冠 ---
surf(X_right, Y_right, Z_right, ...
    'FaceColor', [0.85, 0.35, 0.35], ...
    'FaceAlpha', 0.75, ...
    'EdgeColor', 'none', ...
    'DisplayName', '右球冠');

% --- 左球冠 ---
surf(X_left, Y_left, Z_left, ...
    'FaceColor', [0.85, 0.35, 0.35], ...
    'FaceAlpha', 0.75, ...
    'EdgeColor', 'none', ...
    'DisplayName', '左球冠');

%% ========== 5. 标注与美化 ==========
axis equal;
xlabel('X (轴线方向)', 'FontSize', 12);
ylabel('Y', 'FontSize', 12);
zlabel('Z (竖直方向)', 'FontSize', 12);
title('储油罐三维结构图', 'FontSize', 15, 'FontWeight', 'bold');
subtitle(sprintf('R=%.1f  L_1=%.1f  h_0=%.1f  r=%.2f  L=%.1f', ...
    R, L1, h0, r, L1+2*h0), 'FontSize', 10);

% 绘制坐标轴参考线
plot3([-L1/2-h0-R, L1/2+h0+R], [0, 0], [0, 0], 'k--', 'LineWidth', 0.5);
plot3([0, 0], [-R*1.5, R*1.5], [0, 0], 'k--', 'LineWidth', 0.5);
plot3([0, 0], [0, 0], [-R*1.5, R*1.5], 'k--', 'LineWidth', 0.5);

% 标记关键位置
plot3(L1/2, 0, 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
text(L1/2 + 0.3, 0.4, 0, 'x = L_1/2', 'FontSize', 9);

plot3(-L1/2, 0, 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
text(-L1/2 - 0.3, 0.4, 0, 'x = -L_1/2', 'FontSize', 9, ...
    'HorizontalAlignment', 'right');

plot3(0, 0, 0, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
text(0, 0.6, 0.2, 'O (原点)', 'FontSize', 10, 'FontWeight', 'bold');

% 标注球冠球心
plot3(x_c_right, 0, 0, 'r+', 'MarkerSize', 10, 'LineWidth', 1.5);
text(x_c_right, -0.5, 0.3, '右球心', 'FontSize', 8, 'Color', 'r');
plot3(x_c_left, 0, 0, 'r+', 'MarkerSize', 10, 'LineWidth', 1.5);
text(x_c_left, -0.5, 0.3, '左球心', 'FontSize', 8, 'Color', 'r');

% 光照与视角
grid on;
view(45, 25);
lighting gouraud;
camlight('right');
camlight('left');

% 图例
legend('Location', 'northeastoutside', 'FontSize', 10);

hold off;

fprintf('绘图完成！可在图形窗口中旋转/缩放查看三维结构。\n');
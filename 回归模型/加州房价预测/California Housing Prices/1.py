import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

plt.style.use('seaborn-v0_8-whitegrid')

font_candidates = ['SimHei', 'Microsoft YaHei', 'STXihei', 'SimSun', 'STKaiti', 'STSong', 'Arial Unicode MS']
available_fonts = [f.name for f in fm.fontManager.ttflist]
selected_font = None
for candidate in font_candidates:
    if candidate in available_fonts:
        selected_font = candidate
        break

if selected_font is None:
    keyword_fonts = []
    for f in fm.findSystemFonts():
        lower_name = f.lower()
        if any(k in lower_name for k in ['simhei', 'yahei', 'simsun', 'kaiti', 'xihei', 'stsong']):
            try:
                fm.fontManager.addfont(f)
                keyword_fonts.append(fm.FontProperties(fname=f).get_name())
            except Exception:
                pass
    if keyword_fonts:
        selected_font = keyword_fonts[0]
    else:
        selected_font = 'sans-serif'

plt.rcParams['font.family'] = selected_font
plt.rcParams['font.sans-serif'] = [selected_font] + [c for c in font_candidates if c != selected_font]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.facecolor'] = 'white'
save_path = os.getcwd()

data = pd.read_csv('housing.csv')
print("=== 数据集基本信息 ===")
print(data.info())
print("\n=== 数据集统计描述 ===")
print(data.describe())

print("\n=== 缺失值处理 ===")
print(f"处理前total_bedrooms缺失值数量: {data['total_bedrooms'].isnull().sum()}")
df_processed = data.copy()
df_processed['total_bedrooms'] = df_processed['total_bedrooms'].fillna(df_processed['total_bedrooms'].median())
print(f"处理后total_bedrooms缺失值数量: {df_processed['total_bedrooms'].isnull().sum()}")

print("\n=== 极端值检测 ===")
numeric_columns = data.columns.drop(['longitude', 'latitude','ocean_proximity','median_house_value'])
outlier_info = {}
for col in numeric_columns:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(Q1 - 3*IQR, 0)
    upper_bound = Q3 + 3*IQR
    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
    outlier_info[col] = {
        'count': len(outliers),
        'percentage': round((len(outliers)/len(data))*100,2),
        'lower': lower_bound,
        'upper': upper_bound
    }
for col, info in outlier_info.items():
    print(f"{col}: {info['count']} 个极端值 ({info['percentage']}%)")

print("\n=== 极端值缩尾处理 ===")
features_to_winsorize = numeric_columns
lower_limit = 0.025
upper_limit = 0.975
for feature in features_to_winsorize:
    low = df_processed[feature].quantile(lower_limit)
    high = df_processed[feature].quantile(upper_limit)
    df_processed[feature] = df_processed[feature].clip(low, high)
    print(f"{feature}: 缩尾处理完成（上下尾各截断2.5%）")

numeric_features = ['total_rooms', 'total_bedrooms', 'population',
                    'households', 'median_income', 'median_house_value']
plt.figure(figsize=(18, 15))
for idx, feature in enumerate(numeric_features, 1):
    plt.subplot(2, 3, idx)
    data_boxplot = data.copy()
    if feature in ['total_rooms', 'total_bedrooms', 'population', 'households']:
        data_boxplot[feature] = np.log1p(data_boxplot[feature])
        ylab = f'log({feature})'
    else:
        ylab = feature
    sns.boxplot(y=data_boxplot[feature], linewidth=2, width=0.6)
    plt.title(f'{feature} 分布', fontsize=14, pad=10)
    plt.ylabel(ylab, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    sns.despine(top=True, right=True)
plt.suptitle('数值特征极端值分布箱线图', fontsize=18, y=1.02)
plt.tight_layout()
fig1 = os.path.join(save_path, "加州房价_极端值箱线图.png")
plt.savefig(fig1, dpi=300, bbox_inches='tight')
plt.show()

comparison_features = ['median_income', 'total_rooms', 'total_bedrooms', 'population', 'households']
log_cols = ['total_rooms', 'population', 'total_bedrooms', 'households']

def draw_hist(df, feat, pos, title_str, color):
    xlab = feat
    val = df[feat]
    if feat in log_cols:
        val = np.log1p(df[feat])
        xlab = f'log({feat})'
    plt.subplot(2, 5, pos)
    plt.hist(val, bins=50, alpha=0.7, color=color)
    plt.xlabel(xlab, fontsize=10)
    plt.ylabel('频数', fontsize=10)
    plt.title(f'{feat}\n{title_str}', fontsize=12)
    plt.tick_params(axis='both', labelsize=8)

plt.figure(figsize=(20, 12))
for i, feat in enumerate(comparison_features, 1):
    draw_hist(data, feat, i, '处理前', 'darkorange')
    draw_hist(df_processed, feat, i + 5, '处理后', 'darkgreen')

plt.suptitle('数值特征：极端值处理前后分布对比', fontsize=18, y=1.02)
plt.tight_layout()
fig2 = os.path.join(save_path, "加州房价_分布对比直方图.png")
plt.savefig(fig2, dpi=300, bbox_inches='tight')
plt.show()

df_encoded = df_processed.copy()
df_encoded['ocean_proximity'] = df_encoded['ocean_proximity'].astype('category').cat.codes
corr_matrix = df_encoded.corr()
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5,
            annot_kws={"size": 10}, vmin=-1, vmax=1)
plt.title('加州房价特征相关性热力图', fontsize=18, y=1.02)
plt.tight_layout()
fig3 = os.path.join(save_path, "加州房价_相关性热力图.png")
plt.savefig(fig3, dpi=300, bbox_inches='tight')
plt.show()

print("\n=== 全部图表保存完成 ===")
print(f"图1：{fig1}")
print(f"图2：{fig2}")
print(f"图3：{fig3}")
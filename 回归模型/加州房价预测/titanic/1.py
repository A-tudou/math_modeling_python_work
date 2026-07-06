import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os
import warnings

# ===================== 【第一行优先执行字体配置，解决所有中文方框乱码】 =====================
# 屏蔽字体缺失警告
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

# 图片保存至当前脚本文件夹
save_path = os.getcwd()

# 1. 加载数据集
train_data = pd.read_csv('titanic_train.csv')
test_data = pd.read_csv('titanic_test.csv')

print("=== 训练集基本信息 ===")
print(train_data.info())
print("\n=== 训练集统计描述 ===")
print(train_data.describe())

# 2. 缺失值处理
print("\n=== 缺失值统计 ===")
print(train_data.isnull().sum())

train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
train_data['Cabin'] = train_data['Cabin'].fillna('Unknown')
train_data['Fare'] = train_data['Fare'].replace(0, train_data['Fare'].median())

print("\n=== 处理后缺失值统计 ===")
print(train_data.isnull().sum())

# 3. 特征工程
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
train_data['Title'] = train_data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
title_mapping = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Other', 'Rev': 'Other', 'Col': 'Other', 'Major': 'Other',
    'Mlle': 'Miss', 'Ms': 'Mrs', 'Lady': 'Other', 'Sir': 'Other',
    'Mme': 'Mrs', 'Capt': 'Other', 'Countess': 'Other', 'Jonkheer': 'Other',
    'Don': 'Other'
}
train_data['Title'] = train_data['Title'].map(title_mapping)

# ====================== 第一张图：生存情况核心特征分析图 ======================
fig, axes = plt.subplots(2, 2, figsize=(20, 15))
fig.suptitle('泰坦尼克号乘客生存情况核心特征分析', fontsize=20, y=1.02)

# 整体生存分布
sns.countplot(x='Survived', data=train_data, ax=axes[0,0])
axes[0,0].set_title('整体乘客生存分布', fontsize=14)
axes[0,0].set_xlabel('生存状态 (0=死亡, 1=生存)', fontsize=12)
axes[0,0].set_ylabel('乘客数量', fontsize=12)
for p in axes[0,0].patches:
    height = p.get_height()
    axes[0,0].text(p.get_x() + p.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontsize=12)

# 性别与生存率
sns.barplot(x='Sex', y='Survived', data=train_data, ax=axes[0,1])
axes[0,1].set_title('不同性别的生存率', fontsize=14)
axes[0,1].set_xlabel('性别', fontsize=12)
axes[0,1].set_ylabel('生存率', fontsize=12)
axes[0,1].set_ylim(0, 1)
for p in axes[0,1].patches:
    height = p.get_height()
    axes[0,1].text(p.get_x() + p.get_width()/2., height, f'{height:.1%}', ha='center', va='bottom', fontsize=12)

# 舱位与生存率
sns.barplot(x='Pclass', y='Survived', data=train_data, ax=axes[1,0])
axes[1,0].set_title('不同舱位的生存率', fontsize=14)
axes[1,0].set_xlabel('舱位等级 (1=高等, 2=中等, 3=低等)', fontsize=12)
axes[1,0].set_ylabel('生存率', fontsize=12)
axes[1,0].set_ylim(0, 1)
for p in axes[1,0].patches:
    height = p.get_height()
    axes[1,0].text(p.get_x() + p.get_width()/2., height, f'{height:.1%}', ha='center', va='bottom', fontsize=12)

# 家庭人数与生存率
sns.barplot(x='FamilySize', y='Survived', data=train_data, ax=axes[1,1])
axes[1,1].set_title('不同家庭人数的生存率', fontsize=14)
axes[1,1].set_xlabel('家庭人数', fontsize=12)
axes[1,1].set_ylabel('生存率', fontsize=12)
axes[1,1].set_ylim(0, 1)
for p in axes[1,1].patches:
    height = p.get_height()
    axes[1,1].text(p.get_x() + p.get_width()/2., height, f'{height:.1%}', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
img1 = os.path.join(save_path, "泰坦尼克号_生存情况核心特征分析.png")
plt.savefig(img1, dpi=300, bbox_inches='tight')
plt.show()

# ====================== 第二张图：数值特征分布与生存关系图 ======================
fig, axes = plt.subplots(2, 2, figsize=(20, 15))
fig.suptitle('泰坦尼克号数值特征分布与生存关系', fontsize=20, y=1.02)

# 年龄分布
sns.histplot(data=train_data, x='Age', hue='Survived', kde=True, ax=axes[0,0], bins=30)
axes[0,0].set_title('乘客年龄分布与生存情况', fontsize=14)
axes[0,0].set_xlabel('年龄', fontsize=12)
axes[0,0].set_ylabel('乘客数量', fontsize=12)

# 票价分布
sns.histplot(data=train_data, x=np.log1p(train_data['Fare']), hue='Survived', kde=True, ax=axes[0,1], bins=30)
axes[0,1].set_title('乘客票价分布（对数转换）', fontsize=14)
axes[0,1].set_xlabel('log(票价+1)', fontsize=12)
axes[0,1].set_ylabel('乘客数量', fontsize=12)

# 年龄箱线
sns.boxplot(x='Survived', y='Age', data=train_data, ax=axes[1,0])
axes[1,0].set_title('不同生存状态年龄分布', fontsize=14)
axes[1,0].set_xlabel('生存状态 (0=死亡, 1=生存)', fontsize=12)
axes[1,0].set_ylabel('年龄', fontsize=12)

# 票价箱线
sns.boxplot(x='Survived', y='Fare', data=train_data, ax=axes[1,1])
axes[1,1].set_title('不同生存状态票价分布', fontsize=14)
axes[1,1].set_xlabel('生存状态 (0=死亡, 1=生存)', fontsize=12)
axes[1,1].set_ylabel('票价', fontsize=12)
axes[1,1].set_yscale('log')

plt.tight_layout()
img2 = os.path.join(save_path, "泰坦尼克号_数值特征分布与生存关系.png")
plt.savefig(img2, dpi=300, bbox_inches='tight')
plt.show()

# ====================== 第三张图：特征相关性热力图 ======================
df_encoded = train_data.copy()
df_encoded['Sex'] = df_encoded['Sex'].map({'male': 0, 'female': 1})
df_encoded['Embarked'] = df_encoded['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
df_encoded['Title'] = df_encoded['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Other': 4})
corr_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'Title', 'Embarked']
corr_matrix = df_encoded[corr_cols].corr()

plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, annot_kws={"size": 10}, vmin=-1, vmax=1)
plt.title('泰坦尼克号特征相关性热力图', fontsize=18, y=1.02)
plt.tight_layout()
img3 = os.path.join(save_path, "泰坦尼克号_特征相关性热力图.png")
plt.savefig(img3, dpi=300, bbox_inches='tight')
plt.show()

print("\n=== 分析完成 ===")
print("图片保存路径：")
print(f"1. {img1}")
print(f"2. {img2}")
print(f"3. {img3}")
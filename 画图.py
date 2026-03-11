import numpy as np
import matplotlib.pyplot as plt

# 强制切换后端
import matplotlib
matplotlib.use('Agg')

# 1. 字体兼容性处理
try:
    plt.rcParams["font.family"] = "Times New Roman"
except:
    plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.unicode_minus"] = False

# --- 数据准备 ---
labels = [
    "Privacy Security", 
    "Model Accuracy", 
    "Low Comm. Cost", 
    "Low Comp. Cost", 
    "Drop-out Robustness"
]
num_vars = len(labels)

data = {
    "Differential Privacy (DP)": [2, 1, 5, 5, 5],
    "Secure Aggregation (SecAgg)": [4, 5, 2, 4, 1],
    "Full Homomorphic Encryption (FHE)": [5, 5, 1, 1, 5],
    "Proposed (Selective FHE)": [5, 5, 4, 4, 5]
}

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# --- 绘图 ---
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

# 采用高对比度学术配色，纯实线区分
colors = {
    "Differential Privacy (DP)": "#7F8C8D",      # 灰色
    "Secure Aggregation (SecAgg)": "#27AE60",   # 翠绿色
    "Full Homomorphic Encryption (FHE)": "#003399", # 深蓝色
    "Proposed (Selective FHE)": "#D35400"        # 醒目的深橙红
}

for name, values in data.items():
    plot_values = values + values[:1]
    is_proposed = "Proposed" in name
    
    # 纯实线绘制，无填充，彻底解决覆盖问题
    ax.plot(angles, plot_values, 
            color=colors[name], 
            linestyle='solid', 
            linewidth=4 if is_proposed else 2.5, 
            label=name, 
            marker='o' if is_proposed else None, 
            markersize=8 if is_proposed else 0,
            zorder=10 if is_proposed else 1)

# --- 细节优化 ---
ax.set_theta_offset(np.pi / 2) # 0点在正上方
ax.set_theta_direction(-1)      # 顺时针

# 设置基础标签
ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12, fontweight='bold')

# --- 核心改进：精准控制每个标签的位置，彻底拉开距离 ---
for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
    if np.isclose(angle, 0): # 顶部：Privacy Security
        label.set_horizontalalignment('center')
        label.set_y(0)   # 大幅向上移动，解决“靠下”和“进圆”问题
    elif 0 < angle < np.pi: # 右侧
        label.set_horizontalalignment('left')
        label.set_x(0.15)  # 向右大幅拉开
    elif np.isclose(angle, np.pi): # 底部
        label.set_horizontalalignment('center')
        label.set_y(-0.15) # 向下大幅拉开
    else: # 左侧
        label.set_horizontalalignment('right')
        label.set_x(-0.15) # 向左大幅拉开

# 增强网格清晰度
ax.set_ylim(0, 5) 
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(["1", "2", "3", "4", "5"], color="#444444", fontsize=11)
ax.grid(True, linestyle=':', alpha=0.8, color="#666666") # 网格颜色加深，更清晰
ax.spines['polar'].set_visible(False) # 隐藏外圈

# 图例与标题
plt.legend(loc='upper left', bbox_to_anchor=(1.15, 1.0), frameon=False, fontsize=11, labelspacing=1.3)
plt.title("Performance Evaluation of Federated Privacy Techniques", size=17, pad=60, fontweight='bold')

# 保存
plt.savefig("radar_chart_top_fixed.pdf", format='pdf', bbox_inches='tight')
plt.savefig("radar_chart_top_fixed.png", format='png', bbox_inches='tight', dpi=300)

print("完成！Privacy Security 已向上大幅移动，所有标签均在圆外。")
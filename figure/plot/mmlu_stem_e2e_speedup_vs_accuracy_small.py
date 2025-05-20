import matplotlib.pyplot as plt
import numpy as np

# 模型与设置
# models = ["physics", "chemistry", "biology", "computer science", "math", "engineering"]
models = ["computer science", "math", "engineering"]

configs = ['Baseline', 'CPU', 'GPU']


speedup = np.array([
    # [1.00, 1.16, 2.07],
    # [1.00, 1.19, 2.04],
    # [1.00, 1.21, 2.17],
    [1.00, 1.17, 1.70],
    [1.00, 1.22, 1.66],
    [1.00, 1.19, 1.68]
])





accuracy = np.array([
    # [38.57, 38.55, 38.55],
    # [50.0, 46.67, 46.67],
    # [60.42, 58.33, 58.33],
    [57.14, 54.76, 54.76],
    [28.70, 29.57, 29.57],
    [50.0, 50.0, 50.0]
])


bar_width = 0.28
x = np.arange(len(models))


colors = ['#c7dafa','#73a3f3','#4484f3']


fig, ax1 = plt.subplots(figsize=(10, 4.5))


for i in range(len(configs)):
    ax1.bar(x + i * bar_width, speedup[:, i], width=bar_width,
            label=configs[i], color=colors[i])
    # 添加顶部数值
    for j in range(len(models)):
        ax1.text(x[j] + i * bar_width, speedup[j, i] + 0.0,
                 f'{speedup[j, i]:.2f}x', ha='center', va='bottom',
                 fontsize=12, fontweight='bold')

# 左轴：Speedup
ax1.set_ylabel('E2E Speedup Over Baseline', color='blue',fontsize=20)
ax1.set_ylim(0, 2.5)
ax1.set_xticks(x + bar_width)
ax1.set_xticklabels(models, rotation=0, ha='center', fontsize=20)
ax1.legend(loc='upper left', fontsize=13, ncol=3)
ax1.set_title('E2E Speedup vs Accuracy', fontsize=20)



# 右轴：Accuracy
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy (%)', color='red', fontsize=20)
ax2.set_ylim(0, 100)

# 添加随机猜测精度线 (25%)
ax2.axhline(y=25, color='red', linestyle='--', linewidth=1)
ax2.text(len(models) - 0.15, 26.5, 'Random Guess (25%)', color='red', fontsize=12, ha='right')


# 星号表示 Accuracy
for i in range(len(configs)):
    ax2.plot(x + i * bar_width, accuracy[:, i], color='red', marker='*',
             linestyle='None', markersize=14)

# 图例项
ax2.plot([], [], color='red', marker='*', linestyle='None', label='Accuracy')
ax2.legend(loc='upper right', bbox_to_anchor=(0.86, 1.0), fontsize=13)



# 蓝色箭头 “Better”
# ax1.annotate('Better', xy=(-0.5, 1.4), xytext=(-0.5, 1.0),
#              arrowprops=dict(arrowstyle='->', color='blue', lw=2),
#              fontsize=12, color='blue', rotation=90)

# 保存为 PDF
plt.tight_layout()
plt.savefig("mmlu_stem_e2e_speedup_vs_accuracy_small2.pdf", format="pdf", bbox_inches="tight")

# 显示图像
plt.show()



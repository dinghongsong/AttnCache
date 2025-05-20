import matplotlib.pyplot as plt
import numpy as np

# 模型与设置
models = ["history", "philosophy", "law", "business", "health", "other"]
configs = ['Baseline', 'CPU', 'GPU']
# [ "other]

speedup = np.array([
    [1.00, 2.11, 2.83],
    [1.00, 1.00, 1.00],
    [1.00, 1.00, 1.008],
    [1.00, 1.00, 1.00],
    [1.00, 1.00, 1.00],
    [1.00, 1.00, 1.00]
])


accuracy = np.array([
    [73.27, 72.28, 72.28],
    [64.57, 65.02, 65.02],
    [42.27, 42.27, 42.27],
    [78.72, 78.72, 78.72],
    [65.75, 65.75, 65.75],
    [62.2, 62.2, 62.2]
])


bar_width = 0.25
x = np.arange(len(models))


colors = ['#c7dafa','#73a3f3','#4484f3']


fig, ax1 = plt.subplots(figsize=(10, 5))


for i in range(len(configs)):
    ax1.bar(x + i * bar_width, speedup[:, i], width=bar_width,
            label=configs[i], color=colors[i])
    # 添加顶部数值
    for j in range(len(models)):
        ax1.text(x[j] + i * bar_width, speedup[j, i] + 0.03,
                 f'{speedup[j, i]:.2f}x', ha='center', va='bottom',
                 fontsize=8, fontweight='bold')

# 左轴：Speedup
ax1.set_ylabel('Attn Speedup Over Baseline', color='blue',fontsize=20)
ax1.set_ylim(0, 4)
ax1.set_xticks(x + bar_width)
ax1.set_xticklabels(models, rotation=20, ha='right', fontsize=20)
ax1.legend(loc='upper left', fontsize=11, ncol=3)
ax1.set_title('Attn Speedup vs Accuracy', fontsize=20)



# 右轴：Accuracy
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy (%)', color='red', fontsize=20)
ax2.set_ylim(0, 100)

# 添加随机猜测精度线 (25%)
ax2.axhline(y=25, color='red', linestyle='--', linewidth=1)
ax2.text(len(models) - 0.1, 26.5, 'Random Guess (25%)', color='red', fontsize=10, ha='right')


# 星号表示 Accuracy
for i in range(len(configs)):
    ax2.plot(x + i * bar_width, accuracy[:, i], color='red', marker='*',
             linestyle='None', markersize=12)

# 图例项
ax2.plot([], [], color='red', marker='*', linestyle='None', label='Accuracy')
ax2.legend(loc='upper right', fontsize=11)



# 蓝色箭头 “Better”
ax1.annotate('Better', xy=(-0.5, 1.4), xytext=(-0.5, 1.0),
             arrowprops=dict(arrowstyle='->', color='blue', lw=2),
             fontsize=12, color='blue', rotation=90)

# 保存为 PDF
plt.tight_layout()
plt.savefig("mmlu_humanities_other_attn_speedup_vs_accuracy.pdf", format="pdf", bbox_inches="tight")

# 显示图像
plt.show()



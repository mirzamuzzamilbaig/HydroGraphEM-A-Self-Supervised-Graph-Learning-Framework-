import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

tasks = ['Flood', 'Nitrate (NO3)', 'Sediment', 'Drought']
ssl_scores = [0.6708, 0.6850, 0.2322, -0.0189]
random_scores = [-0.0062, -0.0271, -0.0004, -0.0007]

x = np.arange(len(tasks))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

bars_ssl = ax.bar(x - width/2, ssl_scores, width, label='Self-Supervised (SSL)',
                   color='#2166AC', edgecolor='#1a4d7a', linewidth=0.8, zorder=3)
bars_rand = ax.bar(x + width/2, random_scores, width, label='Random Init',
                    color='#D6604D', edgecolor='#a84a3b', linewidth=0.8,
                    hatch='///', alpha=0.85, zorder=3)

ax.axhline(y=0, color='#333333', linestyle='--', linewidth=1.0, zorder=2, alpha=0.7)

ax.set_ylabel(r'$R^2$ Score (Temporal Holdout)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
ax.set_ylim(-0.15, 0.85)
ax.grid(axis='y', alpha=0.3, linestyle='-', zorder=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar, val in zip(bars_ssl, ssl_scores):
    ypos = max(val, 0) + 0.02
    ax.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:.4f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2166AC')

for bar, val in zip(bars_rand, random_scores):
    ypos = min(val, 0) - 0.02
    ax.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:.4f}',
            ha='center', va='top', fontsize=9, color='#D6604D')

plt.tight_layout()
plt.savefig('d:/all in one/Desktop/new_PhD_project/prism/performance_summary.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("SUCCESS: performance_summary.png saved")

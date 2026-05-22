import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Data
results = [
    {'technique_id': 'T1573.002', 'probability': 0.6316718459129333},
    {'technique_id': 'T1041', 'probability': 0.2429480254650116},
    {'technique_id': 'T1567.002', 'probability': 0.08759909868240356},
    {'technique_id': 'T1132.001', 'probability': 0.013249977491796017},
    {'technique_id': 'T1573.001', 'probability': 0.011688493192195892}
]

input_seq = ['T1071.001', 'T1102', 'T1132.001']

techniques = [r['technique_id'] for r in results]
probs = [r['probability'] for r in results]

# IEEE Standard Font Configuration
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

# ==================== 1. Pie Chart ====================
fig1, ax1 = plt.subplots(figsize=(8, 6))

colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(results)))
explode = [0.1, 0.05, 0, 0, 0]

wedges, texts, autotexts = ax1.pie(
    probs, 
    labels=techniques,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    explode=explode,
    textprops={'fontsize': 9}
)

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(9)

ax1.set_title('Fig. 1. Probability Distribution of Next Technique Predictions', 
              fontsize=11, pad=15)

plt.tight_layout()
plt.savefig('fig1_pie_chart.png', dpi=600, bbox_inches='tight')
plt.show()

# ==================== 2. Waterfall Chart ====================
fig2, ax2 = plt.subplots(figsize=(10, 5))

cumulative = np.cumsum([0] + probs[:-1])
colors_waterfall = plt.cm.viridis(np.linspace(0.3, 0.9, len(results)))

bars = ax2.bar(techniques, probs, bottom=cumulative, color=colors_waterfall, 
               edgecolor='black', linewidth=1.5, alpha=0.85)

# Connection lines
for i in range(len(results)-1):
    ax2.plot([i, i+1], [cumulative[i+1], cumulative[i+1]], 
            'k--', linewidth=1.2, alpha=0.6)

# Probability labels
for i, (bar, prob) in enumerate(zip(bars, probs)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., cumulative[i] + height/2.,
            f'{prob:.1%}', ha='center', va='center', 
            fontsize=9, color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3))

ax2.set_ylabel('Cumulative Probability', fontsize=10)
ax2.set_xlabel('Technique ID', fontsize=10)
ax2.set_title('Fig. 2. Waterfall Chart of Cumulative Prediction Probabilities', 
             fontsize=11, pad=10)
ax2.set_ylim(0, 1.05)
ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax2.axhline(y=1.0, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='Total = 100%')
ax2.legend(fontsize=9, loc='lower right')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('fig2_waterfall_chart.png', dpi=600, bbox_inches='tight')
plt.show()

# ==================== 3. Heatmap ====================
fig3, (ax3_1, ax3_2) = plt.subplots(2, 1, figsize=(12, 6), 
                                     gridspec_kw={'height_ratios': [1, 3]})

# Input sequence display
input_data = np.ones((1, len(input_seq)))
sns.heatmap(input_data, annot=np.array(input_seq).reshape(1, -1), 
            fmt='', cmap='Blues', cbar=False, ax=ax3_1,
            linewidths=2, linecolor='black', annot_kws={'size': 10})
ax3_1.set_title('(a) Input Sequence (Attack Chain)', fontsize=10, pad=8)
ax3_1.set_yticks([0.5])
ax3_1.set_yticklabels(['Input'], fontsize=9)
ax3_1.set_xticks([])

# Predictions display
data = np.array(probs).reshape(1, -1)
sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn_r', 
            xticklabels=techniques, yticklabels=['Probability'],
            cbar_kws={'label': 'Probability Score'}, ax=ax3_2,
            linewidths=2, linecolor='black', vmin=0, vmax=1,
            annot_kws={'size': 10})
ax3_2.set_title('(b) Predicted Next Techniques (Top-5)', fontsize=10, pad=8)
ax3_2.set_xticklabels(techniques, rotation=45, ha='right', fontsize=9)
ax3_2.set_yticklabels(['Probability'], rotation=0, fontsize=9)

fig3.suptitle('Fig. 3. Heatmap Visualization of Input Sequence and Predictions', 
              fontsize=11, y=0.98)

plt.tight_layout()
plt.savefig('fig3_heatmap_chart.png', dpi=600, bbox_inches='tight')
plt.show()

# ==================== 4. Radar Chart ====================
fig4, ax4 = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

angles = np.linspace(0, 2 * np.pi, len(results), endpoint=False).tolist()
probs_plot = probs + [probs[0]]
angles += angles[:1]

# Plot
ax4.plot(angles, probs_plot, 'o-', linewidth=2.5, color='#1f77b4', 
         markersize=8, label='Probability', markerfacecolor='#ff7f0e')
ax4.fill(angles, probs_plot, alpha=0.25, color='#1f77b4')

# Reference lines
for prob_level in [0.2, 0.4, 0.6, 0.8]:
    ax4.plot(angles, [prob_level] * len(angles), 'k--', alpha=0.2, linewidth=0.5)

# Labels
ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(techniques, size=10)
ax4.set_ylim(0, 1)
ax4.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax4.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
ax4.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)

# Values
for angle, prob, tech in zip(angles[:-1], probs, techniques):
    ax4.text(angle, prob + 0.08, f'{prob:.3f}', 
            ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6))

ax4.set_title('Fig. 4. Radar Chart of Prediction Confidence Distribution', 
              size=11, pad=25)
ax4.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=9)

plt.tight_layout()
plt.savefig('fig4_radar_chart.png', dpi=600, bbox_inches='tight')
plt.show()

# ==================== 5. Comprehensive Dashboard ====================
fig5 = plt.figure(figsize=(16, 10))
gs = GridSpec(3, 3, figure=fig5, hspace=0.4, wspace=0.4)

# 1. Main horizontal bar chart
ax5_1 = fig5.add_subplot(gs[0:2, 0:2])
colors_main = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(results)))
bars = ax5_1.barh(techniques, probs, color=colors_main, edgecolor='black', 
                  linewidth=1.5, alpha=0.85)

ax5_1.set_xlabel('Probability', fontsize=10)
ax5_1.set_title('(a) Top-K Next Technique Predictions', fontsize=11, pad=8)
ax5_1.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
ax5_1.set_xlim(0, 1)

for i, (bar, prob) in enumerate(zip(bars, probs)):
    ax5_1.text(prob + 0.02, bar.get_y() + bar.get_height()/2, 
              f'{prob:.3f}', va='center', fontsize=9)

ax5_1.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, 
              alpha=0.5, label='50% Threshold')
ax5_1.legend(fontsize=9, loc='lower right')

# 2. Small pie chart
ax5_2 = fig5.add_subplot(gs[0, 2])
wedges, texts, autotexts = ax5_2.pie(probs, labels=techniques, autopct='%1.0f%%', 
                                       startangle=90, colors=colors_main,
                                       textprops={'fontsize': 7})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(7)
ax5_2.set_title('(b) Distribution', fontsize=10)

# 3. Cumulative chart
ax5_3 = fig5.add_subplot(gs[1, 2])
cumulative_probs = np.cumsum(probs)
ax5_3.plot(range(len(results)), cumulative_probs, marker='o', 
          linewidth=2.5, markersize=8, color='#2ca02c', markerfacecolor='#ff7f0e')
ax5_3.fill_between(range(len(results)), cumulative_probs, alpha=0.3, color='#2ca02c')
ax5_3.set_xticks(range(len(results)))
ax5_3.set_xticklabels(techniques, rotation=45, ha='right', fontsize=8)
ax5_3.set_ylabel('Cumulative Probability', fontsize=9)
ax5_3.set_title('(c) Cumulative Distribution', fontsize=10)
ax5_3.grid(alpha=0.3, linestyle='--', linewidth=0.8)
ax5_3.axhline(y=0.95, color='r', linestyle='--', linewidth=1.5, 
              alpha=0.6, label='95% Coverage')
ax5_3.set_ylim(0, 1.05)
ax5_3.legend(fontsize=8)

for i, (x, y) in enumerate(zip(range(len(results)), cumulative_probs)):
    ax5_3.text(x, y + 0.02, f'{y:.2f}', ha='center', fontsize=8)

# 4. Statistical table
ax5_4 = fig5.add_subplot(gs[2, :])
ax5_4.axis('tight')
ax5_4.axis('off')

entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)
top3_coverage = sum(probs[:3])

table_data = [
    ['Input Sequence', ' $\\rightarrow$ '.join(input_seq)],
    ['Top Prediction', f"{techniques[0]} (Confidence: {probs[0]:.1%})"],
    ['Prediction Strength', 'Very High' if probs[0] > 0.6 else 'High' if probs[0] > 0.4 else 'Medium'],
    ['Shannon Entropy', f"{entropy:.4f} bits"],
    ['Top-3 Coverage', f"{top3_coverage:.1%}"],
    ['Confidence Ratio', f"{probs[0]/probs[1]:.2f}:1 (1st vs 2nd)"]
]

table = ax5_4.table(cellText=table_data, cellLoc='left',
                   colWidths=[0.22, 0.78], loc='center',
                   colLabels=['Metric', 'Value'])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.2)

# Coloring
for i in range(len(table_data) + 1):
    if i == 0:
        table[(i, 0)].set_facecolor('#4CAF50')
        table[(i, 1)].set_facecolor('#4CAF50')
        table[(i, 0)].set_text_props(weight='bold', color='white', size=10)
        table[(i, 1)].set_text_props(weight='bold', color='white', size=10)
    else:
        table[(i, 0)].set_facecolor('#e8f5e9')
        table[(i, 0)].set_text_props(weight='bold')

ax5_4.set_title('(d) Statistical Summary', fontsize=10, pad=10, loc='left')

fig5.suptitle('Fig. 5. Comprehensive Dashboard for Attack Technique Prediction Analysis', 
            fontsize=12, y=0.98)

plt.savefig('fig5_dashboard_complete.png', dpi=600, bbox_inches='tight')
plt.show()

# ==================== 6. Comparison Charts ====================
fig6, axes = plt.subplots(2, 2, figsize=(14, 10))

# 6.1 Vertical bar chart
ax6_1 = axes[0, 0]
bars1 = ax6_1.bar(techniques, probs, color=colors_main, edgecolor='black', 
                  linewidth=1.5, alpha=0.85)
ax6_1.set_ylabel('Probability', fontsize=10)
ax6_1.set_title('(a) Vertical Bar Chart', fontsize=11)
ax6_1.grid(axis='y', alpha=0.3, linewidth=0.8)
ax6_1.set_xticklabels(techniques, rotation=45, ha='right')
for bar, prob in zip(bars1, probs):
    height = bar.get_height()
    ax6_1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
              f'{prob:.3f}', ha='center', va='bottom', fontsize=9)

# 6.2 Line chart
ax6_2 = axes[0, 1]
ax6_2.plot(techniques, probs, marker='o', linewidth=2.5, markersize=10, 
          color='#d62728', markerfacecolor='#ff7f0e')
ax6_2.fill_between(range(len(techniques)), probs, alpha=0.3, color='#d62728')
ax6_2.set_ylabel('Probability', fontsize=10)
ax6_2.set_title('(b) Line Chart with Area Fill', fontsize=11)
ax6_2.grid(alpha=0.3, linewidth=0.8)
ax6_2.set_xticklabels(techniques, rotation=45, ha='right')
for i, (tech, prob) in enumerate(zip(techniques, probs)):
    ax6_2.text(i, prob + 0.03, f'{prob:.3f}', ha='center', fontsize=9)

# 6.3 Lollipop chart
ax6_3 = axes[1, 0]
ax6_3.hlines(y=techniques, xmin=0, xmax=probs, color='gray', alpha=0.4, linewidth=4)
ax6_3.plot(probs, techniques, "o", markersize=12, color=colors_main[0], alpha=0.8)
ax6_3.set_xlabel('Probability', fontsize=10)
ax6_3.set_title('(c) Lollipop Chart', fontsize=11)
ax6_3.grid(axis='x', alpha=0.3, linewidth=0.8)
for tech, prob in zip(techniques, probs):
    ax6_3.text(prob + 0.02, tech, f'{prob:.3f}', va='center', fontsize=9)

# 6.4 Stacked bar chart
ax6_4 = axes[1, 1]
bottom = 0
for i, (tech, prob, color) in enumerate(zip(techniques, probs, colors_main)):
    ax6_4.barh(['Prediction'], [prob], left=bottom, color=color, 
              edgecolor='black', linewidth=1.5, label=tech)
    ax6_4.text(bottom + prob/2, 0, f'{tech}\n{prob:.1%}', 
              ha='center', va='center', fontsize=8, color='white')
    bottom += prob

ax6_4.set_xlabel('Cumulative Probability', fontsize=10)
ax6_4.set_title('(d) Stacked Bar Chart', fontsize=11)
ax6_4.set_xlim(0, 1)
ax6_4.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)

fig6.suptitle('Fig. 6. Comparative Visualization of Prediction Results', 
              fontsize=12, y=0.98)

plt.tight_layout()
plt.savefig('fig6_comparison_charts.png', dpi=600, bbox_inches='tight')
plt.show()

# Summary statistics
print("=" * 60)
print("VISUALIZATION GENERATION COMPLETE")
print("=" * 60)
print(f"\nStatistical Summary:")
print(f"  - Top Prediction: {techniques[0]} = {probs[0]:.4f}")
print(f"  - Shannon Entropy: {entropy:.4f} bits")
print(f"  - Top-3 Coverage: {top3_coverage:.4f}")
print(f"  - Confidence Ratio: {probs[0]/probs[1]:.2f}:1")
print(f"\nAll figures saved at 600 DPI (IEEE standard)")
print("=" * 60)

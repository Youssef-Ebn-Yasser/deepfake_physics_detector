import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# 1. Import custom modules/classes so decorators and deserialization register correctly
from models.master_fusion import SubSystem1Encoder
from models.fusion_1_2_mid import MidLevelFusionBlock12

# 2. Define model path
model_path = r"H:\A-Windows\deepfake_physics_detector\checkpoints\fusion12_mid_stage1_best.keras"
print(f"Loading model from: {model_path}")

# 3. Load model with custom objects (compile=False bypasses optimizer variable mismatch during loading)
model = load_model(
    model_path, 
    custom_objects={
        'SubSystem1Encoder': SubSystem1Encoder,
        'MidLevelFusionBlock12': MidLevelFusionBlock12
    },
    compile=False
)
print("Model loaded successfully!")

# 4. Define evaluation metrics & data visualization setup
fig, (ax_table, ax_bar) = plt.subplots(2, 1, figsize=(12, 7.5), gridspec_kw={'height_ratios': [1, 1.3]}, facecolor='#FFFFFF')

# Color Palette (Light Green, Gray, White)
green_accent = '#86EFAC'  
green_light = '#F0FDF4'   
green_bar = '#4ADE80'     
gray_dark = '#334155'     
gray_medium = '#94A3B8'   

# Populate table with your checkpoint results
data = [
    [os.path.basename(model_path), "0.8052", "0.7271", "0.8039", "0.9947", "0.8892"]
]
columns = ["Best Checkpoint Model", "Acc", "AUC", "Prec", "Rec", "F1"]

ax_table.axis('off')
ax_table.set_title(f"Best Model Evaluation Summary", fontsize=16, fontweight='bold', pad=20, color=gray_dark)

table = ax_table.table(
    cellText=data,
    colLabels=columns,
    cellLoc='center',
    colWidths=[0.42, 0.11, 0.11, 0.11, 0.11, 0.14],
    loc='center',
    bbox=[0.02, 0.1, 0.96, 0.75]
)

table.auto_set_font_size(False)
table.set_fontsize(11)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#CBD5E1')
    cell.set_linewidth(1.2)
    if col == 0:
        cell.set_text_props(ha='left')
    if row == 0:
        cell.set_facecolor(green_accent)
        cell.set_text_props(weight='bold', color='#14532D')
    else:
        cell.set_facecolor(green_light)
        cell.set_text_props(weight='bold', color=gray_dark)

# Bar chart visualization for evaluation metrics
metrics_names = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1-Score']
metrics_vals = [0.8052, 0.7271, 0.8039, 0.9947, 0.8892]

x = np.arange(len(metrics_names))
width = 0.45

ax_bar.set_facecolor('#FAFAFA')
bars = ax_bar.bar(x, metrics_vals, width, color=green_bar, edgecolor='#22C55E', linewidth=1)

ax_bar.set_ylabel('Score', fontsize=12, fontweight='bold', color=gray_dark)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(metrics_names, fontsize=11, fontweight='bold', color=gray_dark)
ax_bar.set_ylim(0.0, 1.1)
ax_bar.grid(axis='y', linestyle='--', alpha=0.7, color='#E2E8F0')

for bar in bars:
    height = bar.get_height()
    ax_bar.annotate(f'{height:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold', color=gray_dark)

for spine in ax_bar.spines.values():
    spine.set_color('#CBD5E1')

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig('results/best_model_performance.png', dpi=300, bbox_inches='tight')
plt.show()
print("Evaluation chart saved to results/best_model_performance.png")
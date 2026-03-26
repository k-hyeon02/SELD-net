import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

LOG_FILE = 'logs/seld_6973.out'
OUT_FILE = 'models/results_paper_6973.png'

# Parse log
epochs, tr_loss, val_loss = [], [], []
f1, er, doa_gt, doa_pred, metric = [], [], [], [], []
best_epoch = None

with open(LOG_FILE) as f:
    for line in f:
        m = re.match(
            r'epoch: (\d+).*tr_loss: ([\d.]+), val_loss: ([\d.]+), '
            r'F1: ([\d.]+), ER: ([\d.]+), doa_gt: ([\d.]+), doa_pred: ([\d.]+), '
            r'metric: ([\d.]+), best_metric: [\d.]+, best_epoch: (\d+)', line)
        if m:
            epochs.append(int(m.group(1)))
            tr_loss.append(float(m.group(2)))
            val_loss.append(float(m.group(3)))
            f1.append(float(m.group(4)))
            er.append(float(m.group(5)))
            doa_gt.append(float(m.group(6)))
            doa_pred.append(float(m.group(7)))
            metric.append(float(m.group(8)))
            best_epoch = int(m.group(9))

epochs = np.array(epochs)
best_idx = np.argmin(metric)

# Plot
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

BLUE   = '#2563EB'
ORANGE = '#EA580C'
GREEN  = '#16A34A'
GRAY   = '#6B7280'
BEST_COLOR = '#DC2626'

def style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.axvline(best_epoch, color=BEST_COLOR, linestyle=':', linewidth=1.5, label=f'Best epoch ({best_epoch})')

# (1) Train / Val Loss
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(epochs, tr_loss,  color=BLUE,   linewidth=1.5, label='Train Loss')
ax1.plot(epochs, val_loss, color=ORANGE, linewidth=1.5, label='Val Loss')
style_ax(ax1, '(a) Training & Validation Loss', 'Epoch', 'Loss')
ax1.legend(fontsize=9, loc='upper right')

# (2) SELD Metric
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(epochs, metric, color=BLUE, linewidth=1.5, label='SELD Metric')
ax2.scatter([best_epoch], [metric[best_idx]], color=BEST_COLOR, zorder=5, s=60,
            label=f'Best: {metric[best_idx]:.4f}')
style_ax(ax2, '(b) SELD Metric (lower is better)', 'Epoch', 'Metric')
ax2.legend(fontsize=9, loc='upper right')

# (3) SED: F1 & ER
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(epochs, f1, color=GREEN,  linewidth=1.5, label='F1 Score')
ax3.plot(epochs, er, color=ORANGE, linewidth=1.5, label='Error Rate (ER)')
ax3.scatter([best_epoch], [f1[best_idx]], color=GREEN,  zorder=5, s=60, label=f'F1@best: {f1[best_idx]:.4f}')
ax3.scatter([best_epoch], [er[best_idx]], color=ORANGE, zorder=5, s=60, label=f'ER@best: {er[best_idx]:.4f}')
style_ax(ax3, '(c) SED Performance', 'Epoch', 'Score')
ax3.legend(fontsize=9, loc='upper right')

# (4) DOA Error
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(epochs, doa_gt,   color=BLUE,  linewidth=1.5, label='DOA Error (GT ref)')
ax4.plot(epochs, doa_pred, color=GREEN, linewidth=1.5, label='DOA Error (Pred ref)')
ax4.scatter([best_epoch], [doa_gt[best_idx]],   color=BLUE,  zorder=5, s=60, label=f'GT@best: {doa_gt[best_idx]:.4f}')
ax4.scatter([best_epoch], [doa_pred[best_idx]], color=GREEN, zorder=5, s=60, label=f'Pred@best: {doa_pred[best_idx]:.4f}')
style_ax(ax4, '(d) DOA Error (lower is better)', 'Epoch', 'Error')
ax4.legend(fontsize=9, loc='upper right')

# Summary box
summary = (
    f"Best Epoch: {best_epoch}    "
    f"SELD Metric: {metric[best_idx]:.4f}    "
    f"F1: {f1[best_idx]:.4f}    "
    f"ER: {er[best_idx]:.4f}    "
    f"DOA$_{{gt}}$: {doa_gt[best_idx]:.4f}    "
    f"DOA$_{{pred}}$: {doa_pred[best_idx]:.4f}"
)
fig.text(0.5, 0.01, summary, ha='center', fontsize=10.5,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#F1F5F9', edgecolor='#CBD5E1'))

fig.suptitle('SELDnet Training Results (ansim, ov1, split1)', fontsize=15, fontweight='bold', y=1.01)

plt.savefig(OUT_FILE, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved to {OUT_FILE}')

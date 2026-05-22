from itertools import cycle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import label_binarize
from sklearn.metrics import auc, roc_curve


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


def plot_topk_predictions(results, title="Top-K Next Technique Prediction"):
    labels = [r["technique_id"] for r in results]
    probs = [r["probability"] for r in results]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, probs)
    plt.title(title)
    plt.ylabel("Probability")
    plt.xlabel("Technique ID")
    plt.ylim(0, 1)

    for bar, prob in zip(bars, probs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{prob:.3f}",
                 ha="center", va="bottom")

    plt.tight_layout()
    plt.show()


def plot_training_loss(losses, save_path="plots/loss_curve.png"):

    """
    Display and save the training Loss chart with automatic folder creation capability
    """
    # 1. ساخت مسیر از طریق pathlib
    path = Path(save_path)
    
    # 2. ایجاد پوشه والد اگر وجود ندارد
    path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', color='b', label='Training Loss')
    
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 3. ذخیره نمودار
    plt.savefig(path)
    print("The loss plot has been successfully saved to the path {save_path}.")
    plt.close() # برای آزاد سازی حافظه
    """
    نمایش و ذخیره نمودار Loss آموزش
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', color='b', label='Training Loss')
    
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # ذخیره نمودار
    plt.savefig(save_path)
    print("The loss plot has been saved to the path {save_path}.")
    
    # نمایش نمودار (اگر در محیط GUI هستید)
    plt.show()



def plot_confusion_matrix(metrics, class_names=None, save_path="confusion_matrix.png", figsize=(10, 8)):
    """
    رسم Confusion Matrix
    
    Args:
        metrics: دیکشنری خروجی از evaluate_model
        class_names: لیست نام کلاس‌ها (اختیاری)
        save_path: مسیر ذخیره تصویر
        figsize: اندازه figure
    """
    cm = metrics["confusion_matrix"]
    
    if class_names is None:
        n_classes = cm.shape[0]
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # رسم heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label', fontsize=10, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=10, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold', pad=15)
    
    # چرخش برچسب‌ها
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    # اضافه کردن آمار کلی
    accuracy = metrics["accuracy"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]
    
    stats_text = f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}"
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"Confusion Matrix saved to: {save_path}")
    plt.show()
    plt.close()


def plot_roc_curve(metrics, class_names=None, save_path="roc_curve.png", figsize=(10, 8)):
    """
    رسم ROC Curve برای مسائل چند کلاسه
    
    Args:
        metrics: دیکشنری خروجی از evaluate_model
        class_names: لیست نام کلاس‌ها (اختیاری)
        save_path: مسیر ذخیره تصویر
        figsize: اندازه figure
    """
    y_true = metrics["y_true"]
    y_prob = metrics["y_prob"]
    
    # تعداد کلاس‌ها
    n_classes = y_prob.shape[1]
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    # تبدیل برچسب‌ها به فرمت binary
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # محاسبه ROC curve و AUC برای هر کلاس
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # محاسبه micro-average ROC curve
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # محاسبه macro-average ROC curve
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # رسم نمودار
    fig, ax = plt.subplots(figsize=figsize)
    
    # رنگ‌های مختلف برای هر کلاس
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 
                    'purple', 'brown', 'pink', 'gray', 'olive'])
    
    # رسم ROC برای هر کلاس
    for i, color in zip(range(n_classes), colors):
        ax.plot(
            fpr[i], tpr[i], color=color, lw=2,
            label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})'
        )
    
    # رسم micro-average
    ax.plot(
        fpr["micro"], tpr["micro"],
        label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
        color='deeppink', linestyle=':', linewidth=3
    )
    
    # رسم macro-average
    ax.plot(
        fpr["macro"], tpr["macro"],
        label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})',
        color='navy', linestyle=':', linewidth=3
    )
    
    # خط مرجع (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=10, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=10, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=12, fontweight='bold', pad=15)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"ROC Curve saved to: {save_path}")
    plt.show()
    plt.close()


def plot_all_metrics(metrics, class_names=None, save_dir="./plots"):
    
    """
    رسم تمام نمودارها (Confusion Matrix و ROC Curve)
    
    Args:
        metrics: دیکشنری خروجی از evaluate_model
        class_names: لیست نام کلاس‌ها (اختیاری)
        save_dir: مسیر پوشه ذخیره تصاویر
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    roc_path = os.path.join(save_dir, "roc_curve.png")
    
    plot_confusion_matrix(metrics, class_names=class_names, save_path=cm_path)
    plot_roc_curve(metrics, class_names=class_names, save_path=roc_path)
    
    print(f"\nAll plots saved to: {save_dir}")
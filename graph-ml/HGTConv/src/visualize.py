from itertools import cycle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import label_binarize
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, 
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import label_binarize
from itertools import cycle
from typing import List, Dict, Optional, Union


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



def plot_confusion_matrix(
    y_true_tactics: List[str],
    y_pred_tactics: List[str],
    class_names: Optional[List[str]] = None,
    save_path: str = "confusion_matrix.png",
    figsize: tuple = (12, 10),
    show_metrics: bool = True
):
    """
    رسم Confusion Matrix بر اساس تاکتیک‌ها
    
    Args:
        y_true_tactics: لیست تاکتیک‌های واقعی (مثل ['reconnaissance', 'initial-access', ...])
        y_pred_tactics: لیست تاکتیک‌های پیش‌بینی شده
        class_names: لیست نام کلاس‌ها برای نمایش (اختیاری، اگر None باشد از unique tactics استفاده می‌شود)
        save_path: مسیر ذخیره تصویر
        figsize: اندازه figure
        show_metrics: نمایش معیارهای کلی در پایین نمودار
    
    Example:
        >>> y_true_tactics = ['reconnaissance', 'initial-access', 'execution', ...]
        >>> y_pred_tactics = ['reconnaissance', 'persistence', 'execution', ...]
        >>> plot_confusion_matrix(y_true_tactics, y_pred_tactics, 
        ...                       save_path="plots/tactic_cm.png")
    """
    # استخراج کلاس‌های یکتا با حفظ ترتیب
    if class_names is None:
        # استخراج یکتاها از هر دو لیست
        unique_tactics = list(dict.fromkeys(y_true_tactics + y_pred_tactics))
        class_names = unique_tactics
    
    # محاسبه confusion matrix
    cm = confusion_matrix(y_true_tactics, y_pred_tactics, labels=class_names)
    
    # محاسبه معیارهای کلی
    accuracy = accuracy_score(y_true_tactics, y_pred_tactics)
    precision = precision_score(y_true_tactics, y_pred_tactics, 
                                average='weighted', zero_division=0)
    recall = recall_score(y_true_tactics, y_pred_tactics, 
                         average='weighted', zero_division=0)
    f1 = f1_score(y_true_tactics, y_pred_tactics, 
                  average='weighted', zero_division=0)
    
    # ایجاد figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # تبدیل نام‌های تاکتیک به فرمت قابل خواندن
    readable_names = [name.replace('-', ' ').title() for name in class_names]
    
    # رسم heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=readable_names,
        yticklabels=readable_names,
        cbar_kws={'label': 'Count'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        square=True
    )
    
    # تنظیمات محورها
    ax.set_xlabel('Predicted Tactic', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Tactic', fontsize=11, fontweight='bold')
    ax.set_title('Tactic-Level Confusion Matrix', 
                 fontsize=13, fontweight='bold', pad=20)
    
    # چرخش برچسب‌ها
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', 
             rotation_mode='anchor', fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    # اضافه کردن آمار کلی
    if show_metrics:
        stats_text = (
            f"Accuracy: {accuracy:.4f}  |  "
            f"Precision: {precision:.4f}  |  "
            f"Recall: {recall:.4f}  |  "
            f"F1-Score: {f1:.4f}"
        )
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight', format='png')
    print(f"✓ Confusion Matrix saved to: {save_path}")
    print(f"  Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | "
          f"Recall: {recall:.4f} | F1: {f1:.4f}")
    plt.close()


def plot_roc_curve(
    y_true_tactics: List[str],
    y_prob: np.ndarray,
    technique_to_tactic_map: Dict[str, str],
    id2technique: Dict[int, str],
    class_names: Optional[List[str]] = None,
    save_path: str = "roc_curve.png",
    figsize: tuple = (12, 10),
    max_classes_to_plot: int = 14
):
    """
    رسم ROC Curve برای تاکتیک‌ها (multi-class)
    
    Args:
        y_true_tactics: لیست تاکتیک‌های واقعی
        y_prob: آرایه احتمالات پیش‌بینی شده (shape: [n_samples, n_technique_classes])
        technique_to_tactic_map: دیکشنری نگاشت تکنیک به تاکتیک
        id2technique: دیکشنری نگاشت ایندکس به external_id تکنیک
        class_names: لیست نام تاکتیک‌ها برای نمایش (اختیاری)
        save_path: مسیر ذخیره تصویر
        figsize: اندازه figure
        max_classes_to_plot: حداکثر تعداد کلاس‌ها برای رسم جداگانه
    
    Example:
        >>> y_true_tactics = ['reconnaissance', 'initial-access', ...]
        >>> y_prob = np.array([[0.1, 0.8, ...], [0.3, 0.5, ...], ...])  # احتمالات تکنیک‌ها
        >>> technique_to_tactic_map = {'T1595': 'reconnaissance', 'T1078': 'initial-access', ...}
        >>> plot_roc_curve(y_true_tactics, y_prob, technique_to_tactic_map, 
        ...                id2technique, save_path="plots/tactic_roc.png")
    """
    # ─────────────────────────────────────────────────────────
    # گام 1: تبدیل احتمالات تکنیک‌ها به احتمالات تاکتیک‌ها
    # ─────────────────────────────────────────────────────────
    
    # استخراج کلاس‌های یکتای تاکتیک
    if class_names is None:
        unique_tactics = sorted(list(set(y_true_tactics)))
        class_names = unique_tactics
    else:
        unique_tactics = class_names
    
    n_tactics = len(unique_tactics)
    n_samples = y_prob.shape[0]
    
    # ایجاد دیکشنری نگاشت تاکتیک به ایندکس
    tactic_to_idx = {tactic: idx for idx, tactic in enumerate(unique_tactics)}
    
    # ایجاد آرایه احتمالات تاکتیک‌ها
    y_prob_tactics = np.zeros((n_samples, n_tactics))
    
    # جمع احتمالات تکنیک‌های متعلق به هر تاکتیک
    for tech_idx, tech_id in id2technique.items():
        if tech_id in technique_to_tactic_map:
            tactic = technique_to_tactic_map[tech_id]
            if tactic in tactic_to_idx:
                tactic_idx = tactic_to_idx[tactic]
                y_prob_tactics[:, tactic_idx] += y_prob[:, tech_idx]
    
    # نرمال‌سازی احتمالات (اختیاری)
    row_sums = y_prob_tactics.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # جلوگیری از تقسیم بر صفر
    y_prob_tactics = y_prob_tactics / row_sums
    
    # ─────────────────────────────────────────────────────────
    # گام 2: تبدیل برچسب‌های تاکتیک به فرمت binary
    # ─────────────────────────────────────────────────────────
    
    # نگاشت تاکتیک‌های واقعی به ایندکس
    y_true_indices = np.array([tactic_to_idx[t] for t in y_true_tactics])
    
    # تبدیل به binary format
    y_true_bin = label_binarize(y_true_indices, classes=range(n_tactics))
    
    # ─────────────────────────────────────────────────────────
    # گام 3: محاسبه ROC curve و AUC برای هر تاکتیک
    # ─────────────────────────────────────────────────────────
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_tactics):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob_tactics[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # محاسبه micro-average ROC curve
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_true_bin.ravel(), y_prob_tactics.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # محاسبه macro-average ROC curve
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_tactics)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_tactics):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_tactics
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # ─────────────────────────────────────────────────────────
    # گام 4: رسم نمودار
    # ─────────────────────────────────────────────────────────
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # رنگ‌های مختلف
    colors = cycle([
        'aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 
        'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 
        'magenta', 'yellow', 'black'
    ])
    
    # تبدیل نام‌های تاکتیک به فرمت قابل خواندن
    readable_names = [name.replace('-', ' ').title() for name in unique_tactics]
    
    # رسم ROC برای هر تاکتیک (محدود به max_classes_to_plot)
    if n_tactics <= max_classes_to_plot:
        for i, color in zip(range(n_tactics), colors):
            ax.plot(
                fpr[i], tpr[i], color=color, lw=2,
                label=f'{readable_names[i]} (AUC = {roc_auc[i]:.3f})'
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
    
    # تنظیمات محورها
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax.set_title('Tactic-Level ROC Curve (Multi-Class)', 
                 fontsize=13, fontweight='bold', pad=20)
    
    # Legend
    if n_tactics <= max_classes_to_plot:
        ax.legend(loc="lower right", fontsize=8, framealpha=0.95, ncol=1)
    else:
        ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight', format='png')
    print(f"✓ ROC Curve saved to: {save_path}")
    print(f"  Micro-average AUC: {roc_auc['micro']:.4f}")
    print(f"  Macro-average AUC: {roc_auc['macro']:.4f}")
    plt.close()


def plot_all_metrics(
    y_true_tactics: List[str],
    y_pred_tactics: List[str],
    y_prob: np.ndarray,
    technique_to_tactic_map: Dict[str, str],
    id2technique: Dict[int, str],
    class_names: Optional[List[str]] = None,
    save_dir: str = "plots"
):
    """
    رسم همه نمودارها (Confusion Matrix + ROC Curve)
    
    Args:
        y_true_tactics: لیست تاکتیک‌های واقعی
        y_pred_tactics: لیست تاکتیک‌های پیش‌بینی شده
        y_prob: آرایه احتمالات
        technique_to_tactic_map: نگاشت تکنیک به تاکتیک
        id2technique: نگاشت ایندکس به تکنیک
        class_names: نام کلاس‌ها (اختیاری)
        save_dir: مسیر پوشه ذخیره
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING EVALUATION PLOTS")
    print("="*60)
    
    # Confusion Matrix
    cm_path = os.path.join(save_dir, "tactic_confusion_matrix.png")
    plot_confusion_matrix(
        y_true_tactics, y_pred_tactics,
        class_names=class_names,
        save_path=cm_path
    )
    
    # ROC Curve
    roc_path = os.path.join(save_dir, "tactic_roc_curve.png")
    plot_roc_curve(
        y_true_tactics, y_prob,
        technique_to_tactic_map, id2technique,
        class_names=class_names,
        save_path=roc_path
    )
    
    print("="*60)
    print("ALL PLOTS GENERATED SUCCESSFULLY")
    print("="*60 + "\n")


   
from pathlib import Path

import matplotlib.pyplot as plt


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
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

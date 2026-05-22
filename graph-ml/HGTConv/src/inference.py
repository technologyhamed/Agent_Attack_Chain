import torch
from src.config import Config
#T1071.001, T1102, T1132.001
config =Config()

def predict_next_techniques(model, input_seq, technique2id, id2technique, data_builder, top_k=config.TOP_K, device=config.DEVICE):
    """
    input_seq: list[str]
    out: top-k next technique softmax
    """
    model.to(device)
    model.eval()

    graph = data_builder(input_seq).to(device)

    with torch.no_grad():
        logits = model(graph)
        probs = torch.softmax(logits, dim=-1)

        top_probs, top_ids = torch.topk(probs, k=min(top_k, probs.shape[-1]))

    results = []
    for p, idx in zip(top_probs.tolist(), top_ids.tolist()):
        results.append(
            {
                "technique_id": id2technique[idx],
                "probability": float(p),
            }
        )

    return results

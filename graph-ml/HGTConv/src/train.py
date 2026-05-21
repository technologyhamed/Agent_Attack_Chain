from pathlib import Path
import torch
from tqdm import tqdm
import torch.nn as nn
from torch_geometric.loader import DataLoader



def train_model(model, train_samples, technique2id, config, device="cpu"):
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()

    losses = []

    for epoch in range(config.EPOCHS):
        total_loss = 0.0
        pbar = tqdm(train_samples, desc=f"Epoch {epoch+1}/{config.EPOCHS}")

        for sample in pbar:
            data = sample["graph"].to(device)
            target = torch.tensor([sample["target_id"]], dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits.unsqueeze(0), target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / max(len(train_samples), 1)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

    return losses

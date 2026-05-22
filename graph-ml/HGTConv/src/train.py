import os
from pathlib import Path
import torch
from tqdm import tqdm
import torch.nn as nn
from torch_geometric.loader import DataLoader
from src.config import Config

config = Config()

def train_model(model, train_chain_seq, technique2id, config, device=config.DEVICE):
    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()

    losses = []
    best_loss = float("inf")
    
    os.makedirs(os.path.dirname(config.CHECKPOINT_PATH), exist_ok=True)

    for epoch in range(config.EPOCHS):

        total_loss = 0.0
        pbar = tqdm(train_chain_seq, desc=f"Epoch {epoch+1}/{config.EPOCHS}")

        for chain_squ in pbar:

            data = chain_squ["graph"].to(device)
            target = torch.tensor([chain_squ["target_id"]], dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits.unsqueeze(0), target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / max(len(train_chain_seq), 1)

        losses.append(avg_loss)
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "epoch": epoch + 1,
                    "technique2id": technique2id,
                },
                config.CHECKPOINT_PATH,
            )
            print(f"Best model saved to {config.CHECKPOINT_PATH} (loss={best_loss:.4f})")

    return losses

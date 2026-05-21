import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear


class EAHGT(nn.Module):
    def __init__(
        self,
        num_techniques: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.num_techniques = num_techniques
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.technique_embedding = nn.Embedding(num_techniques, hidden_dim)
        self.input_proj = Linear(hidden_dim, hidden_dim)

        metadata = (
            ["technique"],
            [("technique", "follows", "technique")]
        )

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=metadata,
                    heads=num_heads,
                )
            )

        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_techniques),
        )

    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        x_dict["technique"] = self.technique_embedding(x_dict["technique"])
        x_dict["technique"] = self.input_proj(x_dict["technique"])

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict["technique"] = F.relu(x_dict["technique"])
            x_dict["technique"] = F.dropout(x_dict["technique"], p=self.dropout, training=self.training)

        x = x_dict["technique"]
        x = self.norm(x)

        # استفاده از آخرین node به عنوان نماینده sequence
        seq_repr = x[-1]
        logits = self.classifier(seq_repr)

        return logits

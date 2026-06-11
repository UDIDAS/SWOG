"""
3-layer GINE (Graph Isomorphism Network with Edge features) for
supernode classification.

Architecture follows SEMIR paper:
  - 3 GINEConv layers, hidden_dim=128
  - Batch normalisation + ReLU between layers
  - Final linear classifier for binary (tumor / background)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, BatchNorm


class SEMIRClassifier(nn.Module):
    """3-layer GINE for supernode-level tumor classification."""

    def __init__(self, node_dim: int = 7, edge_dim: int = 4,
                 hidden_dim: int = 128, n_classes: int = 2):
        super().__init__()

        # Project edge features to match hidden dim
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        # Layer 1: node_dim → hidden_dim
        mlp1 = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1 = GINEConv(mlp1, edge_dim=hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)

        # Layer 2: hidden_dim → hidden_dim
        mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv2 = GINEConv(mlp2, edge_dim=hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)

        # Layer 3: hidden_dim → hidden_dim
        mlp3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv3 = GINEConv(mlp3, edge_dim=hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Project edge features; create dummy if none exist
        if edge_attr is not None and edge_attr.numel() > 0:
            edge_attr = self.edge_proj(edge_attr)
        else:
            # No edges → each node is isolated; create a self-loop per node
            n = x.size(0)
            edge_index = torch.stack([torch.arange(n, device=x.device),
                                      torch.arange(n, device=x.device)], dim=0)
            edge_attr = torch.zeros(n, self.edge_proj.out_features, device=x.device)

        # 3 GINE layers with residual-like structure
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.relu(x)

        return self.classifier(x)


def train_gine(train_graphs: list, val_graphs: list,
               epochs: int = 200, lr: float = 1e-3,
               patience: int = 10, device: str = "cpu"):
    """
    Train GINE on a list of PyG Data objects.

    Returns trained model and training history.
    """
    model = SEMIRClassifier().to(device)
    # Paper Appendix F: "Adam lr=10^-3 and no weight decay"
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    # Class weights for imbalanced tumor/background — capped to avoid instability
    total_pos = sum(int((g.y == 1).sum()) for g in train_graphs)
    total_neg = sum(int((g.y == 0).sum()) for g in train_graphs)
    if total_pos > 0:
        raw_ratio = total_neg / total_pos
        capped_ratio = min(raw_ratio, 50.0)  # cap at 50x to prevent gradient explosion
        weight = torch.tensor([1.0, capped_ratio], dtype=torch.float32).to(device)
        print(f"  Class weight: [1.0, {capped_ratio:.1f}] (raw ratio: {raw_ratio:.1f})")
    else:
        weight = torch.ones(2, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    history = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_dice = -1.0
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        # --- Train (iterate graphs one at a time — they can be large) ---
        model.train()
        total_loss = 0.0
        for g in train_graphs:
            g_dev = g.to(device)
            optimiser.zero_grad()
            logits = model(g_dev)
            loss = criterion(logits, g_dev.y)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
        avg_train = total_loss / len(train_graphs)
        history["train_loss"].append(avg_train)

        # --- Validate ---
        if val_graphs:
            model.eval()
            vloss_total, tp_all, fp_all, fn_all = 0.0, 0, 0, 0
            with torch.no_grad():
                for g in val_graphs:
                    g_dev = g.to(device)
                    logits = model(g_dev)
                    vloss_total += criterion(logits, g_dev.y).item()
                    preds = logits.argmax(dim=1)
                    tp_all += ((preds == 1) & (g_dev.y == 1)).sum().item()
                    fp_all += ((preds == 1) & (g_dev.y == 0)).sum().item()
                    fn_all += ((preds == 0) & (g_dev.y == 1)).sum().item()
            vloss = vloss_total / len(val_graphs)
            dice = float(2 * tp_all / (2 * tp_all + fp_all + fn_all + 1e-8))
            history["val_loss"].append(vloss)
            history["val_dice"].append(dice)

            if dice > best_dice:
                best_dice = dice
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"  Early stopping at epoch {epoch} (best dice={best_dice:.4f})")
                    break
        else:
            history["val_loss"].append(avg_train)
            history["val_dice"].append(0.0)

        if epoch % 20 == 0 or epoch == 1:
            vd = history["val_dice"][-1] if history["val_dice"] else 0
            print(f"  Epoch {epoch:3d}  train_loss={avg_train:.4f}  "
                  f"val_loss={history['val_loss'][-1]:.4f}  val_dice={vd:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model, history

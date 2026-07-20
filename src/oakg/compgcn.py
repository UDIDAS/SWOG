"""CompGCN / R-GCN relational graph-encoder baseline (Section 8.2).

Builds a small relational graph per case (case -hasPhenotype-> feature
-supportedBy-> organ), encodes it with a relational GCN, and reads out a case
embedding. Trained unsupervised (phenotype autoencoder on the train split — no
relevance leakage). Produces per-case embeddings consumable by the same
retrieval harness as WL (`oakg.neural.embedding_scores`).

Requires torch + torch_geometric. One fixed implementation (not optional).
"""
from __future__ import annotations

import numpy as np

from .data import Corpus


def _organ_index(corpus: Corpus) -> dict[str, int]:
    organs = sorted({o for s in corpus.feature_support.values() for o in s})
    return {o: i for i, o in enumerate(organs)}


def _case_graph(cid, X, M, corpus, organ_ix):
    """Return (node_global_ids, node_values, edge_index[2,E], edge_type[E])."""
    import torch
    idx = corpus.case_to_row[cid]
    nF, nO = len(corpus.features), len(organ_ix)
    # global node-id vocabulary: 0=case, 1..nF=features, nF+1..=organs
    node_ids = [0]; node_val = [0.0]
    src, dst, etype = [], [], []
    organ_local: dict[str, int] = {}
    n = 1
    for f, col in corpus.feature_index.items():
        if not M[idx, col]:
            continue
        node_ids.append(1 + col); node_val.append(float(X[idx, col]))
        fnode = n; n += 1
        src += [0, fnode]; dst += [fnode, 0]; etype += [0, 0]        # hasPhenotype (both dirs)
        for organ in corpus.feature_support[f]:
            if organ not in organ_local:
                node_ids.append(1 + nF + organ_ix[organ]); node_val.append(0.0)
                organ_local[organ] = n; n += 1
            onode = organ_local[organ]
            src += [fnode, onode]; dst += [onode, fnode]; etype += [1, 1]  # supportedBy (both dirs)
    if not src:  # isolated case node (no observed features)
        src, dst, etype = [0], [0], [0]
    ei = torch.tensor([src, dst], dtype=torch.long)
    return (torch.tensor(node_ids, dtype=torch.long),
            torch.tensor(node_val, dtype=torch.float32).unsqueeze(1), ei,
            torch.tensor(etype, dtype=torch.long))


class CompGCNEncoder:
    """Relational GCN encoder + phenotype-reconstruction training."""

    def __init__(self, corpus: Corpus, dim: int = 32, seed: int = 2027):
        import torch, torch.nn as nn
        from torch_geometric.nn import RGCNConv
        torch.manual_seed(seed)
        self.corpus = corpus
        self.organ_ix = _organ_index(corpus)
        self.vocab = 1 + len(corpus.features) + len(self.organ_ix)
        self.dim = dim
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class Net(nn.Module):
            def __init__(self, vocab, dim):
                super().__init__()
                self.emb = nn.Embedding(vocab, dim)
                self.vproj = nn.Linear(1, dim)
                self.c1 = RGCNConv(dim, dim, num_relations=2)
                self.c2 = RGCNConv(dim, dim, num_relations=2)
            def encode(self, nid, val, ei, et, batch, n_graphs):
                import torch
                from torch_geometric.utils import scatter
                x = self.emb(nid) + self.vproj(val)
                x = torch.relu(self.c1(x, ei, et))
                x = self.c2(x, ei, et)
                # readout = mean pool per graph
                return scatter(x, batch, dim=0, dim_size=n_graphs, reduce="mean")
        self.net = Net(self.vocab, dim).to(self.dev)
        self.decoder = nn.Linear(dim, len(corpus.features)).to(self.dev)

    def _batch(self, case_ids, X, M):
        import torch
        from torch_geometric.data import Data, Batch
        datas = []
        for cid in case_ids:
            nid, val, ei, et = _case_graph(cid, X, M, self.corpus, self.organ_ix)
            d = Data(x=val, edge_index=ei); d.nid = nid; d.et = et
            datas.append(d)
        b = Batch.from_data_list(datas)
        return (b.nid.to(self.dev), b.x.to(self.dev), b.edge_index.to(self.dev),
                torch.cat([d.et for d in datas]).to(self.dev), b.batch.to(self.dev), len(datas))

    def train(self, X, M, train_ids, epochs: int = 200, lr: float = 1e-2):
        import torch
        params = list(self.net.parameters()) + list(self.decoder.parameters())
        opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
        # target: observed phenotype vector (reconstruct observed entries only)
        rows = [self.corpus.case_to_row[c] for c in train_ids]
        tgt = torch.tensor(np.nan_to_num(X[rows]), dtype=torch.float32, device=self.dev)
        mask = torch.tensor(M[rows], dtype=torch.float32, device=self.dev)
        nid, val, ei, et, batch, ng = self._batch(train_ids, X, M)
        self.net.train(); self.decoder.train()
        for ep in range(epochs):
            opt.zero_grad()
            z = self.net.encode(nid, val, ei, et, batch, ng)
            pred = self.decoder(z)
            loss = (((pred - tgt) * mask) ** 2).sum() / mask.sum().clamp(min=1)
            loss.backward(); opt.step()
        return float(loss.item())

    def embeddings(self, X, M) -> dict[str, np.ndarray]:
        import torch
        self.net.eval()
        ids = self.corpus.case_order
        with torch.no_grad():
            nid, val, ei, et, batch, ng = self._batch(ids, X, M)
            z = self.net.encode(nid, val, ei, et, batch, ng).cpu().numpy()
        # L2-normalize for cosine retrieval (matches WL)
        z = z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-9, None)
        return {c: z[i] for i, c in enumerate(ids)}


def compgcn_embeddings(X, M, corpus: Corpus, train_ids, dim: int = 32,
                       epochs: int = 200, seed: int = 2027) -> dict[str, np.ndarray]:
    """Train a CompGCN encoder on ``train_ids`` graphs, return case embeddings."""
    enc = CompGCNEncoder(corpus, dim=dim, seed=seed)
    enc.train(X, M, train_ids, epochs=epochs)
    return enc.embeddings(X, M)

# Graph Neural Networks and Message Passing

Graph Neural Networks (GNNs) learn from data whose relationships are as important as the individual records. A graph is commonly written as G = (V, E), where V contains nodes and E contains edges. Nodes may represent users, products, papers, or atoms; edges represent friendships, purchases, citations, or chemical bonds.

## When a graph is the right representation

A table describes each row mostly through its columns. A graph additionally represents who or what is connected. Examples include:

- Recommendation: users connect to items through interactions.
- Fraud detection: accounts connect through transfers, devices, or shared identifiers.
- Citation analysis: papers connect through citations.
- Molecular learning: atoms are nodes and chemical bonds are edges.

Do not convert every dataset into a graph. Use one when relational structure carries predictive information and can be constructed without leaking future information.

## Message passing

A common GNN layer updates each node using information from its neighbors:

1. **Message:** Each neighbor produces a message from its current representation.
2. **Aggregate:** Messages are combined with a permutation-invariant operation such as sum, mean, or max.
3. **Update:** The node combines the aggregate with its own representation.

A generic layer is:

\[
h_v^{(k+1)} = \mathrm{UPDATE}\left(h_v^{(k)},\ \mathrm{AGGREGATE}\{\,\mathrm{MESSAGE}(h_v^{(k)},h_u^{(k)},e_{uv}) : u \in \mathcal{N}(v)\,\}\right)
\]

Here, h_v is the representation of node v, e_uv is an optional edge feature, and N(v) is the neighborhood of v. After k layers, a node can incorporate information from nodes up to k hops away.

## Common GNN families

- **GCN:** Uses normalized neighborhood aggregation. It is a useful baseline for node classification on relatively static graphs.
- **GraphSAGE:** Samples and aggregates neighborhoods, making it practical for inductive settings with unseen nodes.
- **GAT:** Learns attention weights so different neighbors can contribute unequally.
- **GIN:** Uses an expressive sum-based aggregation designed to distinguish graph structures under certain assumptions.

These architectures make different assumptions; none is best for every graph.

## PyTorch Geometric mini-example

The following illustrates the data shape and a two-layer node classifier. Install PyTorch and a compatible PyTorch Geometric build first.

~~~python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class NodeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

# x: [number_of_nodes, number_of_features]
# edge_index: [2, number_of_edges], with source and target node IDs
# train_mask selects labeled training nodes.
model = NodeClassifier(in_channels=16, hidden_channels=32, num_classes=3)
logits = model(x, edge_index)
loss = F.cross_entropy(logits[train_mask], labels[train_mask])
~~~

For an undirected graph, ensure edges are represented in both directions unless the chosen layer or preprocessing explicitly handles that. Split nodes or edges according to the real prediction task; random splits can leak information when time or connected components matter.

## Prediction levels

- **Node-level:** classify or regress one output per node.
- **Edge-level:** predict whether a link exists or estimate an edge property.
- **Graph-level:** predict a property of an entire molecule, circuit, or subgraph.

Graph-level tasks usually require a readout such as sum, mean, or attention pooling over node embeddings.

## Evaluation and common pitfalls

- Compare against non-graph baselines using the same feature and split protocol.
- Prevent label leakage through edges, timestamps, or graph construction.
- Check whether train and test nodes share neighbors; this may be valid for a transductive task but not for a new-graph deployment setting.
- Large neighborhoods can make message passing expensive.
- Deep stacks can cause **oversmoothing**, where node embeddings become too similar.
- Evaluate performance on rare nodes and graph regions, not only the overall average.

## A practical learning exercise

Use a small citation or social-network dataset. Train a logistic-regression baseline on node features, then a GCN. Compare macro-F1, per-class recall, runtime, and performance on nodes with few neighbors. Document exactly how the graph and split were built.

## References

- Kipf and Welling, *Semi-Supervised Classification with Graph Convolutional Networks* (2017): https://arxiv.org/abs/1609.02907
- Hamilton, Ying, and Leskovec, *Inductive Representation Learning on Large Graphs* (2017): https://arxiv.org/abs/1706.02216
- PyTorch Geometric documentation: https://pytorch-geometric.readthedocs.io/

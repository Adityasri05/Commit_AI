# Mixture of Experts and Sparse Routing

A Mixture of Experts (MoE) model contains multiple expert subnetworks and a gating mechanism that decides which experts should process each input or token. Sparse MoE activates only a small subset of experts for each item, allowing a model to increase parameter capacity without using every parameter for every token.

MoE is used in some large language models and other high-capacity neural architectures. It is a design pattern, not a guarantee of lower total cost: communication, routing, memory, and expert imbalance can become major bottlenecks.

## Dense versus sparse computation

In a dense feed-forward layer, every input uses the same weights. In an MoE layer:

1. A router computes scores for the available experts.
2. The scores are normalized or otherwise converted into routing weights.
3. A top-k rule selects a small number of experts.
4. Selected experts process the input.
5. Their outputs are combined using the routing weights.

A simplified expression is:

\[
y(x) = \sum_{i \in \mathrm{TopK}(g(x))} p_i(x) E_i(x)
\]

where g(x) is the router, E_i is expert i, and p_i is the selected expert's routing weight.

## Conceptual PyTorch sketch

~~~python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseMoE(nn.Module):
    def __init__(self, width, num_experts=4, top_k=2):
        super().__init__()
        self.router = nn.Linear(width, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                          nn.Linear(2 * width, width))
            for _ in range(num_experts)
        ])
        self.top_k = top_k

    def forward(self, x):
        # x: [batch, sequence, width]
        scores = self.router(x)
        top_scores, top_indices = scores.topk(self.top_k, dim=-1)
        weights = F.softmax(top_scores, dim=-1)

        output = torch.zeros_like(x)
        for slot in range(self.top_k):
            for expert_id, expert in enumerate(self.experts):
                mask = top_indices[..., slot] == expert_id
                if mask.any():
                    expert_input = x[mask]
                    expert_output = expert(expert_input)
                    output[mask] += weights[..., slot][mask].unsqueeze(-1) * expert_output
        return output
~~~

This small educational sketch illustrates routing, but it is not an optimized production implementation. It uses masks and Python loops, does not implement capacity limits or distributed all-to-all communication, and may be inefficient for large token batches.

## Load balancing

A router can collapse onto a few experts, leaving others underused. This is called expert imbalance or routing collapse.

Common techniques include:
- Auxiliary load-balancing losses.
- Router z-loss or other regularization.
- Capacity limits and token dropping or rerouting policies.
- Monitoring token counts and routing probabilities per expert.

A balanced router is not necessarily optimal if all experts are forced to receive identical traffic. The goal is useful specialization without pathological overload.

## Capacity and systems costs

Sparse activation reduces per-token expert computation, but all expert weights may still need to reside in memory. Distributed MoE can require tokens to move between devices, increasing communication latency.

Measure:
- Tokens routed to each expert.
- Dropped or overflowed tokens.
- Router entropy and expert utilization.
- Per-token FLOPs and end-to-end latency.
- Memory footprint and inter-device communication.

## Where MoE helps and where it may not

MoE can be useful when a model needs high capacity and inputs contain different patterns that experts can specialize in. It may be unsuitable for small models, low-throughput deployments, or environments where routing and communication overhead dominate.

## Mini-project

Implement a dense feed-forward layer and a small top-2 MoE layer with the same hidden width. Compare parameter count, active parameters per token, throughput, and validation quality. Plot expert utilization to detect routing collapse.

## References

- Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* (2017): https://arxiv.org/abs/1701.06538
- Fedus, Zoph, and Shazeer, *Switch Transformers* (2021): https://arxiv.org/abs/2101.03961

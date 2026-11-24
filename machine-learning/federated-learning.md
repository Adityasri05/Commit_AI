# Federated Learning and Federated Averaging

Federated Learning (FL) trains a shared model across multiple clients without requiring every client's raw training records to be centralized. Clients may represent phones, hospitals, banks, or organizations.

Instead of uploading all local examples, clients train locally and send model updates or related statistics to an aggregation server. This can reduce raw-data movement, but it does not automatically guarantee privacy or security.

## Basic federated training loop

1. The server initializes a global model.
2. A subset of available clients is selected for a round.
3. Each selected client downloads the current global parameters.
4. Each client trains locally using its own data.
5. Clients send updates or model weights to the server.
6. The server aggregates the contributions into a new global model.
7. The process repeats until a stopping condition is met.

## Federated Averaging (FedAvg)

A common baseline is weighted averaging of client parameters. If client k has n_k training examples and local parameters w_k, then:

\[
w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{\sum_j n_j} w_{k,t+1}
\]

The weights reflect each client's local sample count. This is a parameter aggregation rule, not an average of predictions. It is most straightforward when clients share the same model architecture and parameter layout.

A simplified server-side sketch:

~~~python
def fedavg(client_states, client_sizes):
    total = sum(client_sizes)
    aggregated = {}

    for key in client_states[0]:
        aggregated[key] = sum(
            state[key] * (size / total)
            for state, size in zip(client_states, client_sizes)
        )
    return aggregated
~~~

This sketch assumes compatible tensor keys and shapes and floating-point parameters. Production implementations must handle buffers, optimizer state, failures, secure transport, and model-version consistency deliberately.

## The non-IID challenge

Client data is often **non-independent and identically distributed (non-IID)**. One hospital may serve a different population from another; one phone may use a different language distribution.

Non-IID data can cause:
- Client updates to point in different directions.
- Slower or unstable convergence.
- A global model that performs poorly for smaller or unusual clients.
- Large differences in local performance despite a good global average.

Evaluate both aggregate and per-client or per-group metrics. Consider partial personalization, proximal objectives, or clustered federated learning when a single global model is unsuitable.

## Privacy and security are separate concerns

Federated learning limits raw-data centralization, but model updates can still leak information. Threats include gradient inversion, malicious clients, poisoning, and inference attacks.

Possible defenses include:
- Secure aggregation so the server sees an aggregate rather than each individual update.
- Differential privacy with a clearly tracked privacy budget.
- Client authentication, signed updates, and anomaly checks.
- Robust aggregation against suspicious or corrupted updates.
- Data minimization and explicit retention policies.

These protections have trade-offs. Differential privacy can reduce model utility; robust aggregation can reject legitimate unusual clients; secure aggregation can make individual-update inspection harder.

## Operational considerations

- **Client availability:** Devices may disconnect or be offline during a round.
- **Communication cost:** Model updates can be large; compression and fewer rounds may help.
- **Versioning:** Reject updates produced by incompatible model versions.
- **Fairness:** Report performance for small clients and important groups.
- **Consent:** Make participation, data use, and opt-out behavior explicit.
- **Evaluation:** Keep a representative evaluation protocol that does not silently expose private client data.

## Mini-project

Simulate three clients with deliberately different class distributions. Train a centralized baseline and a FedAvg model. Compare convergence, communication volume, overall accuracy, and per-client test performance. Then explain which privacy threats remain even though raw examples stay local.

## References

- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data* (2017): https://arxiv.org/abs/1602.05629
- TensorFlow Federated: https://www.tensorflow.org/federated
- Flower federated learning framework: https://flower.ai/docs/

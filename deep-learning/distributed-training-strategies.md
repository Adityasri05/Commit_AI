# Distributed Training Strategies for Deep Learning

Distributed training uses multiple devices or machines to train a model faster or to fit models and batches that exceed one device's memory. The right strategy depends on model size, dataset size, interconnect bandwidth, and the amount of communication required.

## Data parallelism

Each worker holds a full copy of the model and processes a different mini-batch. Gradients are synchronized so replicas stay aligned.

**DistributedDataParallel (DDP)** is PyTorch's common multi-GPU data-parallel approach. Each process usually owns one GPU. Gradient all-reduce combines gradients across workers.

The effective global batch size is:

    per-device batch * number of devices * accumulation steps

Changing the global batch can change optimization behavior. Learning rate, warm-up, and training duration may need to be revisited.

## Parameter and optimizer-state sharding

When a full model replica does not fit in each device's memory, sharding can distribute model state:

- **ZeRO-style optimization:** partitions optimizer states, gradients, and, at more aggressive stages, parameters.
- **Fully Sharded Data Parallel (FSDP):** shards parameters, gradients, and optimizer state across workers and gathers parameters as needed for computation.

Sharding reduces per-device memory but adds communication and implementation complexity. Checkpointing and loading sharded state require care.

## Tensor parallelism

Tensor parallelism splits individual layer operations across devices. For example, large matrix multiplications can be partitioned so multiple GPUs compute pieces of the same layer.

It is useful when a layer or model cannot fit efficiently on one device. It often requires fast device-to-device communication because devices exchange intermediate activations during each layer.

## Pipeline parallelism

Pipeline parallelism divides model layers into stages on different devices. A sequence of micro-batches flows through the stages.

It can reduce per-device parameter memory, but pipeline bubbles occur when stages are idle. Good partitioning and enough micro-batches are important.

## Hybrid strategies

Large training systems combine data, tensor, pipeline, and state sharding. The best combination depends on model architecture and hardware topology; more parallelism is not automatically faster.

## Minimal DDP pattern

~~~python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

model = MyModel().to(device)
model = DDP(model, device_ids=[local_rank])

sampler = DistributedSampler(
    train_dataset,
    shuffle=True,
)
loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler,
    pin_memory=True,
)

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(features), labels)
        loss.backward()
        optimizer.step()

dist.destroy_process_group()
~~~

Launch with a distributed launcher such as torchrun and set the process count to the number of participating GPUs. This is a structural example: define the model, dataset, optimizer, criterion, and cleanup/error handling for the actual project.

## Communication and performance

Distributed training can become communication-bound. Profile:
- time spent in forward and backward computation;
- gradient synchronization and parameter all-gathers;
- data-loader stalls;
- device utilization and memory pressure;
- network and interconnect throughput.

Use mixed precision and gradient accumulation where appropriate, but verify numerical stability and the actual throughput gain.

## Checkpoints and recovery

A reliable checkpoint may need model weights, optimizer and scheduler state, random-number generator state, epoch/step, and sharding metadata. Save atomically and test restoration. For large jobs, checkpoint frequency balances recovery cost against storage and training overhead.

## Common pitfalls

- Giving every worker the same unsharded dataset in DDP.
- Forgetting to call set_epoch on a distributed sampler.
- Assuming more GPUs always reduce training time.
- Ignoring communication overhead and data-loader bottlenecks.
- Changing global batch size without checking optimization behavior.
- Saving only model weights when optimizer state is needed to resume training.

## Practice

Train the same small image classifier on one GPU and with DDP on two GPUs. Compare examples per second, epoch time, validation quality, and peak memory. Keep the global batch size controlled so the comparison is meaningful.

## References

- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [PyTorch DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)

**Key takeaway:** Data parallelism, sharding, tensor parallelism, and pipeline parallelism solve different scaling problems. Profile computation and communication before selecting a distributed strategy.
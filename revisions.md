# Review Findings (World Model Focus)

Date: 2026-07-09

## Findings

1. High: Replay observation write shape mismatch in collection path.
- Collection writes a batched/time-expanded observation tensor into replay.
- Replay schema expects per-transition CHW observations.
- Risk: crash or malformed replay data.

2. High: Real interaction collection tracks gradients.
- Collection forward path is inference-only but runs without inference/no-grad guard.
- Recurrent state can retain unnecessary autograd graph history.
- Risk: avoidable memory growth / OOM pressure.

3. High: Imagination environment binds world model/device before Accelerate prepare.
- Imagination env is constructed before distributed wrapping/device placement.
- It caches device at construction time.
- Risk: stale model/device reference under distributed or device-moved runs.

4. Medium: Replay sequence sampler may sample from an empty second-half bucket.
- When second-half valid starts are empty, `np.random.choice` can raise.
- Risk: intermittent training crash on small/edge replay states.

5. Medium: Distributed SIGReg activation is based on local CUDA count.
- Local `torch.cuda.device_count()` can be 1 in multi-node setups.
- Risk: distributed regularizer silently disabled when world size > 1.

## Revision Plan

- [x] Add this review file.
- [x] Fix distributed/gradient issues:
  - Guard real interaction collection with inference mode.
  - Rebind imagination env world model/device after Accelerate prepare.
  - Gate distributed SIGReg by global process count.
- [ ] (Optional next) Fix replay write shape mismatch.
- [ ] (Optional next) Fix empty second-half replay sampler fallback.
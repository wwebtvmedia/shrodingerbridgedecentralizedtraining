# Code Review: Swarm Schrödinger Bridge

## 1. Executive Summary

This is a browser-based decentralized ("swarm") training system for Schrödinger
Bridge generative models, built on **TensorFlow.js** with a Node/Express +
WebSocket consolidation server. It implements a CNN-residual VAE (96×96), axial
attention, a U-Net drift network, LoRA adapters, SARSA-driven task selection, and
P2P/tunnel-based model exchange.

A full-codebase review was performed across the server, networking, storage, ML
layers, core training loop, utilities, and the Python conversion scripts. The
review found a mix of **security gaps on the untrusted edge** and **correctness
bugs that silently disabled key features** (SNR, attention scaling, LoRA
collection, classifier-free guidance, EMA buffers). All identified issues —
critical through low — have been fixed in commit `f313971`.

This document reflects the **post-fix state**: what each subsystem does, what was
wrong, what was changed, and what remains as known limitations / future work.

---

## 2. Resolved Findings

The table below summarizes the issues found and the fix applied. Severities are
the pre-fix severities.

### 2.1 Security

| Sev  | Location                                      | Issue                                                                                                     | Fix                                                                                                                           |
| ---- | --------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| CRIT | `server/index.js`                             | Shared hardcoded auth token (`swarm-prototype-token-2026`), also embedded in client                       | Require `SECRET_TOKEN`; generate an ephemeral random token (with a loud warning) when unset; client token no longer hardcoded |
| CRIT | `server/index.js`                             | `cors()` wide open (`*`) alongside token auth                                                             | CORS restricted to an `ALLOWED_ORIGINS` allowlist (defaults to localhost)                                                     |
| CRIT | `server/index.js`                             | `/api/model/submit` accepted arbitrary base64 with no size/`loss` validation → disk-fill DoS              | Validate finite `loss`, cap decoded size (`MAX_MODEL_BYTES`), prune old model files (`MAX_MODEL_FILES`)                       |
| CRIT | `convert_to_tfjs.py`, `inspect_checkpoint.py` | `torch.load(weights_only=False)` → pickle RCE on untrusted checkpoints                                    | `weights_only=True` (converter) / allowlist `defaultdict` + restricted load with trusted fallback (inspector)                 |
| CRIT | `src/utils/sanitizer.js`                      | Claimed prototype-pollution protection but `__proto__`/`constructor`/`prototype` passed the key regex     | Explicitly drop those keys; build on a null-prototype object                                                                  |
| CRIT | `src/consolidation-client.js`                 | Server-supplied `clientId`/`loss`/etc. rendered into `innerHTML` → XSS; `toFixed` on missing fields threw | Escape all untrusted fields, safe numeric formatting                                                                          |
| CRIT | `src/network/peer.js`                         | Remote weights parsed and cached with no validation / pollution guard                                     | Size-cap frames, sanitize parsed objects before use                                                                           |
| MED  | `server/index.js`                             | Token in URL query string (logged), non-constant-time compare                                             | Token via `Sec-WebSocket-Protocol`; `crypto.timingSafeEqual`                                                                  |
| MED  | `server/index.js`                             | Synchronous full-DB `writeFileSync` on every message; unbounded `clients` map                             | Debounced async atomic writes; idle reaper + per-IP connection cap                                                            |
| MED  | `src/storage/database.js`, `server/index.js`  | `JSON.parse` of untrusted input without a reviver                                                         | Prototype-pollution reviver on all untrusted parses                                                                           |
| MED  | `src/network/tunnel.js`                       | Auth token in WebSocket URL; shipped default secret                                                       | Token via subprotocol; no default secret                                                                                      |
| LOW  | `setup-raspberry-pi.sh`                       | Unpinned `curl \| sudo bash` (documented vendor pattern)                                                  | Left as-is, noted                                                                                                             |

### 2.2 ML Correctness

| Sev  | Location                  | Issue                                                                                                                  | Fix                                                                                                        |
| ---- | ------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| HIGH | `src/config.js`           | `calcSNR` read `mse.data[0]` (a method) → always `NaN`; leaked the scalar                                              | `dataSync()[0]` + `dispose()`                                                                              |
| HIGH | `src/torchjs/models.js`   | Attention scaled by `tf.sqrt(c/numHeads)` (JS number into a tensor op; wrong denominator; single-head, not multi-head) | `1/√c` via `Math.sqrt`                                                                                     |
| HIGH | `src/torchjs/models.js`   | `PercentileRescale` computed EMA but never assigned it → degenerated to plain `tanh`                                   | Write the EMA buffers via `LayerVariable.write`                                                            |
| HIGH | `src/torchjs/training.js` | Variables collected before the first forward → lazily-built LoRA adapters excluded on step 1                           | One-time warmup forward before variable collection; warn if empty                                          |
| HIGH | `src/utils/inference.js`  | `cfgScale` read but never applied; noise used `temperature` in place of the diffusion coefficient; squeeze leak        | Implemented classifier-free guidance; noise = `temperature·g(t)·√dt`; dispose squeezed tensor; clamp label |

### 2.3 Robustness / Correctness

| Sev  | Location                            | Issue                                                                                                                                          | Fix                                                                                                       |
| ---- | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| HIGH | `src/network/tunnel.js`             | `connect()` promise never settled on early close (await hung); async send treated as boolean in queue drain                                    | Settle on close; `await` sends in an async drain; bounded queue; handler cleanup on disconnect            |
| HIGH | `src/network/peer.js`               | Gossip TTL `0 \|\| default` resurrected expired messages; research handlers never dispatched; `process.uptime` in browser                      | Decrement-and-check TTL without re-defaulting; dispatch handlers; `typeof process` guard; channel cleanup |
| HIGH | `enhanced-trainer.js`, `trainer.js` | Sync set `currentEpoch` backwards (reset phase schedule); model-request timeout stalled the loop; `MODEL_SHARE` resolved by hash from any peer | `Math.max` epoch; soft-skip failed sync; verify `MODEL_SHARE` sender; validate data length                |
| HIGH | `src/core/adaptive-logic.js`        | `JSON.parse(JSON.stringify(weights))` turned `NaN`/`Infinity` into `null`                                                                      | `structuredClone`-based deep clone                                                                        |
| MED  | `src/storage/database.js`           | No `onblocked`/`onversionchange` (upgrade hang); string id into autoIncrement store; prune cursor lacked `onerror`; dead `IDBKeyRange`         | All addressed                                                                                             |
| MED  | `src/core/sarsa-optimizer.js`       | Reward could explode on sub-ms timings                                                                                                         | Clamp denominator + `tanh` squash                                                                         |
| MED  | `src/core/phase.js`                 | `getCurrentEpoch()` stubbed to 0; `trend()` divided by zero                                                                                    | Real epoch tracking via `setCurrentEpoch`; zero guard                                                     |
| MED  | `src/core/models.js`                | Model hash non-deterministic (`Date.now()` suffix) and collision-prone                                                                         | Deterministic FNV-1a over weight values                                                                   |
| MED  | `enhanced-trainer.js`               | TAE fed `gradientNorm` the trainer never produced (always 1.0)                                                                                 | Real global gradient norm surfaced in `trainStep` metrics                                                 |
| MED  | `src/utils/data-importer.js`        | Naive `split(",")` CSV parser broke on quotes/commas/newlines                                                                                  | RFC-4180-aware tokenizer                                                                                  |
| MED  | `src/utils/validator.js`            | `type: "any"` fields unbounded; non-finite numbers accepted                                                                                    | `maxLength` bound + finite-number checks                                                                  |
| LOW  | `src/utils/onnx-export.js`          | Decoder block consumed nonexistent tensor `dec_blockin_out`                                                                                    | Correct input-name chaining                                                                               |
| LOW  | `src/core/trainer.js`               | Gossip `setInterval` handle not stored → leaked after `stop()`                                                                                 | Store handle and clear it                                                                                 |
| MED  | `start-swarm.sh`                    | `git reset --hard` + `git clean -fd` wiped uncommitted/untracked data each run                                                                 | Stash by default; destructive path gated behind `FORCE_RESET=1` and protects data dirs                    |

---

## 3. Architecture Notes (current behavior)

- **Engine:** TensorFlow.js (WebGL/WebGPU), Node server on Express + `ws`.
- **Model:** CNN-residual VAE + axial (single-head spatial-split) attention +
  U-Net drift; latent `12×12×8`; `NUM_CLASSES=11` (10 + NULL for CFG).
- **Adaptation:** LoRA rank 8 / alpha 16; base layers frozen, adapters trained.
- **Sampling:** Euler–Maruyama reverse SDE with classifier-free guidance.
- **Coordination:** SARSA task selection, evolutionary crossover/mutation on
  shared parameters, gossip + tunnel/WebSocket exchange.

---

## 4. Verification

- All JS files pass `node --check`; both Python scripts compile.
- `tests/sarsa.test.js` and `tests/evolutionary.test.js` pass.
- `node test.js` (smoke check) passes.
- The hardened server boots, serves `/api/health`, returns 401 without a token,
  200 with a valid token, and 400 for an invalid `loss` submission.
- All changed files conform to the repo Prettier config.

---

## 5. Known Limitations & Future Work

These were **not** changed (they are design choices, performance items, or
prototype stubs), and remain candidates for follow-up:

- **Single-head attention.** `SpatialSplitAttention` does not split into heads
  (`numHeads` is informational only). At the current latent sizes (sequence
  length 12–96) this is fine; multi-head/Flash-Attention-style kernels would add
  complexity for negligible gain at this scale. Flash Attention proper is a
  GPU/CUDA kernel with no TF.js equivalent.
- **`convert_to_tfjs.py` output format.** Emits a custom manifest + `weights.bin`
  that nothing currently consumes, and the rank-based transpose is only correct
  for standard Conv2d/Linear (not ConvTranspose/grouped/1D). Drive the transpose
  by layer type if this path is wired up.
- **`onnx-export.js`** remains a prototype: weights are placeholders and the graph
  is not validated against a real ONNX runtime.
- **Custom layers are not `registerClass`-registered**, so `model.save()`
  round-trips rely on the index-ordered checkpoint path; an architecture reorder
  could silently misload same-sized variables.
- **Performance:** 96×96 CNNs are heavy on GPU-less devices; consider Web Workers
  to avoid UI jank and INT8/FP16 quantization for shared weights.
- **OU bridge** (`USE_OU_BRIDGE`) is off by default; its `t→1` boundary handling
  should be revisited before enabling.

---

**Reviewer:** Claude (Opus 4.8)
**Date:** June 6, 2026
**Reference commit:** `f313971`

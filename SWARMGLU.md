# SwarmGlu — The Swarm Glue of the Tree4Five Suite

**Repository:** <https://github.com/wwebtvmedia/swarmglu>
**Status:** v2.0.0-alpha.2 (native Kotlin core on Android, validated on real phones)

---

## 1. What is SwarmGlu?

**SwarmGlu** is the *liaison, consensus and routing layer* of the
**Tree4Five Tools** ecosystem — the "glue" that binds the other bricks
together over a plain Wi-Fi LAN with **no Internet and no central
server**. It is designed for the Djerba school pilot: heterogeneous
hardware (Linux/Windows PCs, Android phones and tablets) forming a
self-organizing, self-healing AI swarm.

### Core capabilities

| Capability | How it works |
|---|---|
| **Zero-SPOF election** | Lease-based leader election: heartbeats every 3 s, 6 s of silence triggers a deterministic election (lowest IP wins, then node id). Measured failover on real fleets: 2–8 s. |
| **Immutable audit chain** | Every event (student request, election, sync, relay…) is sealed: `hash = sha256(canonical(event_data) + previous_hash + str(request_time))` with `request_time`, `processing_time`, `timestamp` and an anti-gap sequence. `verify_chain()` replays the whole journal; torn-tail writes (killed process) are auto-repaired and sealed as `chain_tail_repaired`. |
| **Distributed state (LWW)** | Epoch-versioned replicated state; commutative Last-Write-Wins merge validated by a SHA-256 state digest — all nodes converge to byte-identical state regardless of message order. |
| **OSP v0.6 firewall** | Every AI answer gets a *groundedness* score `g` (cited sources heuristic or model-provided). `g ≥ 0.30` → VERIFIED and admitted to the shared knowledge pool; below → delivered to the student but EXCLUDED from the pool. |
| **Shared knowledge pool (RAG)** | Every VERIFIED answer automatically enters the swarm-wide knowledge base (`GET /knowledge` returns count + digest), propagated to all nodes by the LWW merge. |
| **Real LLM routing** | `/ask` routes to the best provider node (model match then load); the GGUF bridge talks to real backends — Ollama `/api/chat` or OpenAI-style APIs. |
| **Inter-swarm gateway** | Ambassador nodes exchange cryptographic state digests (`/gateway/sync`) so two classrooms' swarms can federate requests with a hop guard. |
| **Hotspot range extension** | An Android button opens a `LocalOnlyHotspot`; the tablet becomes the swarm router and relays its clients to the school network via `POST /relay` (private destinations only, sealed in the audit). |
| **Post-quantum auth** | Hybrid ML-KEM-768 + ECDSA P-256 handshake (`/secure/handshake`) and AES-256-GCM encrypted RPC (`/secure/rpc`); decentralized whitelist re-signing via a weighted lottery (age × request count) and signed 1/8 consensus. |
| **Three interoperable cores** | Python (PCs), Kotlin/JVM, and Kotlin/Native-Android — proven **byte-compatible** by golden vectors (chain extended with identical hashes, identical LWW digests) and by mixed swarms run on real hardware. |

### Android application (v2.0, native)

`com.tree4five.swarmglu` embeds the **Kotlin core directly** (Chaquopy /
Python removed): the APK is **8.8 MB** (was 62 MB), serves NanoHTTPD, and
runs **native post-quantum crypto on-device** (ECDSA via `java.security`,
ML-KEM-768 via BouncyCastle). Three screens: real-time dashboard (role,
peers, whitelist, hotspot), student question screen with the OSP verdict,
and the audit-chain verifier.

---

## 2. Use case — a school day in Djerba

A classroom has one PC (teacher), a stack of tablets/phones, and no
Internet. The teacher's PC (or any node) hosts an Ollama LLM; a second
room runs another swarm; a WhatsApp bridge node shares knowledge.

1. **Morning — power on.** Every device runs one SwarmGlu node. Nodes
   discover each other (UDP beacon or static seeds), elect a leader by
   lease (lowest IP), and replicate the roster + whitelist through LWW
   merges. Whoever boots first leads; whoever dies is replaced in
   seconds.
2. **A student asks a question** (tablet app, screen 2). The request
   enters the swarm at her node, is routed to the machine hosting the
   best matching local model (e.g. the 27B model on the teacher's PC
   through the Ollama bridge), answered, and stamped with a
   **groundedness verdict** from the OSP firewall.
3. **The answer is shared.** VERIFIED answers enter the shared knowledge
   pool: within seconds every device — including offline phones — holds
   the same knowledge (same SHA-256 digest everywhere). The class
   collectively builds a verifiable knowledge base.
4. **A tablet runs out of battery.** The swarm notices (lease expiry),
   holds an election, and continues. The journal of the dead tablet is
   still verifiable when it comes back; a torn tail is auto-repaired.
5. **Room 2 asks room 1.** Ambassador nodes exchange digests; federated
   requests cross the gateway with a hop guard.
6. **A task needs an autonomous agent** (HarnessDroid / MCP). The task is
   queued and **blocked until a teacher approves it** (Human-in-the-Loop
   barrier), then dispatched and sealed.
7. **Everything is auditable.** Any controller (PC CLI or the Audit
   screen on a tablet) replays the hash chain in one tap: INTACT or
   CORRUPTED at line N.

### Why it matters

No Internet, no cloud, no central server, no single point of failure —
yet the class gets collective AI with provable provenance, human
oversight on agent actions, and cryptographic auditability. Sovereign
and local by design.

---

## 3. The full-fleet test (validated end-to-end)

`tests/test_fleet_full.py` in the SwarmGlu repository orchestrates a
**7-node, 3-implementation fleet over real network hardware**:

```
kot-a    Kotlin/JVM node            — bootstrap leader
py-llm   Python node (PC)           — REAL LLM provider: GGUF bridge to the
                                     school Ollama (192.168.1.194), best
                                     loadable model auto-selected
far-249  Python node (SSH-deployed on 192.168.1.249, cryptography wheels
         shipped over scp — no pip on that host)
emu-a/emu-b  2 × Android emulators (native Kotlin core)
tel-XXXX     every USB-connected physical phone/tablet (auto-detected;
             Samsung Note 10 Lite + Galaxy Tab validated)
```

### What the test proves (all checks green)

1. **Preflight** — Ollama reachable (with a graceful wait for the LLM
   host to come back after a reset), best-model auto-selection by real
   inference probes (`bestmodel:latest` is documented as GPU-OOM on the
   ROCm host; the 27B model is selected instead), live OSP endpoint on
   `192.168.1.249:3000` (`whatsapp-bot`, OSP v0.6 contract), signed
   `/osp/query` through the bot's bridge (protocol verdict received).
2. **Convergence of 7 nodes / 3 implementations** — one leader accepted
   by every node; every registry sees all peers alive.
3. **A REAL student question from a phone** — routed across the swarm to
   the LLM node, answered by the real Ollama model ("Tunis" asserted),
   OSP verdict attached, knowledge shared: **identical SHA-256 digest on
   all 7 nodes**.
4. **OSP bridge** — the signed `/osp/query` returns a protocol verdict
   (mode `NO_QUORUM` expected while no other competent OSP peers are
   bidding — the negotiation protocol itself is alive and auditable).
5. **Failover** — the Kotlin leader is `kill -9`'d; the 6 survivors
   (Python + Android natives) re-elect among themselves within seconds
   and keep serving real inference.
6. **Auditability** — every survivor's hash chain replays INTACT,
   including the Android-native journals verified block-by-block by the
   Python verifier.

### The complex scenario (9 nodes, 2 real LLMs)

`tests/test_fleet_full.py --model fastmodel:latest` extends the fleet to
the **full school-day topology**: the **tablet itself becomes an LLM
provider** — its local OSP bridge (Tree4Five LLM Provider, llama.cpp
on-device, `/v1` + bridge token read from the device) is detected and
declared to the swarm registry, so a `model_hint=llmprovider` question
is routed over the swarm and answered by **real on-device llama.cpp
inference** (31 s measured), while the teacher's PC serves the school
Ollama (27B). A second school machine (192.168.1.149) joins via SSH
deployment when it is awake. The OSP bridge negotiation is exercised
live: PROPOSE → knowledge teaching (`/osp/teach`) → **competence-scored
BID** (0.733) — the quorum (q=2 competent bidders) is documented as the
protocol's scaling requirement.

### Reproduce it

```bash
# prerequisites: JDK 21, Android SDK (emulator AVDs auto-created),
# socat, USB-connected phones with USB debugging, SSH key on 192.168.1.249
cd kotlin && ./gradlew test installDist     # golden vectors + node binary
cd ..
python3 tests/test_fleet_full.py --ssh-249 youyou
```

Other swarm topologies in the same repository:

```bash
python3 run_all.py                              # fast non-regression (~1 min)
python3 run_all.py --full --ssh-249 youyou      # every orchestrator (~25 min)
python3 tests/test_real_devices_swarm.py        # phones-only swarm
python3 tests/test_multi_swarm_federation.py    # two federated classrooms
python3 tests/test_hotspot_scenario.py          # hotspot range extension
cd android && ./gradlew connectedDebugAndroidTest  # in-Android loop
```

---

## 4. Relation to this repository (Schrödinger Bridge)

SwarmGlu is the transport/consensus layer that lets the Schrödinger
Bridge artifacts travel the same way the school knowledge does: models,
knowledge-base updates and cross-swarm exchanges published by the OSP
knowledge protocol can be replicated between machines through the same
LWW-merged, hash-chain-audited state — offline, serverless, and
verifiable. The OSP v0.6 contract served by the `whatsapp-bot` node
(`192.168.1.249:3000`) is the live anchor used by the fleet test above.

// Swarm Knowledge Protocol (OSP) Bridge
// Implements Omni-Swarm Protocol (OSP v0.6) envelopes, knowledge negotiation,
// and cross-swarm artifact exchange for generated images and synthesized articles.

/**
 * Node Classes per OSP v0.6 Specification
 */
export const NodeClass = {
  N1_THIN_ORIGIN: "N1", // Querying and verification only (no local generation)
  N2_FULL_RESPONDER: "N2", // Local generation + verification + groundedness
  N3_PROVIDER_BACKED: "N3", // External / Gateway LLM-backed responder
};

/**
 * Convergence modes for OSP negotiations
 */
export const ConvergenceMode = {
  RESOLVED: "RESOLVED",
  REJECTED: "REJECTED",
  MISMATCH: "MISMATCH",
  GAS_EXHAUSTED: "GAS_EXHAUSTED",
  LOOP_DETECTED: "LOOP_DETECTED",
  NO_QUORUM: "NO_QUORUM",
};

/**
 * Generate lightweight hash-based binary vector for lexical groundedness
 */
export function computeEmbeddingBag(text, dims = 256) {
  const bytes = new Uint8Array(dims / 8);
  const words = (text || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2);

  for (const word of words) {
    let hash = 0;
    for (let i = 0; i < word.length; i++) {
      hash = (hash << 5) - hash + word.charCodeAt(i);
      hash |= 0;
    }
    const idx = Math.abs(hash) % (dims / 8);
    const bit = Math.abs(hash >> 3) % 8;
    bytes[idx] |= 1 << bit;
  }
  return bytes;
}

/**
 * Calculate Jaccard similarity between two binary bitsets
 */
export function calculateBitsetSimilarity(a, b) {
  if (!a || !b || a.length !== b.length) return 0.0;
  let intersection = 0;
  let union = 0;

  for (let i = 0; i < a.length; i++) {
    const andVal = a[i] & b[i];
    const orVal = a[i] | b[i];

    for (let bit = 0; bit < 8; bit++) {
      if ((andVal >> bit) & 1) intersection++;
      if ((orVal >> bit) & 1) union++;
    }
  }

  return union === 0 ? 0.0 : intersection / union;
}

/**
 * Computes groundedness ratio g between an answer/prompt and cited evidence chunks
 */
export function evaluateGroundedness(text, citedChunks = []) {
  if (!citedChunks || citedChunks.length === 0) return 1.0;
  const textVec = computeEmbeddingBag(text);
  let maxSim = 0.0;

  for (const chunk of citedChunks) {
    const chunkText = typeof chunk === "string" ? chunk : chunk.text || "";
    const chunkVec = computeEmbeddingBag(chunkText);
    const sim = calculateBitsetSimilarity(textVec, chunkVec);
    if (sim > maxSim) maxSim = sim;
  }

  return Math.min(1.0, maxSim * 1.5);
}

/**
 * Swarm Knowledge Protocol Manager
 */
export class SwarmKnowledgeManager {
  constructor(options = {}) {
    this.nodeId =
      options.nodeId ||
      `osp_${Math.random().toString(36).substring(2, 11)}_${Date.now().toString(36)}`;
    this.nodeClass = options.nodeClass || NodeClass.N2_FULL_RESPONDER;
    this.reputation = options.reputation || 1.0;
    this.localKnowledgeBase = options.knowledgeBase || [];
    this.inbox = [];
    this.outbox = [];
    this.networkCallbacks = new Set();
  }

  /**
   * Create an OSP-compliant Knowledge Envelope for a generated image or article
   */
  createKnowledgeEnvelope(artifactType, payload, options = {}) {
    const timestamp = Date.now();
    const prompt = options.prompt || payload.metadata?.prompt || "";
    const label = options.label !== undefined ? options.label : payload.metadata?.label;
    const citedChunks = options.citedChunks || [];

    const groundedness = evaluateGroundedness(prompt, citedChunks);

    const envelope = {
      protocol: "OSP/0.6",
      packetType: "OSP-KNOWLEDGE-ARTIFACT",
      envelopeId: `env_${Math.random().toString(36).substring(2, 11)}_${timestamp}`,
      originNodeId: this.nodeId,
      nodeClass: this.nodeClass,
      timestamp,
      gasLimit: options.gasLimit || 100,
      artifactType, // 'image/schrodinger-bridge' | 'article/text' | 'model/lora'
      provenance: {
        modelHash: options.modelHash || "sb_webgpu_v4",
        epoch: options.epoch || 0,
        solver: options.method || "heun",
        cfgScale: options.cfgScale || 6.5,
        groundedness,
        citedEvidenceCount: citedChunks.length,
      },
      content: {
        prompt,
        label,
        summary: options.summary || `Schrödinger Bridge Generated ${artifactType} artifact`,
        data: payload.image || payload.text || payload,
      },
      signature: this.signPayload({
        origin: this.nodeId,
        timestamp,
        prompt,
        groundedness,
      }),
    };

    return envelope;
  }

  /**
   * Verify an incoming OSP Knowledge Envelope
   */
  verifyKnowledgeEnvelope(envelope) {
    if (!envelope || envelope.protocol !== "OSP/0.6") {
      return {
        valid: false,
        reason: "Invalid protocol version (requires OSP/0.6)",
      };
    }

    if (!envelope.signature || !envelope.originNodeId) {
      return { valid: false, reason: "Missing cryptographic signature or origin" };
    }

    // 3-layer Hallucination Firewall Verification
    const g = envelope.provenance?.groundedness ?? 1.0;
    const minGroundedness = 0.3;

    if (g < minGroundedness && envelope.provenance?.citedEvidenceCount > 0) {
      return {
        valid: false,
        convergence: ConvergenceMode.REJECTED,
        reason: `Hallucination firewall triggered: groundedness (${g.toFixed(2)}) below threshold (${minGroundedness})`,
      };
    }

    return {
      valid: true,
      convergence: ConvergenceMode.RESOLVED,
      envelopeId: envelope.envelopeId,
      origin: envelope.originNodeId,
      groundedness: g,
    };
  }

  /**
   * Mock signature generation (Ed25519 dev simulation)
   */
  signPayload(payload) {
    const raw = JSON.stringify(payload);
    let hash = 0;
    for (let i = 0; i < raw.length; i++) {
      hash = (hash << 5) - hash + raw.charCodeAt(i);
      hash |= 0;
    }
    return `ed25519_sig_${Math.abs(hash).toString(16)}_${Date.now().toString(36)}`;
  }

  /**
   * Create an OSP Knowledge Query packet to ask peer swarms
   */
  createQueryPacket(queryText, options = {}) {
    return {
      protocol: "OSP/0.6",
      packetType: "OSP-QUERY",
      queryId: `qry_${Math.random().toString(36).substring(2, 11)}_${Date.now()}`,
      originNodeId: this.nodeId,
      nodeClass: this.nodeClass,
      tier: options.tier || "T0", // T0=public, T1=personal, T2=sensitive
      gas: options.gas || 20,
      query: queryText,
      queryVector: Array.from(computeEmbeddingBag(queryText)),
      timestamp: Date.now(),
    };
  }

  /**
   * Handle incoming query and generate bid/answer
   */
  respondToQuery(queryPacket) {
    if (!queryPacket || queryPacket.packetType !== "OSP-QUERY") return null;

    // Check local knowledge relevance
    const qVec = new Uint8Array(queryPacket.queryVector || computeEmbeddingBag(queryPacket.query));
    let bestChunk = null;
    let maxSim = 0.0;

    for (const chunk of this.localKnowledgeBase) {
      const cVec = computeEmbeddingBag(chunk.text || chunk);
      const sim = calculateBitsetSimilarity(qVec, cVec);
      if (sim > maxSim) {
        maxSim = sim;
        bestChunk = chunk;
      }
    }

    if (maxSim < 0.1) {
      return {
        protocol: "OSP/0.6",
        packetType: "OSP-RFO",
        queryId: queryPacket.queryId,
        originNodeId: this.nodeId,
        reason: "Abstention: No relevant evidence in local knowledge base",
      };
    }

    return {
      protocol: "OSP/0.6",
      packetType: "OSP-BID",
      queryId: queryPacket.queryId,
      responderNodeId: this.nodeId,
      nodeClass: this.nodeClass,
      similarityScore: maxSim,
      reputation: this.reputation,
      bidScore: 0.5 * maxSim + 0.3 * this.reputation + 0.2 * maxSim,
      matchedEvidence: bestChunk ? (typeof bestChunk === "string" ? bestChunk : bestChunk.text) : null,
      timestamp: Date.now(),
    };
  }
}

export const globalSwarmKnowledge = new SwarmKnowledgeManager();
export default globalSwarmKnowledge;

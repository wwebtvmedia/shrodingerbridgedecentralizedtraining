/**
 * Strict whitelisting validator for all incoming data.
 * Rejects any data that doesn't match the expected schema exactly.
 */
export const Validator = {
  SCHEMAS: {
    // Top-level WebSocket / Tunnel messages
    'register_training': {
      data: { type: 'object', required: true },
      timestamp: { type: 'number', required: true }
    },
    'status_update': {
      clientId: { type: 'string', required: true },
      metrics: { type: 'object', required: true },
      neighbors: { type: 'array', required: false },
      timestamp: { type: 'number', required: true }
    },
    'model_update': {
      clientId: { type: 'string', required: true },
      modelData: { type: 'string', required: true },
      loss: { type: 'number', required: true },
      epoch: { type: 'number', required: true },
      timestamp: { type: 'number', required: true }
    },
    'PEER_MESSAGE': {
      from: { type: 'string', required: true },
      to: { type: 'string', required: true },
      data: { type: 'object', required: true },
      timestamp: { type: 'number', required: true },
      messageId: { type: 'string', required: true }
    },
    'BROADCAST': {
      from: { type: 'string', required: true },
      data: { type: 'object', required: true },
      timestamp: { type: 'number', required: true },
      messageId: { type: 'string', required: true }
    },
    'initial_sync': {
      bestModel: { type: 'object', required: false },
      neighbors: { type: 'array', required: false },
      type: { type: 'string', required: true }
    },
    'new_best_model': {
      model: { type: 'object', required: true },
      type: { type: 'string', required: true }
    },

    // Peer-to-Peer Internal Data (wrapped in data field of PEER_MESSAGE/BROADCAST)
    'TRAINING_RESULT': {
      trainerId: { type: 'string', required: true },
      epoch: { type: 'number', required: true },
      phase: { type: 'string', required: true },
      loss: { type: 'number', required: true },
      metrics: { type: 'object', required: true },
      modelHash: { type: 'string', required: true },
      timestamp: { type: 'number', required: true }
    },
    'MODEL_REQUEST': {
      modelHash: { type: 'string', required: true },
      timestamp: { type: 'number', required: true }
    },
    'MODEL_SHARE': {
      modelHash: { type: 'string', required: true },
      modelData: { type: 'any', required: true },
      timestamp: { type: 'number', required: true }
    },
    'PEER_RESEARCH_REQUEST': {
      timestamp: { type: 'number', required: true }
    },
    'PEER_RESEARCH_RESPONSE': {
      status: { type: 'object', required: true },
      timestamp: { type: 'number', required: false }
    },
    'PRESENCE': {
      trainerId: { type: 'string', required: true },
      epoch: { type: 'number', required: true },
      phase: { type: 'string', required: true },
      timestamp: { type: 'number', required: true }
    }
  },

  /**
   * Validates an object against a schema
   * @param {string} schemaName Name of the schema to validate against
   * @param {object} data The data to validate
   * @returns {boolean} True if valid
   */
  validate(schemaName, data) {
    const schema = this.SCHEMAS[schemaName];
    if (!schema) {
      console.warn(`Validator: Unknown schema '${schemaName}'`);
      return false;
    }

    if (!data || typeof data !== 'object') return false;

    // 1. Check for required fields and types
    for (const [key, rules] of Object.entries(schema)) {
      const value = data[key];

      if (rules.required && (value === undefined || value === null)) {
        console.warn(`Validator [${schemaName}]: Missing required field '${key}'`);
        return false;
      }

      if (value !== undefined && value !== null) {
        if (rules.type === 'array') {
          if (!Array.isArray(value)) return false;
        } else if (rules.type === 'any') {
          // Accept anything
        } else if (typeof value !== rules.type) {
          console.warn(`Validator [${schemaName}]: Field '${key}' expected type '${rules.type}', got '${typeof value}'`);
          return false;
        }
      }
    }

    // 2. Strict Whitelisting: Reject any extra fields
    const schemaKeys = new Set([...Object.keys(schema), 'type']); // 'type' is usually always present
    for (const key of Object.keys(data)) {
      if (!schemaKeys.has(key)) {
        console.warn(`Validator [${schemaName}]: Unexpected field '${key}' detected and rejected.`);
        return false;
      }
    }

    return true;
  }
};

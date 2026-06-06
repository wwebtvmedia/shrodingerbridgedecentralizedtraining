/**
 * Utility tool for sanitizing data sent from client to server.
 * Prevents injection attacks, XSS in logs, and ensures data integrity.
 */
export const Sanitizer = {
  /**
   * Recursively sanitizes an object or value
   */
  sanitize(data) {
    if (data === null || data === undefined) return data;

    if (typeof data === "string") {
      return this.sanitizeString(data);
    }

    if (Array.isArray(data)) {
      return data.map((item) => this.sanitize(item));
    }

    if (typeof data === "object") {
      // Build on a null-prototype object so an assigned "__proto__"/"constructor"
      // key cannot reach Object.prototype.
      const sanitized = Object.create(null);
      for (const [key, value] of Object.entries(data)) {
        // Sanitize keys to prevent JSON injection
        const safeKey = this.sanitizeString(key).replace(/[^a-zA-Z0-9_-]/g, "");
        // Explicitly drop keys that can pollute the prototype chain.
        if (Sanitizer.FORBIDDEN_KEYS.has(safeKey)) continue;
        sanitized[safeKey] = this.sanitize(value);
      }
      // Hand back a plain object literal copy (null-proto objects break some
      // consumers / JSON tooling), but only with the now-safe own keys.
      return Object.assign({}, sanitized);
    }

    return data;
  },

  // Keys that must never survive sanitization, regardless of casing of intent.
  FORBIDDEN_KEYS: new Set(["__proto__", "constructor", "prototype"]),

  /**
   * Sanitizes a string by removing potentially dangerous characters
   */
  sanitizeString(str) {
    if (typeof str !== "string") return str;

    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;")
      .replace(/\0/g, "") // Remove null bytes
      .trim();
  },

  /**
   * Specifically sanitizes metrics to ensure they are numeric where expected
   */
  sanitizeMetrics(metrics) {
    const safe = this.sanitize(metrics);
    if (safe.loss !== undefined) safe.loss = parseFloat(safe.loss) || 0;
    if (safe.epoch !== undefined) safe.epoch = parseInt(safe.epoch) || 0;
    return safe;
  },
};

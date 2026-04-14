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
      const sanitized = {};
      for (const [key, value] of Object.entries(data)) {
        // Sanitize keys to prevent prototype pollution or JSON injection
        const safeKey = this.sanitizeString(key).replace(/[^a-zA-Z0-9_-]/g, "");
        sanitized[safeKey] = this.sanitize(value);
      }
      return sanitized;
    }

    return data;
  },

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

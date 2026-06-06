// Swarm end-to-end checker.
// Verifies the live WebSocket swarm: auth -> host-signed identity -> register
// -> peer appears in the /api/peers directory -> clean departure.
//
// Usage (from the project dir so `ws` resolves):
//   SECRET_TOKEN=<token> BASE=https://www.tree4five.com node swarm-check.mjs
//
// BASE defaults to https://www.tree4five.com. The WS URL is derived from it
// (https->wss). SECRET_TOKEN must match the live server's SECRET_TOKEN.

import WebSocket from "ws";

const BASE = process.env.BASE || "https://www.tree4five.com";
const TOKEN = process.env.SECRET_TOKEN;
const WS_URL = BASE.replace(/^http/, "ws") + "/?token=" + encodeURIComponent(TOKEN || "");

let pass = 0,
  fail = 0;
const ok = (m) => (console.log("  ✅ " + m), pass++);
const bad = (m) => (console.log("  ❌ " + m), fail++);

if (!TOKEN) {
  console.error("SECRET_TOKEN env var is required (must match the live server).");
  process.exit(2);
}

const getPeers = async () => {
  const r = await fetch(BASE + "/api/peers");
  if (!r.ok) throw new Error("/api/peers -> HTTP " + r.status);
  return r.json(); // { count, peers: [{peerId, metadata}] }
};

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  console.log("Swarm check against " + BASE + "\n");

  // 1. Baseline directory read.
  let before;
  try {
    before = await getPeers();
    ok(`/api/peers reachable (count=${before.count})`);
  } catch (e) {
    bad("/api/peers failed: " + e.message);
    process.exit(1);
  }

  // 2. Connect + complete the handshake.
  const ws = new WebSocket(WS_URL);
  let identity = null;
  let gotRoster = false;
  const timeout = setTimeout(() => {
    bad("handshake timed out (no identity within 10s)");
    finish(ws);
  }, 10000);

  ws.on("unexpected-response", (_req, res) =>
    bad(`WS upgrade rejected: HTTP ${res.statusCode} (edge/WAF may block WebSocket upgrades)`),
  );
  ws.on("error", (e) => bad("WS error: " + e.message));
  ws.on("close", (code, reason) => {
    if (code === 4001) bad("closed 4001 Unauthorized — SECRET_TOKEN is wrong");
    else if (code === 4002) bad("closed 4002 Too many connections (per-IP cap)");
  });

  ws.on("open", () => ok("WebSocket connected + authenticated"));

  ws.on("message", async (raw) => {
    let msg;
    try {
      msg = JSON.parse(raw.toString());
    } catch {
      return;
    }

    if (msg.type === "identity") {
      clearTimeout(timeout);
      identity = msg;
      if (msg.peerId && msg.signature && typeof msg.issuedAt === "number")
        ok(`host-signed identity issued (${msg.peerId})`);
      else return bad("identity message malformed");

      // 3. Register, echoing the signature back so the host verifies us.
      ws.send(
        JSON.stringify({
          type: "register_training",
          data: { signature: identity.signature, name: "swarm-check", capabilities: ["test"] },
          timestamp: Date.now(),
        }),
      );

      // 4. Give the host a moment, then confirm we're in the directory.
      await wait(1500);
      try {
        const after = await getPeers();
        const mine = after.peers.find((p) => p.peerId === identity.peerId);
        if (mine) ok(`peer registered + visible in /api/peers (count ${before.count} -> ${after.count})`);
        else bad(`peer NOT in /api/peers after registering (count=${after.count}) — registration rejected?`);
      } catch (e) {
        bad("/api/peers re-read failed: " + e.message);
      }

      // 5. Clean departure: close and verify we leave the directory.
      ws.close(1000, "done");
      await wait(1500);
      try {
        const gone = await getPeers();
        if (!gone.peers.find((p) => p.peerId === identity.peerId))
          ok("peer removed from directory after disconnect");
        else bad("peer still listed after disconnect (phantom connection)");
      } catch (e) {
        bad("/api/peers final read failed: " + e.message);
      }
      summarize();
    }

    if (msg.type === "PEER_CONNECTED" && !gotRoster) {
      gotRoster = true;
      ok("received PEER_CONNECTED roster/announce");
    }
  });

  function finish(sock) {
    try {
      sock.close();
    } catch {}
    summarize();
  }
})();

function summarize() {
  console.log(`\nResult: ${pass} passed, ${fail} failed.`);
  process.exit(fail ? 1 : 0);
}

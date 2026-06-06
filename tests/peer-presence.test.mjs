// Smoke test for the peer discovery protocol:
//   auth -> identity (host-signed) -> register -> roster + PEER_CONNECTED/DISCONNECTED
// Two clients should discover each other via the host-issued identities, a
// disconnect should be announced, and a tampered signature must be rejected.
import { spawn } from "child_process";
import WebSocket from "ws";

const TOKEN = "testtoken123";
const PORT = 3099;

const server = spawn("node", ["server/index.js"], {
  env: { ...process.env, SECRET_TOKEN: TOKEN, PORT: String(PORT) },
  stdio: ["ignore", "pipe", "pipe"],
});
server.stdout.on("data", (d) => process.stdout.write(`[srv] ${d}`));
server.stderr.on("data", (d) => process.stderr.write(`[srv] ${d}`));

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// tamper: flip the registration signature to prove the host rejects forgeries.
function makeClient(label, { tamper = false } = {}) {
  const ws = new WebSocket(`ws://localhost:${PORT}`, [TOKEN]);
  const state = { ws, label, peerId: null, events: [] };
  ws.on("message", (raw) => {
    const m = JSON.parse(raw.toString());
    if (m.type === "identity") {
      state.peerId = m.peerId; // host-assigned, signed
      ws.send(
        JSON.stringify({
          type: "register_training",
          data: {
            peerId: m.peerId,
            name: label,
            issuedAt: m.issuedAt,
            signature: tamper ? `${m.signature}00` : m.signature,
          },
          timestamp: Date.now(),
        }),
      );
    } else if (m.type === "PEER_CONNECTED" || m.type === "PEER_DISCONNECTED") {
      state.events.push(m);
    }
  });
  return state;
}

let failed = false;
const assert = (cond, msg) => {
  console.log(`${cond ? "✅" : "❌"} ${msg}`);
  if (!cond) failed = true;
};

try {
  await sleep(800); // let server boot

  const A = makeClient("trainer_A");
  await sleep(300);
  const B = makeClient("trainer_B");
  await sleep(500);

  assert(
    /^peer_[0-9a-f]{32}$/.test(A.peerId || ""),
    "A got a host-issued random id",
  );
  assert(A.peerId !== B.peerId, "A and B have distinct ids");
  assert(
    A.events.some((e) => e.type === "PEER_CONNECTED" && e.peerId === B.peerId),
    "A sees B connect (by host-issued id)",
  );
  assert(
    B.events.some((e) => e.type === "PEER_CONNECTED" && e.peerId === A.peerId),
    "B sees A connect (by host-issued id)",
  );

  const dir = await (await fetch(`http://localhost:${PORT}/api/peers`)).json();
  assert(dir.count === 2, `directory lists 2 peers (got ${dir.count})`);

  // A forger with a tampered signature must NOT join the directory.
  const C = makeClient("forger_C", { tamper: true });
  await sleep(500);
  const dir2 = await (await fetch(`http://localhost:${PORT}/api/peers`)).json();
  assert(dir2.count === 2, `forged signature rejected (still ${dir2.count})`);
  C.ws.close();

  // B leaves; A should be told.
  B.ws.close();
  await sleep(500);
  assert(
    A.events.some(
      (e) => e.type === "PEER_DISCONNECTED" && e.peerId === B.peerId,
    ),
    "A sees B disconnect",
  );

  A.ws.close();
} finally {
  server.kill();
  await sleep(200);
  process.exit(failed ? 1 : 0);
}

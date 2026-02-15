import { EventEmitter } from "events";

// Use globalThis to survive Next.js hot reloads in dev
const globalForBus = globalThis as unknown as { __stepBus?: EventEmitter };

if (!globalForBus.__stepBus) {
  globalForBus.__stepBus = new EventEmitter();
  globalForBus.__stepBus.setMaxListeners(50);
}

const bus = globalForBus.__stepBus;

export function publishStep(
  patientId: string,
  data: Record<string, unknown>
) {
  bus.emit(`steps:${patientId}`, data);
}

export function subscribeSteps(
  patientId: string,
  callback: (data: Record<string, unknown>) => void
): () => void {
  const key = `steps:${patientId}`;
  bus.on(key, callback);
  return () => {
    bus.off(key, callback);
  };
}

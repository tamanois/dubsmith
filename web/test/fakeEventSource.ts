import { vi } from "vitest";

type Listener = (event: MessageEvent<string>) => void;

/** Minimal EventSource replacement for `subscribeJob` tests. */
export class FakeEventSource {
  static instances: FakeEventSource[] = [];

  readonly url: string;
  readonly listeners = new Map<string, Listener[]>();
  readonly close = vi.fn();
  onerror: (() => void) | null = null;

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }

  addEventListener(name: string, listener: Listener): void {
    const current = this.listeners.get(name) ?? [];
    current.push(listener);
    this.listeners.set(name, current);
  }

  emit(name: string, payload: unknown): void {
    const data = typeof payload === "string" ? payload : JSON.stringify(payload);
    for (const listener of this.listeners.get(name) ?? []) {
      listener({ data } as MessageEvent<string>);
    }
  }

  fail(): void {
    this.onerror?.();
  }

  static reset(): void {
    FakeEventSource.instances = [];
  }
}

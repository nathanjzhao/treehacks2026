/* eslint-disable @typescript-eslint/no-explicit-any */

export interface SSECallbacks {
  onStep?: (event: {
    index: number;
    label: string;
    status: string;
    detail?: string;
    searches?: string[];
  }) => void;
  onStepDone?: (event: { index: number }) => void;
  onText?: (event: { chunk: string }) => void;
  onResult?: (event: {
    ok: boolean;
    reply: string;
    action: string;
    request_id?: string;
    object_name?: string;
    event_id?: string;
    citations?: Array<{ title?: string; url: string; evidence_grade?: string }>;
  }) => void;
  onError?: (error: Error) => void;
}

export async function consumeChatSSE(
  patientId: string,
  message: string,
  callbacks: SSECallbacks
): Promise<void> {
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ patient_id: patientId, message }),
    });

    if (!res.ok || !res.body) {
      callbacks.onError?.(new Error("Chat request failed"));
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const jsonStr = line.slice(6);
        if (!jsonStr) continue;

        try {
          const event = JSON.parse(jsonStr);
          switch (event.type) {
            case "step":
              callbacks.onStep?.(event);
              break;
            case "step_done":
              callbacks.onStepDone?.(event);
              break;
            case "text":
              callbacks.onText?.(event);
              break;
            case "result":
              callbacks.onResult?.(event);
              break;
          }
        } catch {
          // skip malformed JSON
        }
      }
    }
  } catch (err) {
    callbacks.onError?.(
      err instanceof Error ? err : new Error("Unknown error")
    );
  }
}

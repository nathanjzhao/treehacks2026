import { NextRequest } from "next/server";
import { getSupabaseServerClient } from "@/lib/supabase-server";
import { appendEvent, getRecentEvents, logEscalation } from "@/lib/event-spine";
import { sendSms } from "@/lib/twilio";

// ────────────────────────────────────────────────────────────────
// Tool definitions for function calling
// ────────────────────────────────────────────────────────────────

const TOOLS = [
  {
    type: "function" as const,
    function: {
      name: "find_object",
      description:
        "Search for a physical object that the resident has lost or is looking for. Use this when the resident mentions they can't find something, asks where something is, or needs help locating an item.",
      parameters: {
        type: "object",
        properties: {
          object_name: {
            type: "string",
            description:
              "The name of the object to find (e.g. 'glasses', 'pills', 'remote', 'phone')",
          },
        },
        required: ["object_name"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "escalate_to_caregiver",
      description:
        "Alert the caregiver immediately. Use this when the resident reports an emergency, is in distress, mentions falling, chest pain, difficulty breathing, severe pain, bleeding, or any situation requiring immediate human help. ALWAYS use this for safety-critical situations - err on the side of caution.",
      parameters: {
        type: "object",
        properties: {
          reason: {
            type: "string",
            description: "Brief description of why escalation is needed",
          },
          urgency: {
            type: "string",
            enum: ["critical", "high", "medium"],
            description: "How urgent is this situation",
          },
        },
        required: ["reason", "urgency"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "lookup_medication",
      description:
        "Look up details about a specific medication from the resident's record. Use this when the resident asks about a particular medication - when to take it, what it's for, whether to take it with food, etc.",
      parameters: {
        type: "object",
        properties: {
          medication_query: {
            type: "string",
            description:
              "The medication name or what the resident is asking about",
          },
        },
        required: ["medication_query"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "search_medical_info",
      description:
        "Search for current medical information online using Perplexity. Use this when the resident asks about medication side effects, drug interactions, general health questions, symptoms, recent medical guidelines, or anything requiring current medical knowledge that isn't available in their patient record. Do NOT use this for finding physical objects or emergencies.",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description:
              "The medical question to research, contextualized with relevant patient details (age, conditions) but without personally identifying information",
          },
        },
        required: ["query"],
      },
    },
  },
];

// ────────────────────────────────────────────────────────────────
// PHI Firewall
// ────────────────────────────────────────────────────────────────

interface PatientCardShape {
  display_name?: string;
  demographics?: { age_range?: string; sex?: string };
  conditions?: Array<{ name: string; onset_year?: number }>;
  allergies?: Array<{ substance: string; reaction?: string }>;
  meds?: Array<{
    name: string;
    strength?: string;
    schedule_times?: string[];
    purpose?: string;
    with_food?: boolean;
    is_critical?: boolean;
  }>;
  contacts?: { caregiver_name?: string; caregiver_phone?: string };
}

function minimizeForLLM(card: PatientCardShape): string {
  return JSON.stringify(
    {
      demographics: {
        age_range: card.demographics?.age_range || "unknown",
        sex: card.demographics?.sex || "unknown",
      },
      conditions: card.conditions?.map((c) => c.name) || [],
      allergies: card.allergies?.map((a) => a.substance) || [],
      medications:
        card.meds?.map((m) => ({
          name: m.name,
          strength: m.strength,
          schedule: m.schedule_times,
          purpose: m.purpose || "not specified",
          with_food: m.with_food,
          is_critical: m.is_critical,
        })) || [],
    },
    null,
    2
  );
}

// ────────────────────────────────────────────────────────────────
// Perplexity Sonar (web search for medical info)
// ────────────────────────────────────────────────────────────────

interface SonarResult {
  answer: string;
  citations: Array<{ title?: string; url: string }>;
}

async function callSonar(
  query: string,
  patientCard: PatientCardShape
): Promise<SonarResult> {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) throw new Error("OPENROUTER_API_KEY not configured");

  // Contextualize with de-identified patient info
  const contextParts = [
    patientCard.demographics?.age_range
      ? `Patient: ${patientCard.demographics.age_range} year old`
      : "",
    patientCard.demographics?.sex || "",
    patientCard.conditions?.length
      ? `Conditions: ${patientCard.conditions.map((c) => c.name).join(", ")}`
      : "",
    patientCard.meds?.length
      ? `Current medications: ${patientCard.meds.map((m) => m.name).join(", ")}`
      : "",
  ]
    .filter(Boolean)
    .join(". ");

  const fullQuery = contextParts ? `${contextParts}. Question: ${query}` : query;

  const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      "HTTP-Referer":
        process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000",
      "X-Title": "Mira",
    },
    body: JSON.stringify({
      model: "perplexity/sonar",
      messages: [
        {
          role: "system",
          content:
            "You are a medical information assistant for an assisted living facility. Provide accurate, evidence-based medical information. Keep answers concise (2-4 sentences). Always note that this is general information and the patient should consult their healthcare provider for personalized advice.",
        },
        { role: "user", content: fullQuery },
      ],
      temperature: 0.3,
      max_tokens: 500,
    }),
  });

  if (!res.ok) {
    const errText = await res.text();
    console.error("[Sonar] Error:", res.status, errText);
    throw new Error(`Sonar API error: ${res.status}`);
  }

  const data = await res.json();
  const choice = data.choices?.[0];
  const answer = choice?.message?.content || "No results found.";

  // Sonar returns citations in the top-level response
  const rawCitations = data.citations || [];
  const citations: Array<{ title?: string; url: string }> = rawCitations.map(
    (c: string | { url: string; title?: string }) =>
      typeof c === "string" ? { url: c } : c
  );

  return { answer, citations };
}

// ────────────────────────────────────────────────────────────────
// System prompt
// ────────────────────────────────────────────────────────────────

const SYSTEM_PROMPT = `You are Mira, a warm, friendly, and genuinely caring AI assistant for residents in an assisted living facility. You talk like a kind, knowledgeable friend - not a robot.

PERSONALITY:
- Warm and conversational. Use natural language. Say "Sure thing!" not "Certainly, I can assist you."
- Remember context from the conversation. If they mentioned something earlier, reference it.
- Show empathy. If they seem frustrated or worried, acknowledge it before helping.
- Keep responses concise (1-3 sentences) but never feel rushed. Be present.
- Use the resident's name occasionally when it feels natural.

TOOLS - YOU MUST USE THEM:
- If the resident can't find something or asks where something is → call find_object
- If there's ANY safety concern (pain, falls, breathing, emergencies) → call escalate_to_caregiver IMMEDIATELY
- If they ask about a specific medication → call lookup_medication to give accurate details
- You can call multiple tools if needed (e.g. "my chest hurts and I can't find my inhaler" → escalate + find_object)

GUARDRAILS:
1. You NEVER diagnose or prescribe. You inform and explain based on the record only.
2. For anything potentially urgent, ALWAYS escalate. Better safe than sorry.
3. If info isn't in their record, say so honestly: "I don't have that in your file, but your caregiver would know."
4. Never reveal raw data formats. Speak naturally about medical information.
5. For medication timing questions, always reference their specific schedule from the record.

When you use tools, still respond conversationally. Don't just say "I've escalated" - show you care.`;

// ────────────────────────────────────────────────────────────────
// Conversation history builder
// ────────────────────────────────────────────────────────────────

interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

function buildConversationHistory(
  recentEvents: Array<{
    type: string;
    receipt_text?: string;
    created_at: string;
  }>
): ChatTurn[] {
  const chatEvents = recentEvents
    .filter(
      (e) =>
        e.type === "CHAT_USER_UTTERANCE" ||
        e.type === "CHAT_ASSISTANT_RESPONSE"
    )
    .reverse();

  const trimmed = chatEvents.slice(-20);

  return trimmed.map((e) => ({
    role:
      e.type === "CHAT_USER_UTTERANCE"
        ? ("user" as const)
        : ("assistant" as const),
    content: e.receipt_text || "",
  }));
}

// ────────────────────────────────────────────────────────────────
// LLM call
// ────────────────────────────────────────────────────────────────

/* eslint-disable @typescript-eslint/no-explicit-any */

interface ToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

interface LLMResponse {
  reply: string;
  toolCalls: ToolCall[];
  rawAssistantMessage: Record<string, any>;
}

async function callLLM(
  messages: Array<Record<string, any>>,
  includeTools: boolean = true
): Promise<LLMResponse> {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) throw new Error("OPENROUTER_API_KEY not configured");

  const body: Record<string, unknown> = {
    model: "openai/gpt-4o-mini",
    messages,
    temperature: 0.5,
    max_tokens: 400,
  };

  if (includeTools) {
    body.tools = TOOLS;
    body.tool_choice = "auto";
  }

  const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      "HTTP-Referer":
        process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000",
      "X-Title": "Mira",
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const errText = await res.text();
    console.error("[OpenRouter] Error:", res.status, errText);
    throw new Error(`OpenRouter API error: ${res.status}`);
  }

  const data = await res.json();
  const choice = data.choices?.[0];

  return {
    reply: choice?.message?.content || "",
    toolCalls: choice?.message?.tool_calls || [],
    rawAssistantMessage: choice?.message || {},
  };
}

// ────────────────────────────────────────────────────────────────
// Streaming LLM call (token-by-token)
// ────────────────────────────────────────────────────────────────

async function callLLMStreaming(
  messages: Array<Record<string, any>>,
  onChunk: (text: string) => void
): Promise<string> {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) throw new Error("OPENROUTER_API_KEY not configured");

  const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      "HTTP-Referer":
        process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000",
      "X-Title": "Mira",
    },
    body: JSON.stringify({
      model: "openai/gpt-4o-mini",
      messages,
      temperature: 0.5,
      max_tokens: 400,
      stream: true,
    }),
  });

  if (!res.ok) {
    const errText = await res.text();
    console.error("[OpenRouter Stream] Error:", res.status, errText);
    throw new Error(`OpenRouter API error: ${res.status}`);
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let fullReply = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const data = line.slice(6).trim();
      if (data === "[DONE]") continue;

      try {
        const parsed = JSON.parse(data);
        const delta = parsed.choices?.[0]?.delta?.content;
        if (delta) {
          fullReply += delta;
          onChunk(delta);
        }
      } catch {
        // skip malformed chunk
      }
    }
  }

  return fullReply;
}

// ────────────────────────────────────────────────────────────────
// Tool execution
// ────────────────────────────────────────────────────────────────

interface ToolResult {
  action: "FIND_OBJECT" | "ESCALATE" | "ANSWER";
  request_id?: string;
  object_name?: string;
  toolOutput: string;
}

async function executeTool(
  toolCall: ToolCall,
  patient_id: string,
  patientCard: PatientCardShape
): Promise<ToolResult> {
  const args = JSON.parse(toolCall.function.arguments);
  const supabase = getSupabaseServerClient();

  switch (toolCall.function.name) {
    case "find_object": {
      const objectName = args.object_name;

      const { data: reqRow, error: reqErr } = await supabase
        .from("object_requests")
        .insert({
          patient_id,
          object_name: objectName,
          context: "chat_request",
          status: "PENDING",
        })
        .select("id")
        .single();

      if (reqErr) throw reqErr;

      await appendEvent({
        patient_id,
        type: "FIND_OBJECT_REQUESTED",
        severity: "GREEN",
        receipt_text: `Looking for: ${objectName}`,
        payload: { object_name: objectName, request_id: reqRow.id },
        source: "device",
      });

      return {
        action: "FIND_OBJECT",
        request_id: reqRow.id,
        object_name: objectName,
        toolOutput: `Object search initiated for "${objectName}". Request ID: ${reqRow.id}. The camera system is now looking for it. The resident will be notified when found.`,
      };
    }

    case "escalate_to_caregiver": {
      const { reason, urgency } = args;

      const escalateEvent = await appendEvent({
        patient_id,
        type: "ESCALATED",
        severity: "RED",
        receipt_text: `Escalation: ${reason}`,
        payload: { reason, urgency, source: "chat_assistant" },
        source: "device",
      });

      await logEscalation({
        patient_id,
        event_id: escalateEvent.id,
        channel: "sms",
        status: "sent",
        response_payload: { reason, urgency },
      });

      const displayName = patientCard.display_name || "A resident";
      sendSms(
        `Mira Alert [${urgency.toUpperCase()}]: ${displayName} - ${reason}`
      ).catch(console.error);

      return {
        action: "ESCALATE",
        toolOutput: `Caregiver has been notified. Urgency: ${urgency}. Reason: ${reason}. SMS alert sent to ${patientCard.contacts?.caregiver_name || "caregiver"}.`,
      };
    }

    case "lookup_medication": {
      const query = args.medication_query.toLowerCase();
      const meds = patientCard.meds || [];
      const match = meds.find(
        (m) =>
          m.name.toLowerCase().includes(query) ||
          query.includes(m.name.toLowerCase())
      );

      if (match) {
        return {
          action: "ANSWER",
          toolOutput: JSON.stringify({
            found: true,
            name: match.name,
            strength: match.strength,
            schedule: match.schedule_times,
            purpose: match.purpose,
            with_food: match.with_food,
            is_critical: match.is_critical,
          }),
        };
      }

      return {
        action: "ANSWER",
        toolOutput: JSON.stringify({
          found: false,
          message: `No medication matching "${args.medication_query}" found in the resident's record.`,
          available_meds: meds.map((m) => m.name),
        }),
      };
    }

    case "search_medical_info": {
      const query = args.query;

      try {
        const sonarResult = await callSonar(query, patientCard);

        await appendEvent({
          patient_id,
          type: "WEB_SEARCH_COMPLETED",
          severity: "GREEN",
          receipt_text: `Researched: ${query}`,
          payload: {
            query,
            citations: sonarResult.citations,
            source_model: "perplexity/sonar",
          },
          source: "backend",
        });

        return {
          action: "ANSWER",
          toolOutput: JSON.stringify({
            answer: sonarResult.answer,
            citations: sonarResult.citations,
            disclaimer:
              "This is general medical information. Always consult your healthcare provider.",
          }),
        };
      } catch (err) {
        console.error("[Sonar] Search failed:", err);
        return {
          action: "ANSWER",
          toolOutput: JSON.stringify({
            error: "Web search temporarily unavailable",
            fallback:
              "I couldn't look that up right now. Please ask your caregiver for detailed medical information.",
          }),
        };
      }
    }

    default:
      return { action: "ANSWER", toolOutput: "Unknown tool called." };
  }
}

// ────────────────────────────────────────────────────────────────
// Demo fallback
// ────────────────────────────────────────────────────────────────

function getDemoResponse(message: string, card: PatientCardShape): string {
  const lower = message.toLowerCase();
  if (lower.includes("med") || lower.includes("pill")) {
    const meds = card.meds || [];
    if (meds.length === 0)
      return "I don't see any medications in your record right now.";
    return `You have ${meds.length} medications: ${meds.map((m) => `${m.name} ${m.strength || ""}`).join(", ")}. Would you like details on any of them?`;
  }
  if (lower.includes("allerg")) {
    const allergies = card.allergies || [];
    return allergies.length > 0
      ? `Your known allergies are: ${allergies.map((a) => a.substance).join(", ")}.`
      : "You have no known allergies on file.";
  }
  return "I can help with questions about your medications, conditions, allergies, or help find things. What would you like to know?";
}

// ────────────────────────────────────────────────────────────────
// Tool name to human-readable step label
// ────────────────────────────────────────────────────────────────

function toolLabel(name: string, args: Record<string, any>): {
  label: string;
  searches?: string[];
} {
  switch (name) {
    case "find_object":
      return {
        label: `Searching for ${args.object_name || "item"}`,
        searches: [args.object_name || "item"],
      };
    case "escalate_to_caregiver":
      return {
        label: "Alerting caregiver",
      };
    case "lookup_medication":
      return {
        label: "Looking up medication",
        searches: [args.medication_query || "medication"],
      };
    case "search_medical_info":
      return {
        label: "Researching online",
        searches: [args.query || "medical info"],
      };
    default:
      return { label: name };
  }
}

// ────────────────────────────────────────────────────────────────
// SSE Streaming handler
// ────────────────────────────────────────────────────────────────

export async function POST(request: NextRequest) {
  const body = await request.json();
  const { patient_id, message } = body;

  if (!patient_id || typeof patient_id !== "string") {
    return new Response(
      JSON.stringify({ ok: false, error: "patient_id is required" }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }
  if (!message || typeof message !== "string") {
    return new Response(
      JSON.stringify({ ok: false, error: "message is required" }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }

  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    async start(controller) {
      let stepIndex = 0;

      const send = (data: Record<string, any>) => {
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify(data)}\n\n`)
        );
      };

      const emitStep = (
        label: string,
        status: "active" | "done",
        detail?: string,
        searches?: string[]
      ) => {
        send({ type: "step", index: stepIndex, label, status, detail, searches });
        if (status === "active") stepIndex++;
      };

      const markDone = (index: number) => {
        // Mark a previously emitted step as done (re-emit with same index)
        // Client handles this by updating the step at that index
        send({ type: "step_done", index });
      };

      try {
        const supabase = getSupabaseServerClient();

        // Step: Interpreting
        emitStep("Interpreting your message", "active");

        // Log user utterance
        const userEvent = await appendEvent({
          patient_id,
          type: "CHAT_USER_UTTERANCE",
          severity: "GREEN",
          receipt_text: message,
          source: "device",
        });

        markDone(0);

        // Step: Loading records
        emitStep("Reviewing health records", "active");

        const { data: cardData } = await supabase
          .from("patient_cards")
          .select("card_json")
          .eq("patient_id", patient_id)
          .order("created_at", { ascending: false })
          .limit(1)
          .maybeSingle();

        const patientCard = (cardData?.card_json as PatientCardShape) || {};
        const context = minimizeForLLM(patientCard);

        markDone(1);

        // Step: Recalling conversation
        emitStep("Recalling conversation", "active");

        const recentEvents = await getRecentEvents(patient_id, 30);
        const conversationHistory = buildConversationHistory(
          recentEvents as Array<{
            type: string;
            receipt_text?: string;
            created_at: string;
          }>
        );

        markDone(2);

        // Build LLM messages
        const llmMessages: Array<Record<string, any>> = [
          {
            role: "system",
            content: `${SYSTEM_PROMPT}\n\nResident's Health Record:\n${context}`,
          },
          ...conversationHistory.map((turn) => ({
            role: turn.role,
            content: turn.content,
          })),
          { role: "user", content: message },
        ];

        let reply: string;
        let action: "FIND_OBJECT" | "ESCALATE" | "ANSWER" = "ANSWER";
        let request_id: string | undefined;
        let object_name: string | undefined;
        let citations: Array<{ title?: string; url: string }> = [];

        try {
          // Step: Thinking
          emitStep("Thinking", "active", "Analyzing intent\u2026");

          const firstResponse = await callLLM(llmMessages);
          markDone(3);

          if (firstResponse.toolCalls.length > 0) {
            // Execute tools - emit a step for each
            const toolResults: ToolResult[] = [];
            for (const tc of firstResponse.toolCalls) {
              const args = JSON.parse(tc.function.arguments);
              const { label, searches } = toolLabel(tc.function.name, args);
              const currentIdx = stepIndex;
              emitStep(label, "active", undefined, searches);

              const result = await executeTool(tc, patient_id, patientCard);
              toolResults.push(result);
              markDone(currentIdx);
            }

            // Set action from results
            if (toolResults.some((r) => r.action === "ESCALATE")) {
              action = "ESCALATE";
            } else if (toolResults.some((r) => r.action === "FIND_OBJECT")) {
              action = "FIND_OBJECT";
              const findResult = toolResults.find(
                (r) => r.action === "FIND_OBJECT"
              );
              request_id = findResult?.request_id;
              object_name = findResult?.object_name;
            }

            // Collect citations from Sonar results
            for (let ti = 0; ti < firstResponse.toolCalls.length; ti++) {
              if (firstResponse.toolCalls[ti].function.name === "search_medical_info") {
                try {
                  const parsed = JSON.parse(toolResults[ti].toolOutput);
                  if (parsed.citations) citations = parsed.citations;
                } catch { /* ignore */ }
              }
            }

            // Step: Composing response
            const composeIdx = stepIndex;
            emitStep("Composing response", "active");

            const followUpMessages: Array<Record<string, any>> = [
              ...llmMessages,
              firstResponse.rawAssistantMessage,
              ...firstResponse.toolCalls.map((tc, i) => ({
                role: "tool",
                content: toolResults[i].toolOutput,
                tool_call_id: tc.id,
              })),
            ];

            reply = await callLLMStreaming(followUpMessages, (chunk) => {
              send({ type: "text", chunk });
            });
            if (!reply) {
              reply = firstResponse.reply || "I've taken care of that for you.";
            }

            markDone(composeIdx);
          } else {
            // No tools — emit the reply as text chunks for streaming effect
            reply = firstResponse.reply;
            if (reply) {
              // Split into small chunks for visual streaming
              const words = reply.split(/(\s+)/);
              for (const word of words) {
                if (word) send({ type: "text", chunk: word });
              }
            }
          }

          if (!reply) {
            reply = getDemoResponse(message, patientCard);
          }
        } catch (err) {
          console.error("LLM error:", err);
          reply = getDemoResponse(message, patientCard);
        }

        // Log assistant response
        const assistantEvent = await appendEvent({
          patient_id,
          type: "CHAT_ASSISTANT_RESPONSE",
          severity: action === "ESCALATE" ? "RED" : "GREEN",
          receipt_text: reply,
          payload: {
            action,
            linked_user_event_id: userEvent.id,
            ...(object_name ? { object_name } : {}),
            ...(request_id ? { request_id } : {}),
            ...(citations.length > 0 ? { citations } : {}),
          },
          source: "backend",
        });

        // Send final result
        send({
          type: "result",
          ok: true,
          reply,
          action,
          ...(request_id ? { request_id } : {}),
          ...(object_name ? { object_name } : {}),
          ...(citations.length > 0 ? { citations } : {}),
          event_id: assistantEvent.id,
        });
      } catch (error) {
        console.error("Chat stream error:", error);
        send({
          type: "result",
          ok: false,
          reply: "Sorry, something went wrong. Please try again.",
          action: "ANSWER",
        });
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}

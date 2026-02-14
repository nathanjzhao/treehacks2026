import { NextRequest, NextResponse } from "next/server";
import { appendEvent, getEvents } from "@/lib/event-spine";
import { parseEventInput, parseQueryFilters } from "@/lib/contracts";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const input = parseEventInput(body);
    const event = await appendEvent(input);
    return NextResponse.json({ ok: true, event }, { status: 201 });
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "unknown_error" },
      { status: 400 }
    );
  }
}

export async function GET(request: NextRequest) {
  try {
    const { patient_id, since } = parseQueryFilters(request.nextUrl.searchParams);
    const events = await getEvents(patient_id, since);
    return NextResponse.json({ ok: true, events });
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "unknown_error" },
      { status: 400 }
    );
  }
}

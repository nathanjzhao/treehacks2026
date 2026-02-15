# AR Care System — Architecture Flowchart

Interactive system architecture diagram for the AR-Assisted Care pipeline.

## Usage

Open `flowchart.html` in a browser.

## Modes

- **EXPLORE** (default) — Hover any node to highlight its connections, upstream sources, and downstream targets.
- **DATA FLOW** — Animates through predefined data flow paths, showing how data moves between components step by step. Auto-scrolls to follow the active node.

Toggle between modes using the buttons in the top-right corner.

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `←` / `→` | Skip to previous / next data flow path (in DATA FLOW mode) |
| `A` | Switch LLM Agent badge to **Claude** (Anthropic) |
| `S` | Switch LLM Agent badge to **GPT-4o** (OpenAI) |
| `D` | Switch LLM Agent badge to **Gemini** (Google) |

## Brand Badges

Nodes display the service/platform they run on:

- **Meta** — Ray-Ban smart glasses hardware
- **Modal** — All offline processing (3D reconstruction, object detection, reference images, point cloud, spatial DB, reference DB)
- **OpenAI** — Speech-to-Text (Whisper) and Text-to-Speech
- **Claude / GPT-4o / Gemini** — LLM Agent (switchable with A/S/D)
- **OpenEvidence** — Patient Info lookup
- **Perplexity** — Medication Lookup (web search)

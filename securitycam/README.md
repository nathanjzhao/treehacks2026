# Phone Camera → localhost Stream

Stream your iPhone camera to `localhost` on your computer over any network using mediamtx + Larix Broadcaster + Tailscale.

## Prerequisites

- **mediamtx** — lightweight media server ([releases](https://github.com/bluenviron/mediamtx/releases))
- **Larix Broadcaster** — iOS RTMP streaming app ([App Store](https://apps.apple.com/us/app/larix-broadcaster/id1042474385))
- **Tailscale** — mesh VPN so both devices can reach each other on any network ([tailscale.com](https://tailscale.com))

## Setup

### 1. Install mediamtx

```bash
# macOS
brew install mediamtx

# Or download binary from GitHub releases for your platform
```

### 2. Create config

```yaml
# mediamtx.yml
paths:
  all_others:
```

This accepts any incoming stream path.

### 3. Install Tailscale

Install on both your computer and iPhone. Sign into the same account.

Get your computer's Tailscale IP:

```bash
tailscale ip -4
# e.g. 100.64.x.x
```

### 4. Configure Larix Broadcaster

1. Open Larix → **Settings** → **Connections** → **+** → **Connection**
2. Set URL to: `rtmp://<tailscale-ip>:1935/live/stream`
3. Save

### 5. Start streaming

```bash
# Terminal
mediamtx mediamtx.yml
```

Then tap the red record button in Larix.

## Endpoints

Once the stream is live, access it on your computer at:

| Protocol | URL                                      | Latency |
|----------|------------------------------------------|---------|
| WebRTC   | `http://localhost:8889/live/stream/`     | ~200ms  |
| RTSP     | `rtsp://localhost:8554/live/stream`      | ~500ms  |
| HLS      | `http://localhost:8888/live/stream/`     | 2-6s    |

Open the WebRTC URL in a browser to verify the stream is working.

## Notes

- Tailscale adds negligible latency (~1-5ms LAN, depends on internet for remote)
- Larix free tier adds a watermark after 30 min — restart the stream to reset
- mediamtx requires no dependencies; it's a single binary
- The `all_others` path config accepts any stream name, so you can change `live/stream` to whatever you want
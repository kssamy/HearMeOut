# MeetNotes

Local-first meeting transcription, prototyped in Python (Granola-style: no bot
joins your call — audio is captured on your Mac). Microphone (+ optionally the
other side of the call via a loopback device) → Deepgram streaming ASR → live
diarized transcript in the browser, with every finalized segment persisted to
SQLite as it arrives.

## Prerequisites

- macOS (Apple Silicon or Intel)
- [uv](https://docs.astral.sh/uv/) (`brew install uv`) — it will fetch
  Python 3.12 automatically
- PortAudio is bundled with the `sounddevice` wheel; no extra install needed
- A Deepgram API key — sign up free at
  [console.deepgram.com](https://console.deepgram.com) and create a key
  (the free tier is plenty for testing)

## Run it

```sh
uv sync
uv run meetnotes
```

The server starts on `http://127.0.0.1:8756` and opens your browser. On first
run you'll be steered to **Settings** — paste your Deepgram key there. It is
stored in the **macOS Keychain** (never on disk, never in env files or logs).

Then click **New meeting** and start talking. Interim text appears in gray
within about a second; finalized lines commit with `[speaker N]` labels and
timestamps, and are written to SQLite immediately (`~/.meetnotes/meetnotes.db`).

## Microphone permission (macOS)

The first capture attempt makes macOS prompt for microphone access for the
process that launched MeetNotes (your terminal). If you denied it — or if the
transcript stays empty and the app shows the "No microphone signal" help —
fix it here:

**System Settings → Privacy & Security → Microphone** → enable your terminal
app (Terminal, iTerm2, VS Code, …), then restart `uv run meetnotes`.

## Behavior worth knowing

- **Network drops are lossless.** If the Deepgram connection fails, audio is
  buffered locally (up to 10 minutes, warning at 80%) and the UI shows a
  reconnecting banner with the buffered-seconds count. On reconnect
  (exponential backoff, 0.25 s → 8 s) all unacknowledged audio is replayed.
- **Crash-safe.** Finalized segments are committed the moment they arrive.
  A `kill -9` loses at most the in-flight interim line; on restart the
  interrupted meeting is closed with its transcript intact.
- **Channel-aware.** Every frame and segment carries a `channel`
  (0 = mic/"Me", 1 = system audio/"Them").

## System audio — transcribing the other side of a call

MeetNotes can record what *you hear* (Zoom, Meet, …) as channel 1 via a
loopback driver:

1. `brew install blackhole-2ch`, then restart MeetNotes.
2. Open **Audio MIDI Setup** → “+” → **Create Multi-Output Device**; check both
   your speakers/headphones and *BlackHole 2ch*.
3. Select that Multi-Output Device as your Mac's sound output during calls
   (you keep hearing the call; BlackHole gets a copy).
4. In MeetNotes **Settings → System audio**, pick *BlackHole 2ch*, click
   **Test signal** while playing any audio to see the level meter move, then
   **Use this device**.

Meetings then stream two channels to Deepgram (multichannel), and the live
transcript labels lines **Me** / **Them**. If the saved device disappears
(e.g. BlackHole uninstalled), meetings fall back to mic-only with a warning.

## Project layout

```
src/meetnotes/
  main.py           uv run meetnotes entry point
  app.py            FastAPI factory + lifespan (DB, orphan recovery)
  api.py            REST + /ws endpoints
  session.py        MeetingSession: captures → ring → ASR → SQLite + browser
  capture.py        sounddevice input capture (PortAudio callback → asyncio)
  devices.py        input enumeration, BlackHole detection, level probe
  mixer.py          pairs mic + system channels into interleaved PCM
  ring.py           bounded frame queue (drop-oldest on overflow)
  bus.py            WebSocket broadcast hub
  db.py             aiosqlite + numbered SQL migrations
  migrations/       0001_init.sql …
  secrets.py        Deepgram + Anthropic keys in the macOS Keychain (keyring)
  notegen.py        Claude note generation (expand notes, map-reduce, templates)
  kb/               knowledge base: extract, chunk, embed, ingest
  copilot/          live copilot: triggers, suppression, RRF, Haiku extraction, engine
  asr/base.py       AsrProvider ABC + TranscriptEvent (provider-agnostic)
  asr/replay.py     replay buffer: lossless reconnects
  asr/deepgram.py   nova-3 streaming over raw WebSocket (no SDK)
  static/           vanilla HTML/JS/CSS frontend (no build step)
tests/              pytest + pytest-asyncio
```

## Meeting copilot (knowledge base)

While a meeting records, the copilot watches the transcript, retrieves relevant
chunks from a local knowledge base, and shows context cards in a sidebar with
one-click **Insert into notes** (inserted `> [KB] …` lines keep their source
attribution in the generated note).

### Ingesting documents

```sh
uv run meetnotes ingest ~/Documents/product-docs        # files or directories
uv run meetnotes ingest pricing.pdf faq.md              # .pdf .docx .md .txt .html
uv run meetnotes ingest --meetings                      # past meeting notes/transcripts
```

Or from **Settings → Knowledge base** (path field + Ingest button, per-doc
remove/re-ingest). Re-ingesting a changed file replaces its chunks (sha256
change detection); unchanged files are skipped.

Embeddings are computed locally with `sentence-transformers`
(`all-MiniLM-L6-v2`) — **the first run downloads the model (~90 MB)**, after
which everything works offline at $0. Vectors live in the app's SQLite DB via
`sqlite-vec`, alongside an FTS5 keyword index (retrieval is hybrid:
vector + BM25, fused with reciprocal rank fusion). If sqlite-vec can't load,
the KB degrades to keyword-only search.

### How the copilot triggers

- `periodic` — every 30 s, if at least 3 new finalized segments arrived
- `question` — a finalized segment shaped like a question (ends with `?`)
- `entity` — a segment mentions a KB doc title or heading term
- `hotkey` — the **Help me here** button or ⌘K; bypasses all suppression

Automatic cards are suppressed by: a relevance floor (cosine < 0.35 by
default, adjustable in Settings), per-meeting chunk dedupe, one-card-per-doc
collapse, a 120 s rate cap, and dismissal memory (dismissing a card silences
its doc for the rest of the meeting). Query extraction uses one small
`claude-haiku-4-5` call per cycle (2 s timeout, falls back to the raw
question text — and to pure local retrieval if no Anthropic key is set).

Tuning constants: window/tick/top-k in `src/meetnotes/copilot/engine.py`
(`WINDOW_S`, `TICK_S`, `TOP_CHUNKS`), trigger cadence in
`copilot/triggers.py` (`PERIODIC_INTERVAL_S`, `PERIODIC_MIN_SEGMENTS`),
suppression in `copilot/suppression.py` (`RELEVANCE_FLOOR_DEFAULT`,
`AUTO_RATE_CAP_S`); the relevance floor is also settable per-install in
Settings. The copilot is fully optional: toggle in Settings — default ON only
once at least one doc is ingested, and it makes **zero** API calls when OFF
or when the KB is empty.

## Development

```sh
uv run pytest        # unit + integration tests (mock WS server, fake ASR e2e)
uv run ruff check .
uv run ruff format .
uv run mypy          # strict on src/meetnotes
```

## Note generation (Claude)

Add an **Anthropic API key** in Settings (stored in the Keychain, like the
Deepgram key — get one at [console.anthropic.com](https://console.anthropic.com)).

- During a meeting, jot rough notes in the notepad next to the live transcript.
  They autosave every 2 seconds with per-line timestamps.
- When you end the meeting, MeetNotes calls Claude (`claude-sonnet-4-6`) with
  your notes, the diarized transcript, and a template. The prompt **expands your
  notes using the transcript as ground truth** — your notes decide what matters;
  the transcript supplies the specifics — rather than generically summarizing.
- The result is a markdown note plus structured action items
  (`task / owner / due / source timestamp`) and decisions, shown on the meeting
  page with an editor, a template picker + **Regenerate** button, and
  **Export .md**.
- Transcripts over ~150K tokens are chunked and map-reduced before the final
  note is generated.

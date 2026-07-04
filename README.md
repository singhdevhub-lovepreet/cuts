# cuts

`cuts` is a deterministic, EDL-centric Phase 0 short-video editor.

It ingests raw phone/camera clips, analyzes them through a node graph, emits a JSON Edit Decision List (EDL), and separately renders that EDL with ffmpeg.

Phase 0 is **deterministic only**:
- no LLMs
- no VLMs
- no network calls
- no randomness

Later phases can add an AI editorial brain, but the EDL remains the contract.

## Install

System dependency:
- `ffmpeg` (includes `ffprobe`)

Python:

```bash
pip install -e .[dev]
```

Optional analysis/render extras are declared in the code as lazy imports so the CLI help and tests remain usable without them installed.

## Usage

Analyze clips into an EDL:

```bash
cuts analyze clip1.mp4 clip2.mov --output edl.json
```

Render an EDL into a final MP4:

```bash
cuts render edl.json --output final.mp4
```

End-to-end:

```bash
cuts run clip1.mp4 clip2.mov --output final.mp4
```

Optional music and a target duration:

```bash
cuts run clip1.mp4 clip2.mov   --music music.mp3   --target-duration 45   --output final.mp4
```

## Architecture

```text
raw clips
   │
   ▼
[ ingest ] ──► [ shots ] ──► [ motion ] ──► [ silence ] ──► [ assemble ] ──► EDL JSON
      │              │              │             │
      ├──────────────┼──────────────┼─────────────┤
      ▼              ▼              ▼             ▼
  clip metadata   scene cuts     waste scores   speech regions

optional branches:
- [ transcribe ] → word timestamps → captions in the EDL
- [ beats ]      → beat grid       → beat-snapped cuts

EDL JSON
   │
   ▼
[ render ] ──► ffmpeg ──► final MP4
```

## Notes

- The pipeline is deterministic by design; the same inputs produce the same EDL and render command.
- The renderer currently center-crops to 9:16 and leaves a hook for subject-aware reframing later.
- AI-driven editorial phases can plug in as additional nodes without changing the EDL contract.

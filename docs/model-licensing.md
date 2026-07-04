# ML Model & Library Licensing Assessment

**Question:** Is it safe to use ML models like **SAM 2** and others in this product (a commercial video editor)?

**Short answer:** **Yes for SAM 2** (Apache 2.0) and for the whole core Phase 0 stack. But some models/tools people reach for in this space are **non-commercial** or **conditionally licensed** — those are called out below so we don't accidentally build a commercial feature on a research-only weight.

> Disclaimer: This is engineering due diligence, not legal advice. Licenses change and can differ between *code* and *model weights*. Verify the exact license/version before shipping a commercial release, and keep a `THIRD_PARTY_LICENSES` file.

---

## TL;DR — SAM 2
**Meta Segment Anything 2 (SAM 2)** — code **and** released weights are **Apache 2.0**. ✅ **Safe for commercial use** (permissive, patent grant included). Great fit for our subject-tracking / auto-reframe / masking / object-aware effects needs. (The original SAM was also Apache 2.0.)

---

## ✅ Safe for commercial use (permissive: MIT / BSD / Apache / ISC)
These cover essentially all of our **Phase 0** pipeline and most of what we need going forward.

| Tool / Model | Role | License |
|---|---|---|
| **SAM 2** (Meta) | Segmentation, subject tracking, auto-reframe, masking | Apache 2.0 |
| **Whisper** (OpenAI) | Speech-to-text | MIT |
| **faster-whisper** + CTranslate2 | Fast Whisper inference | MIT |
| **Silero VAD** | Voice activity / silence detection | MIT |
| **PySceneDetect** | Shot/scene detection | BSD-3-Clause |
| **TransNetV2** | Shot-boundary detection | MIT |
| **OpenCV** | Optical flow, blur/sharpness, CV utils | Apache 2.0 |
| **MediaPipe** (Google) | Face/pose/hand detection | Apache 2.0 |
| **librosa** | Beat/tempo/audio analysis | ISC |
| **CLIP** (OpenAI) | Image/text embeddings | MIT |
| **SigLIP** (Google) | Image/text embeddings | Apache 2.0 |
| **LAION aesthetic predictor** | Frame aesthetic scoring | MIT |
| **PyAV**, **ffprobe** (probing) | Media I/O / metadata | BSD / LGPL |
| **pydantic, typer, pytest, ruff, mypy** | Tooling | MIT/permissive |

For the **"vibe/editorial brain"** (Phase 1), the safest open weights are **SigLIP** (Apache 2.0) for embeddings and, if self-hosting a VLM, **Qwen2.5-VL-7B** (Apache 2.0) — but see the caution note on other Qwen sizes.

---

## ⚠️ Conditional — usable commercially but READ THE TERMS
| Tool / Model | Issue | What to do |
|---|---|---|
| **ffmpeg with libx264 / libx265** | The H.264/H.265 encoders are **GPL**; an ffmpeg build including them is GPL. **Also**: H.264/H.265 are **patent-encumbered** (MPEG-LA / Access Advance pools). | We invoke ffmpeg as a **separate process** (not linking its code into ours), which avoids the GPL *code* obligation. But **codec patent royalties** can still apply at scale for H.264/H.265 distribution. Consider **AV1/VP9** (royalty-free) for output, or budget for codec licensing. |
| **Remotion** (motion-graphics renderer) | **Not fully free.** Free for individuals & small teams; **companies above a size threshold need a paid company license.** | Fine to prototype. If we commercialize with a company, buy the Remotion company license, or use an ffmpeg/`gl-transitions`-only render path (permissive) to avoid the dependency. |
| **Qwen2.5-VL (some sizes)** | 7B is Apache 2.0; **other sizes (e.g. 3B, 72B) ship under the Qwen license**, which has use restrictions/attribution and thresholds. | Prefer the **7B (Apache 2.0)** for commercial, or read the specific size's license. |
| **ElevenLabs SFX / TTS, and hosted APIs (Gemini, GPT-4o, Runway, Kling, Veo, Sora)** | Commercial use is governed by each provider's **paid API/TOS**, not an open-source license. Usually fine on a paid plan, but check output-ownership & content terms. | Use paid tiers; review TOS for commercial redistribution & who owns generated media. |
| **madmom** (beat tracking) | BSD code, but some algorithms are **patented / flagged for non-commercial** in the license notes. | We already default to **librosa (ISC)** — keep it as the primary. Avoid madmom in commercial builds unless cleared. |

---

## ⛔ Avoid for commercial (research / non-commercial weights)
Don't build shipping features on these weights without a separate commercial agreement.

| Model | Problem |
|---|---|
| **Stable Audio (open weights)** | Stability's open weights are **non-commercial** without a commercial membership/agreement. Use a licensed audio library or ElevenLabs SFX instead. |
| **Meta AudioGen / MusicGen (AudioCraft)** | Code is MIT but **model weights are CC-BY-NC (non-commercial)**. Don't use the weights in a paid product. |
| **InsightFace / RetinaFace pretrained models** | Popular face models, but many pretrained weights are **non-commercial / research-only**. Use **MediaPipe (Apache 2.0)** for face detection instead. |
| **VideoLLaMA / many academic video-LLMs** | Often research-only or built on non-commercial components. Prefer **Qwen2.5-VL-7B** or hosted APIs. |
| **NIMA and various "aesthetic" research checkpoints** | Frequently research-only. Use the **LAION aesthetic predictor (MIT)** or a CLIP-based scorer. |

---

## Recommendation
- **SAM 2 is a green light** — Apache 2.0, use it freely for subject tracking / auto-reframe / masking / object-aware effects.
- Our entire **Phase 0** stack is permissively licensed and commercial-safe.
- The only real watch-items for a commercial launch are: **(1)** video **codec patents** (H.264/H.265) for output at scale — consider AV1/VP9; **(2)** **Remotion**'s company license if we adopt it; **(3)** avoid **non-commercial audio/face weights** (AudioGen, Stable Audio, InsightFace) — permissive alternatives exist for all of them.
- Maintain a `THIRD_PARTY_LICENSES` manifest and re-check before any commercial release.

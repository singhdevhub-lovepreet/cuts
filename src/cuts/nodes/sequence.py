from __future__ import annotations

from cuts.graph import Context, Node
from cuts.vlm.client import VLMClient
from cuts.vlm.gemini import build_gemini_client
from cuts.vlm.mock import MockVLMClient
from cuts.vlm.models import VibeIntent


class SequencerNode(Node):
    name = "sequencer"
    requires = ("shot_tags",)
    provides = ("sequence_plan",)

    def __init__(self, client: VLMClient | None = None) -> None:
        self.client = client

    def run(self, context: Context) -> Context:
        intent = VibeIntent(
            vibe_prompt=context.vibe_prompt,
            platform=context.platform,
            target_duration=context.target_duration,
        )
        client = self.client or self._default_client()
        if client is None:
            context.sequence_plan = None
            context.warnings.append("VLM sequencing unavailable; using phase-0 path")
            return context
        shot_tags = context.shot_tags
        if not shot_tags:
            context.sequence_plan = None
            return context
        context.sequence_plan = client.sequence(shot_tags, intent)
        return context

    def _default_client(self) -> VLMClient | None:
        client = build_gemini_client()
        return client if client is not None else MockVLMClient()

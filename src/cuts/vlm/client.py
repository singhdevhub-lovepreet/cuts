from __future__ import annotations

from typing import Protocol

from cuts.vlm.models import SequencePlan, ShotObservation, ShotTags, VibeIntent


class VLMClient(Protocol):
    model_name: str

    def describe_shots(self, shots: list[ShotObservation], intent: VibeIntent) -> list[ShotTags]:
        raise NotImplementedError

    def sequence(self, tags: list[ShotTags], intent: VibeIntent) -> SequencePlan:
        raise NotImplementedError

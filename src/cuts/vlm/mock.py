from __future__ import annotations

import hashlib

from cuts.vlm.client import VLMClient
from cuts.vlm.models import SequencePlan, SequencePlanItem, ShotObservation, ShotTags, VibeIntent


class MockVLMClient(VLMClient):
    def __init__(self, model_name: str = "mock-vlm") -> None:
        self.model_name = model_name

    def describe_shots(self, shots: list[ShotObservation], intent: VibeIntent) -> list[ShotTags]:
        return [self._describe_shot(shot, intent) for shot in shots]

    def sequence(self, tags: list[ShotTags], intent: VibeIntent) -> SequencePlan:
        scored = [tag for tag in tags if tag.role.lower() not in {"waste", "unknown"}]
        scored.sort(
            key=lambda tag: (
                self._role_priority(tag.role),
                -tag.energy,
                tag.shot_index,
                tag.clip_id,
                tag.shot_start,
            )
        )
        return SequencePlan(
            rationale=f"Mock sequence for {intent.platform.value}: {intent.vibe_prompt}".strip(),
            ordered_shots=[
                SequencePlanItem(
                    shot_index=tag.shot_index,
                    clip_id=tag.clip_id,
                    shot_start=tag.shot_start,
                    shot_end=tag.shot_end,
                    keep=True,
                    trim_in=tag.shot_start,
                    trim_out=tag.shot_end,
                    rationale=f"mock keeps {tag.role} shot {tag.shot_index}",
                )
                for tag in scored
            ],
        )

    def _describe_shot(self, shot: ShotObservation, intent: VibeIntent) -> ShotTags:
        frame_seed = self._hash_text(
            f"{shot.clip_id}|{shot.shot_index}|{shot.shot_start:.3f}|{shot.shot_end:.3f}|{len(shot.frames)}|{intent.platform.value}|{intent.vibe_prompt}"
        )
        subjects = ["person", "speaker", "landscape", "street", "details"]
        actions = ["talking", "moving", "pausing", "reacting", "showing"]
        shot_types = ["close-up", "medium", "wide", "establishing", "detail"]
        settings = ["indoors", "outdoors", "office", "street", "studio"]
        moods = ["energetic", "warm", "calm", "bright", "moody"]
        subject = subjects[frame_seed % len(subjects)]
        action = actions[(frame_seed // 3) % len(actions)]
        shot_type = shot_types[(frame_seed // 5) % len(shot_types)]
        setting = settings[(frame_seed // 7) % len(settings)]
        energy = ((frame_seed % 1000) / 1000.0) * 0.5 + 0.25
        role = self._role_from_hash(frame_seed)
        caption = f"{subject} {action} in {setting}"
        mood_tags = [moods[(frame_seed // 11) % len(moods)], intent.platform.value]
        if intent.vibe_prompt:
            mood_tags.append(intent.vibe_prompt.split()[0].lower())
        return ShotTags(
            shot_index=shot.shot_index,
            clip_id=shot.clip_id,
            shot_start=shot.shot_start,
            shot_end=shot.shot_end,
            subject=subject,
            action=action,
            shot_type=shot_type,
            setting=setting,
            energy=min(1.0, energy),
            mood_tags=sorted(dict.fromkeys(mood_tags)),
            role=role,
            caption=caption,
        )

    def _role_from_hash(self, seed: int) -> str:
        if seed % 11 == 0:
            return "waste"
        if seed % 5 == 0:
            return "connective"
        if seed % 3 == 0:
            return "broll"
        return "hero"

    def _role_priority(self, role: str) -> int:
        priorities = {"hero": 0, "connective": 1, "broll": 2, "waste": 3, "unknown": 4}
        return priorities.get(role.lower(), 4)

    def _hash_text(self, text: str) -> int:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        return int(digest[:12], 16)

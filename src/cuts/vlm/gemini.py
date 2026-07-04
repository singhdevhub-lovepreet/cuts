from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from cuts.vlm.client import VLMClient
from cuts.vlm.models import SequencePlan, SequencePlanItem, ShotObservation, ShotTags, VibeIntent

TModel = TypeVar("TModel", bound=BaseModel)


class GeminiShotTagsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shot_tags: list[ShotTags] = Field(default_factory=list)


class GeminiSequencePlanResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rationale: str
    ordered_shots: list[SequencePlanItem] = Field(default_factory=list)


@dataclass(slots=True)
class GeminiVLMClient(VLMClient):
    model_name: str = "gemini-2.5-flash"
    api_key: str | None = None

    def describe_shots(self, shots: list[ShotObservation], intent: VibeIntent) -> list[ShotTags]:
        response = self._generate(
            self._describe_prompt(shots, intent),
            response_schema=GeminiShotTagsResponse,
        )
        return response.shot_tags

    def sequence(self, tags: list[ShotTags], intent: VibeIntent) -> SequencePlan:
        response = self._generate(
            self._sequence_prompt(tags, intent),
            response_schema=GeminiSequencePlanResponse,
        )
        return SequencePlan(rationale=response.rationale, ordered_shots=response.ordered_shots)

    def _generate(self, prompt: list[Any], response_schema: type[TModel]) -> TModel:
        client = self._client()
        config = self._config(response_schema)
        result = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )
        text = getattr(result, "text", None)
        if isinstance(text, str) and text.strip():
            return self._parse_model_json(text, response_schema)
        parts = getattr(result, "candidates", [])
        if parts:
            content = parts[0].content
            text_parts = getattr(content, "parts", [])
            collected = "".join(getattr(part, "text", "") for part in text_parts)
            if collected.strip():
                return self._parse_model_json(collected, response_schema)
        raise RuntimeError("Gemini response did not contain text")

    def _client(self) -> Any:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("google-genai is not installed") from exc
        key = self.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Gemini API key not configured")
        return genai.Client(api_key=key)

    def _config(self, response_schema: type[BaseModel]) -> Any:
        try:
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("google-genai is not installed") from exc
        schema_json = (
            {
                "type": "object",
                "properties": {
                    "shot_tags": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "shot_index": {"type": "integer"},
                                "clip_id": {"type": "string"},
                                "shot_start": {"type": "number"},
                                "shot_end": {"type": "number"},
                                "subject": {"type": "string"},
                                "action": {"type": "string"},
                                "shot_type": {"type": "string"},
                                "setting": {"type": "string"},
                                "energy": {"type": "number"},
                                "mood_tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "role": {"type": "string"},
                                "caption": {"type": "string"},
                            },
                            "required": [
                                "shot_index",
                                "clip_id",
                                "shot_start",
                                "shot_end",
                                "subject",
                                "action",
                                "shot_type",
                                "setting",
                                "energy",
                                "mood_tags",
                                "role",
                                "caption",
                            ],
                        },
                    }
                },
                "required": ["shot_tags"],
            }
            if response_schema is GeminiShotTagsResponse
            else {
                "type": "object",
                "properties": {
                    "rationale": {"type": "string"},
                    "ordered_shots": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "shot_index": {"type": "integer"},
                                "clip_id": {"type": "string"},
                                "shot_start": {"type": "number"},
                                "shot_end": {"type": "number"},
                                "keep": {"type": "boolean"},
                                "trim_in": {"type": "number"},
                                "trim_out": {"type": "number"},
                                "rationale": {"type": "string"},
                            },
                            "required": [
                                "shot_index",
                                "clip_id",
                                "shot_start",
                                "shot_end",
                                "keep",
                                "rationale",
                            ],
                        },
                    },
                },
                "required": ["rationale", "ordered_shots"],
            }
        )
        return types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=schema_json,
        )

    def _describe_prompt(self, shots: list[ShotObservation], intent: VibeIntent) -> list[Any]:
        try:
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("google-genai is not installed") from exc
        contents: list[Any] = [
            "You are a short-video editorial tagging assistant. "
            "Analyze the provided shot frames and return JSON matching the schema."
        ]
        contents.append(f"Intent: {intent.model_dump_json()}")
        for shot in shots:
            contents.append(
                "Shot "
                f"{shot.shot_index} clip={shot.clip_id} "
                f"start={shot.shot_start} end={shot.shot_end}"
            )
            for frame in shot.frames:
                contents.append(
                    types.Part.from_bytes(
                        data=frame.frame_path.read_bytes(),
                        mime_type="image/jpeg",
                    )
                )
        return contents

    def _sequence_prompt(self, tags: list[ShotTags], intent: VibeIntent) -> list[Any]:
        payload = {
            "intent": intent.model_dump(mode="json"),
            "shot_tags": [tag.model_dump(mode="json") for tag in tags],
        }
        return [
            (
                "You are a short-video sequencing assistant. Return JSON with "
                "rationale and ordered_shots; drop waste shots."
            ),
            json.dumps(payload, separators=(",", ":")),
        ]

    def _parse_model_json(self, text: str, response_schema: type[TModel]) -> TModel:
        stripped = text.strip()
        try:
            return response_schema.model_validate_json(stripped)
        except Exception:
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start >= 0 and end > start:
                return response_schema.model_validate_json(stripped[start : end + 1])
            raise


def build_gemini_client() -> GeminiVLMClient | None:
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return GeminiVLMClient(model_name=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    return None

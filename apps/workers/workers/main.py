"""arq worker entrypoint.

Task stubs for the pipeline; each becomes a real implementation in its own
milestone:
  - parse_resume: resume file -> structured JSON -> persona draft
  - enrich_persona: voice-session transcript -> persona delta
  - scrape_jobs: per-source job board fetch + normalize + upsert
  - match_user: rank jobs against a persona
  - auto_apply: generate cover letter/resume variant and submit via ATS adapter
"""

import os
from typing import Any

from arq.connections import RedisSettings


async def parse_resume(ctx: dict[str, Any], resume_id: str) -> str:
    return f"parsed:{resume_id}"


async def enrich_persona(ctx: dict[str, Any], voice_session_id: str) -> str:
    return f"enriched:{voice_session_id}"


async def scrape_jobs(ctx: dict[str, Any], source: str) -> str:
    return f"scraped:{source}"


async def match_user(ctx: dict[str, Any], user_id: str) -> str:
    return f"matched:{user_id}"


async def auto_apply(ctx: dict[str, Any], application_id: str) -> str:
    return f"applied:{application_id}"


class WorkerSettings:
    functions = [parse_resume, enrich_persona, scrape_jobs, match_user, auto_apply]
    redis_settings = RedisSettings.from_dsn(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))

# JobPilot

Async AI agent that applies to jobs for US candidates: upload a resume, refine
your persona (including a voice interview), get a ranked job feed, and — on the
Pro plan — let the agent auto-apply with a tailored cover letter and resume.

## Monorepo layout

```
apps/web       Next.js 16 frontend (Apple-style light-first design)
apps/api       FastAPI backend (REST /api/v1, SSE progress)
apps/workers   arq background workers (parse, enrich, scrape, match, auto-apply)
packages/shared  Cross-app contracts (OpenAPI-generated TS client)
```

## Local development

```bash
docker compose up -d          # Postgres (pgvector) + Redis

# API
cd apps/api
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
uvicorn app.main:app --reload --port 8000

# Workers
cd apps/workers
pip install -e '.[dev]'
arq workers.main.WorkerSettings

# Web
cd apps/web
npm install
npm run dev                   # http://localhost:3000
```

Copy `apps/api/.env.example` to `apps/api/.env`. Provider keys are dummy
values in development.

## Checks

- API/workers: `ruff check . && ruff format --check .`, `mypy .`, `pytest -q`
- Web: `npm run lint`, `npx tsc --noEmit`, `npm run build`

## Roadmap (milestones)

1. Scaffold (this) — monorepo, design tokens, CI, compose
2. Google auth + resume upload + parse -> persona draft
3. Persona editor + persona events
4. Job sources (Greenhouse/Lever/Ashby) + matcher + feed UI
5. CRM dashboard (kanban, application detail, interviews)
6. Voice enrichment (Sarvam STT/TTS interviewer)
7. Auto-apply agent (Greenhouse/Lever adapters behind a reusable ATS interface)
8. Billing (Dodo Payments) + plan gating + landing/pricing

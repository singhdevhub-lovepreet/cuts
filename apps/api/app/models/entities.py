import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Enum, Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, uuid_pk


class Plan(enum.StrEnum):
    FREE = "free"
    PRO = "pro"


class PersonaEventSource(enum.StrEnum):
    RESUME = "resume"
    VOICE = "voice"
    MANUAL = "manual"


class MatchState(enum.StrEnum):
    SUGGESTED = "suggested"
    SAVED = "saved"
    DISMISSED = "dismissed"


class ApplicationMode(enum.StrEnum):
    MANUAL = "manual"
    AUTO = "auto"


class ApplicationStatus(enum.StrEnum):
    DRAFT = "draft"
    NEEDS_USER = "needs_user"
    SUBMITTED = "submitted"
    SCREENED = "screened"
    INTERVIEW = "interview"
    OFFER = "offer"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = uuid_pk()
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    plan: Mapped[Plan] = mapped_column(Enum(Plan), default=Plan.FREE)
    google_sub: Mapped[str | None] = mapped_column(String(255), unique=True)
    dodo_customer_id: Mapped[str | None] = mapped_column(String(255))
    # Auto-apply guardrails: review-before-send (default) vs auto-approve,
    # max applications/day, salary floor, excluded companies.
    guardrails: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Resume(TimestampMixin, Base):
    __tablename__ = "resumes"

    id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    storage_key: Mapped[str] = mapped_column(String(500))
    parsed_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    version: Mapped[int] = mapped_column(default=1)
    is_active: Mapped[bool] = mapped_column(default=True)


class Persona(TimestampMixin, Base):
    """Current persona state — exactly one row per user (latest merged view)."""

    __tablename__ = "personas"

    id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), unique=True, index=True)
    summary: Mapped[str | None] = mapped_column(Text)
    titles: Mapped[list[str]] = mapped_column(JSON, default=list)
    skills: Mapped[list[str]] = mapped_column(JSON, default=list)
    seniority: Mapped[str | None] = mapped_column(String(50))
    locations: Mapped[list[str]] = mapped_column(JSON, default=list)
    remote_pref: Mapped[str | None] = mapped_column(String(20))
    salary_min: Mapped[int | None] = mapped_column()
    work_auth: Mapped[str | None] = mapped_column(String(100))
    deal_breakers: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class PersonaEvent(TimestampMixin, Base):
    """Append-only audit log of persona changes; `delta` is the JSON patch applied."""

    __tablename__ = "persona_events"

    id: Mapped[uuid.UUID] = uuid_pk()
    persona_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("personas.id"), index=True)
    source: Mapped[PersonaEventSource] = mapped_column(Enum(PersonaEventSource))
    delta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    transcript_ref: Mapped[str | None] = mapped_column(String(500))


class VoiceSession(TimestampMixin, Base):
    __tablename__ = "voice_sessions"

    id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    transcript: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    audio_storage_key: Mapped[str | None] = mapped_column(String(500))
    distilled_delta: Mapped[dict[str, Any] | None] = mapped_column(JSON)


class Job(TimestampMixin, Base):
    __tablename__ = "jobs"
    __table_args__ = (UniqueConstraint("source", "external_id"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    source: Mapped[str] = mapped_column(String(50), index=True)
    external_id: Mapped[str] = mapped_column(String(255))
    company: Mapped[str] = mapped_column(String(255), index=True)
    title: Mapped[str] = mapped_column(String(255))
    location: Mapped[str | None] = mapped_column(String(255))
    remote: Mapped[bool | None] = mapped_column()
    salary_min: Mapped[int | None] = mapped_column()
    salary_max: Mapped[int | None] = mapped_column()
    description: Mapped[str | None] = mapped_column(Text)
    apply_url: Mapped[str] = mapped_column(String(1000))
    ats: Mapped[str | None] = mapped_column(String(50))
    posted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    raw: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Match(TimestampMixin, Base):
    __tablename__ = "matches"
    __table_args__ = (UniqueConstraint("user_id", "job_id"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id"), index=True)
    score: Mapped[float] = mapped_column(Float)
    reasons: Mapped[list[str]] = mapped_column(JSON, default=list)
    state: Mapped[MatchState] = mapped_column(Enum(MatchState), default=MatchState.SUGGESTED)


class Application(TimestampMixin, Base):
    __tablename__ = "applications"

    id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id"), index=True)
    mode: Mapped[ApplicationMode] = mapped_column(Enum(ApplicationMode))
    status: Mapped[ApplicationStatus] = mapped_column(
        Enum(ApplicationStatus), default=ApplicationStatus.DRAFT, index=True
    )
    cover_letter_key: Mapped[str | None] = mapped_column(String(500))
    resume_variant_key: Mapped[str | None] = mapped_column(String(500))
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    answers: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    agent_log: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)


class Interview(TimestampMixin, Base):
    """Interview schedule metadata (date/type/notes). MVP tracks these manually;
    Google Calendar / Gmail-driven auto-updates are a later phase."""

    __tablename__ = "interviews"

    id: Mapped[uuid.UUID] = uuid_pk()
    application_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("applications.id"), index=True)
    kind: Mapped[str] = mapped_column(String(50))
    scheduled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    notes: Mapped[str | None] = mapped_column(Text)


class AgentRun(TimestampMixin, Base):
    __tablename__ = "agent_runs"

    id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    kind: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(20), default="queued")
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

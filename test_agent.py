"""
AI Appointment Agent — conversational booking API.

- Input: JSON list of messages (user ↔ assistant).
- Goal: collect first name, last name, email, booking date, meeting start/end time (booking only).
- Off-topic: redirect; goal is only to book a meeting.
- When complete: check `ai_appointment_agent` for time conflicts; suggest alternatives until resolved.
- When no conflict: save the booking to `ai_appointment_agent` and return gathered information (`complete: true`, `booking_id`).

The same table is used for conflict detection (overlap query) and for storing successful bookings.

Run: uvicorn test_agent:app --reload
Requires: DEEPSEEK_API_KEY, MYSQL_* in .env
"""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from mysql.connector import Error as MySQLError
import mysql.connector
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()
logger = logging.getLogger(__name__)

CREATE_AI_APPOINTMENT_AGENT_TABLE = """
CREATE TABLE IF NOT EXISTS ai_appointment_agent (
    id                   BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    first_name           VARCHAR(100)  NOT NULL,
    last_name            VARCHAR(100)  NOT NULL,
    email                VARCHAR(255)  NULL,
    phone_number         VARCHAR(50)   NULL,
    appointment_date     DATE          NOT NULL,
    start_time           TIME          NOT NULL,
    end_time             TIME          NOT NULL,
    duration_minutes     INT UNSIGNED  NULL,
    status               VARCHAR(32)   NOT NULL DEFAULT 'scheduled',
    source               VARCHAR(64)   NULL DEFAULT 'api',
    created_at           DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at           DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_date_status (appointment_date, status),
    INDEX idx_overlap (appointment_date, start_time, end_time)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
"""


def get_deepseek_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="DEEPSEEK_API_KEY is not set in .env.",
        )
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def get_db_connection():
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    user = os.getenv("MYSQL_USER")
    password = os.getenv("MYSQL_PASSWORD")
    database = os.getenv("MYSQL_DATABASE")
    if not all([user, password, database]):
        raise HTTPException(
            status_code=500,
            detail="Set MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE in .env.",
        )
    return mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        charset="latin1",
    )


def _ensure_table_once(conn) -> None:
    cursor = conn.cursor()
    try:
        for stmt in CREATE_AI_APPOINTMENT_AGENT_TABLE.split(";"):
            s = stmt.strip()
            if s:
                cursor.execute(s)
        conn.commit()
    finally:
        cursor.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        conn = get_db_connection()
        try:
            _ensure_table_once(conn)
            logger.info("ai_appointment_agent table ensured (startup).")
        finally:
            conn.close()
    except Exception as e:
        logger.warning("Startup: could not ensure table: %s", e)
    yield


app = FastAPI(
    title="AI Appointment Agent",
    description="Conversational meeting booking: messages in, assistant reply + conflict check out.",
    version="3.0.0",
    lifespan=lifespan,
)


BOOKING_AGENT_SYSTEM_PROMPT = """You are an AI assistant for ONE task only: helping the user book a meeting.

SCOPE (strict):
- Only discuss meeting booking: collect first name, last name, email, booking date, and meeting time (start and end time, or start time and you infer a reasonable end time such as one hour later if the user gives only a start time).
- If the user asks about anything else (weather, jokes, other topics), politely refuse and say your only purpose is to help them book a meeting. Do not answer off-topic questions. Set `is_off_topic_attempt` to true when they try.

LANGUAGE: The user may write in English or Arabic (or mix them). Understand and extract booking details in either language. Reply in the same language the user is primarily using (Arabic if they write in Arabic, English if they write in English).

CONVERSATION:
- Be brief and friendly. Ask for missing fields one or a few at a time.
- If the user previously had a time conflict and the system suggested alternative times, help them confirm or choose until they agree to a valid slot.

OUTPUT: You MUST respond with a single JSON object and nothing else (no markdown fences):

{
  "assistant_message": "string — what you say next to the user",
  "is_off_topic_attempt": true or false,
  "gathered": {
    "first_name": null or string,
    "last_name": null or string,
    "email": null or string,
    "date": null or string,
    "start_time": null or string,
    "end_time": null or string
  },
  "info_complete": true or false
}

Rules for `gathered`:
- Use ISO date YYYY-MM-DD for `date` when possible.
- Use 24h times like "14:00" or "14:00:00" for start_time and end_time.
- Set `info_complete` to true ONLY when all six fields are clearly present in the conversation (first_name, last_name, email, date, start_time, end_time). If the user only gave start time, set end_time to a reasonable end (e.g. one hour after start) and then you may set info_complete true when everything else is known.

If `is_off_topic_attempt` is true, still set assistant_message to redirect; gathered may stay partial."""


def _extract_json_from_response(content: str) -> dict | None:
    content = (content or "").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    start = content.find("{")
    if start != -1:
        depth = 0
        for i, c in enumerate(content[start:], start=start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(content[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def _normalize_date_str(d: str) -> str | None:
    if not d or not isinstance(d, str):
        return None
    d = d.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", d):
        return d
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(d[:10], fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _normalize_time_str(t: str) -> str | None:
    if not t or not isinstance(t, str):
        return None
    t = t.strip()
    parts = re.split(r"[:.]", t)
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        h, m = int(parts[0]), int(parts[1])
        sec = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
        if 0 <= h <= 23 and 0 <= m <= 59 and 0 <= sec <= 59:
            return f"{h:02d}:{m:02d}:{sec:02d}"
    return None


def _time_to_timedelta(t) -> timedelta:
    if isinstance(t, timedelta):
        return t
    if hasattr(t, "hour"):
        return timedelta(hours=t.hour, minutes=t.minute, seconds=getattr(t, "second", 0))
    s = str(t)
    parts = s.split(":")
    h, m = int(parts[0]), int(parts[1])
    sec = int(parts[2]) if len(parts) > 2 else 0
    return timedelta(hours=h, minutes=m, seconds=sec)


def _timedelta_to_time_str(td: timedelta) -> str:
    total = int(td.total_seconds()) % 86400
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _duration_minutes(start_s: str, end_s: str) -> int:
    a = _time_to_timedelta(_normalize_time_str(start_s) or start_s)
    b = _time_to_timedelta(_normalize_time_str(end_s) or end_s)
    return max(0, int((b - a).total_seconds() // 60))


def _suggest_after_conflict(
    conflicting_max_end: timedelta,
    requested_start_td: timedelta,
    requested_end_td: timedelta,
) -> tuple[str, str]:
    duration = requested_end_td - requested_start_td
    if duration <= timedelta(0):
        duration = timedelta(minutes=60)
    new_start = conflicting_max_end
    new_end = new_start + duration
    return _timedelta_to_time_str(new_start), _timedelta_to_time_str(new_end)


def _find_overlapping_appointments(conn, appointment_date: str, start_norm: str, end_norm: str):
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT id, start_time, end_time
            FROM ai_appointment_agent
            WHERE appointment_date = %s
              AND status <> 'cancelled'
              AND start_time < %s
              AND end_time > %s
            """,
            (appointment_date, end_norm, start_norm),
        )
        return cursor.fetchall()
    finally:
        cursor.close()


def _max_end_among(rows) -> timedelta | None:
    if not rows:
        return None
    ends = [_time_to_timedelta(r[2]) for r in rows]
    return max(ends)


# --- Models ---

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1)


class ConversationRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)


class GatheredBooking(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    date: str | None = None
    start_time: str | None = None
    end_time: str | None = None


class AgentAppointmentResponse(BaseModel):
    assistant_message: str = Field(..., description="What the bot should show the user next.")
    off_topic: bool = False
    gathered: GatheredBooking = Field(default_factory=GatheredBooking)
    info_complete: bool = False
    conflict: bool | None = None
    suggested_start_time: str | None = None
    suggested_end_time: str | None = None
    complete: bool = Field(
        False,
        description="True when all info is gathered, there is no DB conflict, booking was saved, and details are final.",
    )
    message: str | None = Field(None, description="Status message for the client.")
    booking_id: int | None = Field(None, description="Row id in ai_appointment_agent when saved.")
    saved: bool = Field(False, description="True when a new row was inserted into ai_appointment_agent.")


def _gathered_from_llm(g: dict | None) -> GatheredBooking:
    if not g or not isinstance(g, dict):
        return GatheredBooking()
    return GatheredBooking(
        first_name=g.get("first_name") or None,
        last_name=g.get("last_name") or None,
        email=g.get("email") or None,
        date=g.get("date") or None,
        start_time=g.get("start_time") or None,
        end_time=g.get("end_time") or None,
    )


def _all_fields_non_empty(g: GatheredBooking) -> bool:
    return all(
        [
            g.first_name and str(g.first_name).strip(),
            g.last_name and str(g.last_name).strip(),
            g.email and str(g.email).strip(),
            g.date and str(g.date).strip(),
            g.start_time and str(g.start_time).strip(),
            g.end_time and str(g.end_time).strip(),
        ]
    )


def _insert_booking_row(
    first_name: str,
    last_name: str,
    email: str,
    date_val: str,
    start_norm: str,
    end_norm: str,
) -> int:
    """Insert a successful booking into ai_appointment_agent. Returns new id."""
    duration = _duration_minutes(start_norm, end_norm)
    try:
        conn = get_db_connection()
    except MySQLError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO ai_appointment_agent (
                first_name, last_name, email, phone_number,
                appointment_date, start_time, end_time, duration_minutes,
                status, source
            ) VALUES (%s, %s, %s, NULL, %s, %s, %s, %s, 'scheduled', 'ai_appointment_agent')
            """,
            (
                first_name,
                last_name,
                email,
                date_val,
                start_norm,
                end_norm,
                duration or None,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)
    except MySQLError as e:
        conn.rollback()
        raise HTTPException(status_code=503, detail=f"Could not save booking: {e}") from e
    finally:
        cursor.close()
        conn.close()


def _run_conflict_check(date_val: str, start_norm: str, end_norm: str) -> tuple[bool, list[int], str | None, str | None]:
    """Returns (has_conflict, ids, suggested_start, suggested_end)."""
    try:
        conn = get_db_connection()
    except MySQLError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    try:
        rows = _find_overlapping_appointments(conn, date_val, start_norm, end_norm)
        req_start_td = _time_to_timedelta(start_norm)
        req_end_td = _time_to_timedelta(end_norm)
        if not rows:
            return False, [], None, None
        max_end = _max_end_among(rows)
        if max_end is None:
            max_end = req_end_td
        sug_start, sug_end = _suggest_after_conflict(max_end, req_start_td, req_end_td)
        ids = [int(r[0]) for r in rows]
        return True, ids, sug_start, sug_end
    finally:
        conn.close()


@app.get("/")
def root():
    return {
        "service": "ai_appointment_agent",
        "endpoint": "POST /agent/appointment",
        "input": "JSON body: { \"messages\": [ { \"role\": \"user|assistant\", \"content\": \"...\" } ] }",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/agent/appointment",
    response_model=AgentAppointmentResponse,
    response_model_exclude_none=True,
)
def agent_appointment(body: ConversationRequest):
    """
    Pass the full conversation. The assistant reply, gathered fields, conflict state,
    and `complete=true` with final JSON when the slot is free. Successful bookings are
    inserted into `ai_appointment_agent` (same table used for conflict detection).
    """
    client = get_deepseek_client()
    api_messages = [
        {"role": "system", "content": BOOKING_AGENT_SYSTEM_PROMPT},
        *[{"role": m.role, "content": m.content} for m in body.messages],
    ]

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=api_messages,
            temperature=0.3,
            max_tokens=2048,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"DeepSeek API error: {str(e)}") from e

    raw = (response.choices[0].message.content or "").strip()
    data = _extract_json_from_response(raw)
    if not data:
        raise HTTPException(
            status_code=502,
            detail="Could not parse model response. Try again.",
        )

    assistant_message = data.get("assistant_message") or "How can I help you book a meeting?"
    off_topic = bool(data.get("is_off_topic_attempt"))
    gathered = _gathered_from_llm(data.get("gathered"))

    if off_topic:
        return AgentAppointmentResponse(
            assistant_message=assistant_message,
            off_topic=True,
            gathered=gathered,
            info_complete=_all_fields_non_empty(gathered),
            message="Off-topic; only booking is supported.",
        )

    if not _all_fields_non_empty(gathered):
        return AgentAppointmentResponse(
            assistant_message=assistant_message,
            off_topic=False,
            gathered=gathered,
            info_complete=False,
            message="Collecting booking details.",
        )

    # All fields present — normalize and check DB
    date_val = _normalize_date_str(gathered.date or "") or gathered.date
    start_norm = _normalize_time_str(gathered.start_time or "") or _normalize_time_str(
        str(gathered.start_time)
    )
    end_norm = _normalize_time_str(gathered.end_time or "") or _normalize_time_str(str(gathered.end_time))

    if not date_val or not re.match(r"^\d{4}-\d{2}-\d{2}$", date_val):
        return AgentAppointmentResponse(
            assistant_message=assistant_message or "Please confirm the booking date (YYYY-MM-DD).",
            gathered=gathered,
            info_complete=False,
            message="Date could not be normalized.",
        )
    if not start_norm or not end_norm:
        return AgentAppointmentResponse(
            assistant_message=assistant_message or "Please confirm start and end time for the meeting.",
            gathered=gathered,
            info_complete=False,
            message="Times could not be normalized.",
        )
    if _time_to_timedelta(end_norm) <= _time_to_timedelta(start_norm):
        return AgentAppointmentResponse(
            assistant_message=assistant_message or "End time must be after start time.",
            gathered=gathered,
            info_complete=False,
            message="Invalid time range.",
        )

    has_conflict, _ids, sug_start, sug_end = _run_conflict_check(date_val, start_norm, end_norm)

    if has_conflict and sug_start and sug_end:
        # Build helpful reply if model did not already propose alternatives
        extra = (
            f" That time is already booked. Please consider {sug_start}–{sug_end} instead, "
            "or tell me another time that works for you."
        )
        if "booked" not in assistant_message.lower() and "unavailable" not in assistant_message.lower():
            assistant_message = (assistant_message.rstrip() + extra)[:4000]
        return AgentAppointmentResponse(
            assistant_message=assistant_message,
            gathered=gathered,
            info_complete=True,
            conflict=True,
            suggested_start_time=sug_start,
            suggested_end_time=sug_end,
            complete=False,
            message="Conflict with existing appointment; suggest alternative time.",
        )

    # No conflict — persist to ai_appointment_agent, then return final JSON
    final = GatheredBooking(
        first_name=gathered.first_name.strip() if gathered.first_name else None,
        last_name=gathered.last_name.strip() if gathered.last_name else None,
        email=gathered.email.strip() if gathered.email else None,
        date=date_val,
        start_time=start_norm,
        end_time=end_norm,
    )
    assert final.first_name and final.last_name and final.email
    booking_id = _insert_booking_row(
        final.first_name,
        final.last_name,
        final.email,
        date_val,
        start_norm,
        end_norm,
    )
    return AgentAppointmentResponse(
        assistant_message=assistant_message
        or "Your meeting is available. Here are your booking details.",
        gathered=final,
        info_complete=True,
        conflict=False,
        complete=True,
        message="Booking saved; no conflict.",
        booking_id=booking_id,
        saved=True,
    )


"""
Booking Intent Detection API (final)

Single API that:
1. Detects meeting booking intent from messages (English or Arabic).
2. If no intent → returns no intent message.
3. If intent + incomplete info → returns missing_info_prompt; client sends more messages and calls again.
4. If intent + complete info → checks appointment_bookings for duplicate (same person, same date, overlapping time).
   - If conflict → returns conflict message.
   - If no conflict → inserts booking and returns success with booking details.

Uses: DeepSeek for intent + extraction; MySQL table appointment_bookings in database (e.g. giganeti_gym_management).
"""

import json
import os
import re
from datetime import datetime
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field
import mysql.connector
from mysql.connector import Error as MySQLError

load_dotenv()

app = FastAPI(
    title="Booking Intent Detection API",
    description="Detects meeting booking intent, extracts details, checks for conflicts, and creates bookings in appointment_bookings. Supports English and Arabic.",
    version="1.0.0",
)

# --- Config ---

def get_deepseek_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="DEEPSEEK_API_KEY is not set. Add it to your .env file.",
        )
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def get_db_connection():
    """MySQL connection using .env (MYSQL_*). Database should be the one containing appointment_bookings (e.g. giganeti_gym_management)."""
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    user = os.getenv("MYSQL_USER")
    password = os.getenv("MYSQL_PASSWORD")
    database = os.getenv("MYSQL_DATABASE")
    if not all([user, password, database]):
        raise HTTPException(
            status_code=500,
            detail="MySQL credentials not set. Set MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE in .env.",
        )
    return mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
    )


# --- Request / Response models ---

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1)


class DetectBookingIntentRequest(BaseModel):
    messages: list[ChatMessage] = Field(
        ...,
        min_length=1,
        description="Conversation messages (English or Arabic) to analyze for booking intent and to extract meeting details.",
    )


class ExtractedBookingInfo(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    date: str | None = None
    start_time: str | None = None
    end_time: str | None = None


class DetectBookingIntentResponse(BaseModel):
    has_meeting_intent: bool = Field(
        ...,
        description="True if the conversation shows intent to book a meeting.",
    )
    message: str | None = Field(
        default=None,
        description="No-intent message, missing-info prompt, conflict message, or success message.",
    )
    info_complete: bool | None = Field(
        default=None,
        description="When intent is true: true if all required fields are present; false otherwise.",
    )
    extracted: ExtractedBookingInfo | None = Field(
        default=None,
        description="Extracted first_name, last_name, date, start_time, end_time when intent is detected.",
    )
    booking_conflict: bool | None = Field(
        default=None,
        description="When intent is true and info complete: true if an existing booking exists at the same time.",
    )
    booking_created: bool | None = Field(
        default=None,
        description="When intent is true and info complete and no conflict: true if the booking was inserted.",
    )
    booking_id: int | None = Field(default=None, description="appointment_booking_id of the created row when booking_created is true.")


# --- LLM prompt: intent + extract first_name, last_name, date, start_time, end_time ---

DETECT_BOOKING_INTENT_SYSTEM_PROMPT = """You are an expert at analyzing conversations to detect meeting/appointment booking intent and extracting booking details.

The conversation may be in ENGLISH or ARABIC. Analyze and respond in the same language for any user-facing text.

STEP 1 - Detect intent: Is there intent to book or schedule a meeting, call, session, or appointment?

Count as YES (has_meeting_intent = true) when the user has done ANY of the following:
- Asked whether booking is possible (e.g. "whether booking a session here is possible", "can I book through chat")
- Stated they want an appointment/session/meeting (e.g. "I definitely want an appointment", "I want to book a session")
- Started providing booking details (name, date, time) even if piece by piece or across many messages
- Asked to save or confirm what they have given so far (e.g. "Can you save what I gave you so far?")
- Used phrases like "let's meet", "schedule a call", "book a slot", "جدولة اجتماع", "حجز موعد"

Intent can be established early and then the user fills in details over many messages. The whole conversation is one booking flow.

Count as NO only when: purely small talk, or "nice meeting you" / goodbye with no scheduling, or the user clearly declined or cancelled booking.

When in doubt, if the user has expressed wanting to book or has provided any booking-related details (name, date, time), set has_meeting_intent = true.

STEP 2 - If intent is YES: Extract from the messages:
- first_name (person who wants the meeting or is being scheduled)
- last_name
- date (prefer ISO format YYYY-MM-DD when possible, e.g. 2025-03-15; otherwise a clear date phrase)
- start_time (prefer 24h HH:MM or HH:MM:SS, e.g. 14:00 for 2pm)
- end_time (same format as start_time)

STEP 3 - Info complete? Set info_complete to true ONLY if all five (first_name, last_name, date, start_time, end_time) are clearly present. Otherwise false.

STEP 4 - If info_complete is false: Write missing_info_prompt in the SAME language as the conversation, politely listing only what is missing (e.g. "Please provide your last name, the meeting date, and the end time.").

Respond with a single valid JSON object, no other text:

{
  "has_meeting_intent": true or false,
  "info_complete": true or false or null,
  "first_name": "string or null",
  "last_name": "string or null",
  "date": "string or null",
  "start_time": "string or null",
  "end_time": "string or null",
  "missing_info_prompt": "string or null"
}

If has_meeting_intent is false: set info_complete, first_name, last_name, date, start_time, end_time, missing_info_prompt to null. If true, fill extracted fields (null if missing), set info_complete, and set missing_info_prompt only when info_complete is false.

Example: User asked "whether booking a session is possible", then said "I definitely want an appointment", gave name Sophia Turner, date 2026-04-16, start time 18:00, but has not confirmed end time. Correct output: has_meeting_intent true, info_complete false, first_name "Sophia", last_name "Turner", date "2026-04-16", start_time "18:00", end_time null, missing_info_prompt asking for the end time."""


def _extract_json_from_response(content: str) -> dict | None:
    content = content.strip()
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


def _is_intent_true(value: object) -> bool:
    """Treat True, 'true', 'yes', 'YES' as intent true; anything else as false."""
    if value is True:
        return True
    if isinstance(value, str) and value.strip().lower() in ("true", "yes", "1"):
        return True
    return False


def _salvage_data_from_raw_content(content: str) -> dict | None:
    """When JSON parse fails, try to detect has_meeting_intent and basic fields from raw text."""
    if not content or not isinstance(content, str):
        return None
    content = content.strip()
    # Look for has_meeting_intent: true (with optional quotes/spacing)
    if re.search(r'"has_meeting_intent"\s*:\s*true', content, re.IGNORECASE):
        # Build minimal data; try to extract field values with simple regex
        data = {"has_meeting_intent": True, "info_complete": False}
        for key, pattern in [
            ("first_name", r'"first_name"\s*:\s*"([^"]*)"'),
            ("last_name", r'"last_name"\s*:\s*"([^"]*)"'),
            ("date", r'"date"\s*:\s*"([^"]*)"'),
            ("expected_date", r'"expected_date"\s*:\s*"([^"]*)"'),
            ("start_time", r'"start_time"\s*:\s*"([^"]*)"'),
            ("end_time", r'"end_time"\s*:\s*"([^"]*)"'),
        ]:
            m = re.search(pattern, content)
            if m and m.group(1).strip():
                data[key] = m.group(1).strip()
        if "date" not in data and "expected_date" in data:
            data["date"] = data["expected_date"]
        return data
    return None


def _build_missing_info_message(
    first_name: str | None,
    last_name: str | None,
    date_str: str | None,
    start_time: str | None,
    end_time: str | None,
) -> str:
    """Build a message that explicitly lists which booking fields are missing."""
    missing = []
    if not (first_name and str(first_name).strip()):
        missing.append("first name")
    if not (last_name and str(last_name).strip()):
        missing.append("last name")
    if not (date_str and str(date_str).strip()):
        missing.append("date")
    if not (start_time and str(start_time).strip()):
        missing.append("start time")
    if not (end_time and str(end_time).strip()):
        missing.append("end time")
    if not missing:
        return "Please provide the missing information."
    return "Please provide the following missing information: " + ", ".join(missing) + "."


def _parse_time_to_minutes(t: str) -> int | None:
    """Convert time string (HH:MM or HH:MM:SS or 2pm-style) to minutes since midnight if possible."""
    if not t or not isinstance(t, str):
        return None
    t = t.strip()
    # HH:MM or HH:MM:SS
    parts = re.split(r"[:.]", t)
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        h = int(parts[0])
        m = int(parts[1])
        if 0 <= h <= 23 and 0 <= m <= 59:
            return h * 60 + m
    # Try 2pm / 14:00 style
    lower = t.lower().replace(" ", "")
    if "pm" in lower or "am" in lower:
        num = re.search(r"(\d{1,2})", t)
        if num:
            h = int(num.group(1))
            if "pm" in lower and h < 12:
                h += 12
            elif "am" in lower and h == 12:
                h = 0
            return h * 60
    return None


def _compute_duration_minutes(start_time: str, end_time: str) -> int | None:
    s = _parse_time_to_minutes(start_time)
    e = _parse_time_to_minutes(end_time)
    if s is not None and e is not None and e > s:
        return e - s
    return None


def _normalize_time_for_db(t: str) -> str | None:
    """Try to return HH:MM or HH:MM:SS for MySQL TIME."""
    if not t or not isinstance(t, str):
        return None
    t = t.strip()
    parts = re.split(r"[:.]", t)
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        h = int(parts[0])
        m = int(parts[1])
        sec = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
        if 0 <= h <= 23 and 0 <= m <= 59 and 0 <= sec <= 59:
            return f"{h:02d}:{m:02d}:{sec:02d}" if sec else f"{h:02d}:{m:02d}:00"
    return None


def _normalize_date_for_db(d: str) -> str | None:
    """Return YYYY-MM-DD if parseable; otherwise None (caller may still use raw string in message)."""
    if not d or not isinstance(d, str):
        return None
    d = d.strip()
    # Already ISO
    if re.match(r"^\d{4}-\d{2}-\d{2}$", d):
        return d
    # Try parsing common formats
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            dt = datetime.strptime(d[:10], fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


# --- Duplicate check: same first_name, last_name, date, overlapping start_time/end_time ---

def _check_booking_conflict(conn, first_name: str, last_name: str, date_str: str, start_time: str, end_time: str) -> bool:
    """Return True if there is an existing row with same person and overlapping time on the same date."""
    cursor = conn.cursor()
    try:
        # Normalize date for comparison (use as-is if not ISO so we can still match)
        date_val = _normalize_date_for_db(date_str) or date_str
        start_norm = _normalize_time_for_db(start_time) or start_time
        end_norm = _normalize_time_for_db(end_time) or end_time

        # Overlap: existing.start_time < new.end_time AND existing.end_time > new.start_time
        query = """
            SELECT 1 FROM appointment_bookings
            WHERE first_name = %s AND last_name = %s AND date = %s
              AND start_time < %s AND end_time > %s
            LIMIT 1
        """
        cursor.execute(query, (first_name, last_name, date_val, end_norm, start_norm))
        return cursor.fetchone() is not None
    finally:
        cursor.close()


def _insert_booking(
    conn,
    first_name: str,
    last_name: str,
    date_str: str,
    start_time: str,
    end_time: str,
) -> int:
    """Insert into appointment_bookings. Returns appointment_booking_id. Uses defaults for columns not provided."""
    date_val = _normalize_date_for_db(date_str) or date_str
    start_norm = _normalize_time_for_db(start_time) or start_time
    end_norm = _normalize_time_for_db(end_time) or end_time
    duration = _compute_duration_minutes(start_time, end_time)

    cursor = conn.cursor()
    try:
        insert_sql = """
            INSERT INTO appointment_bookings (
                gym_name, appointment_name, appointment_category, instructor, price, location,
                first_name, last_name, phone_number, date, start_time, end_time, duration,
                status, attended, added_by
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
        """
        cursor.execute(insert_sql, (
            "",   # gym_name
            "Meeting",  # appointment_name
            "",   # appointment_category
            "",   # instructor
            0,    # price
            "",   # location
            first_name,
            last_name,
            None, # phone_number (often nullable)
            date_val,
            start_norm,
            end_norm,
            duration or 0,
            "scheduled",
            0,    # attended
            "api",
        ))
        conn.commit()
        return cursor.lastrowid
    finally:
        cursor.close()


# --- Routes ---

@app.get("/")
def root():
    return {"name": "booking_intent_detection", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/detect-booking-intent",
    response_model=DetectBookingIntentResponse,
    response_model_exclude_none=True,
    summary="Detect booking intent, extract details, check conflicts, create booking",
    description="Single endpoint: detects intent from messages (EN/AR). If no intent, returns message. If intent but info incomplete, returns missing_info_prompt. If intent and info complete, checks for duplicate booking (same person, date, overlapping time); if conflict returns message; if no conflict inserts into appointment_bookings and returns success.",
)
def detect_booking_intent(request: DetectBookingIntentRequest) -> DetectBookingIntentResponse:
    client = get_deepseek_client()
    api_messages = [
        {"role": "system", "content": DETECT_BOOKING_INTENT_SYSTEM_PROMPT},
        *[{"role": m.role, "content": m.content} for m in request.messages],
    ]

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=api_messages,
            temperature=0.2,
            max_tokens=1024,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"DeepSeek API error: {str(e)}") from e

    content = (response.choices[0].message.content or "").strip()
    data = _extract_json_from_response(content)
    if data is None:
        data = _salvage_data_from_raw_content(content)

    if not data:
        return DetectBookingIntentResponse(
            has_meeting_intent=False,
            message="There is no intent of booking a meeting.",
        )

    has_intent = _is_intent_true(data.get("has_meeting_intent"))

    if not has_intent:
        return DetectBookingIntentResponse(
            has_meeting_intent=False,
            message="There is no intent of booking a meeting.",
        )

    # Intent is true
    first_name = data.get("first_name") or None
    last_name = data.get("last_name") or None
    date_str = data.get("date") or data.get("expected_date") or None
    start_time = data.get("start_time") or None
    end_time = data.get("end_time") or None
    info_complete = data.get("info_complete") is True
    missing_info_prompt = data.get("missing_info_prompt") or None

    extracted = ExtractedBookingInfo(
        first_name=first_name,
        last_name=last_name,
        date=date_str,
        start_time=start_time,
        end_time=end_time,
    )

    if not info_complete:
        return DetectBookingIntentResponse(
            has_meeting_intent=True,
            message=_build_missing_info_message(first_name, last_name, date_str, start_time, end_time),
            info_complete=False,
            extracted=extracted,
        )

    # Info complete: check conflict then insert
    if not first_name or not last_name or not date_str or not start_time or not end_time:
        return DetectBookingIntentResponse(
            has_meeting_intent=True,
            message=_build_missing_info_message(first_name, last_name, date_str, start_time, end_time),
            info_complete=False,
            extracted=extracted,
        )

    try:
        conn = get_db_connection()
    except MySQLError as e:
        raise HTTPException(status_code=503, detail=f"Database connection error: {str(e)}") from e

    try:
        conflict = _check_booking_conflict(conn, first_name, last_name, date_str, start_time, end_time)
        if conflict:
            return DetectBookingIntentResponse(
                has_meeting_intent=True,
                message="There is a conflict with an existing meeting at this time. Please choose a different time or date.",
                info_complete=True,
                extracted=extracted,
                booking_conflict=True,
            )

        booking_id = _insert_booking(conn, first_name, last_name, date_str, start_time, end_time)
        return DetectBookingIntentResponse(
            has_meeting_intent=True,
            message="Booking created successfully.",
            info_complete=True,
            extracted=extracted,
            booking_conflict=False,
            booking_created=True,
            booking_id=booking_id,
        )
    except MySQLError as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}") from e
    finally:
        conn.close()


@app.post(
    "/intent-only-no-db",
    response_model=DetectBookingIntentResponse,
    response_model_exclude_none=True,
    summary="Detect intent and check conflict only (no DB insert)",
    description="Same as detect-booking-intent but never inserts. Detects intent, extracts details; if info complete, checks DB for conflict. If conflict, returns conflict message. If no conflict, returns the extracted data in the response without creating a booking.",
)
def intent_only_no_db(request: DetectBookingIntentRequest) -> DetectBookingIntentResponse:
    client = get_deepseek_client()
    api_messages = [
        {"role": "system", "content": DETECT_BOOKING_INTENT_SYSTEM_PROMPT},
        *[{"role": m.role, "content": m.content} for m in request.messages],
    ]

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=api_messages,
            temperature=0.2,
            max_tokens=1024,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"DeepSeek API error: {str(e)}") from e

    content = (response.choices[0].message.content or "").strip()
    data = _extract_json_from_response(content)
    if data is None:
        data = _salvage_data_from_raw_content(content)

    if not data:
        return DetectBookingIntentResponse(
            has_meeting_intent=False,
            message="There is no intent of booking a meeting.",
        )

    has_intent = _is_intent_true(data.get("has_meeting_intent"))

    if not has_intent:
        return DetectBookingIntentResponse(
            has_meeting_intent=False,
            message="There is no intent of booking a meeting.",
        )

    first_name = data.get("first_name") or None
    last_name = data.get("last_name") or None
    date_str = data.get("date") or data.get("expected_date") or None
    start_time = data.get("start_time") or None
    end_time = data.get("end_time") or None
    info_complete = data.get("info_complete") is True
    missing_info_prompt = data.get("missing_info_prompt") or None

    extracted = ExtractedBookingInfo(
        first_name=first_name,
        last_name=last_name,
        date=date_str,
        start_time=start_time,
        end_time=end_time,
    )

    if not info_complete:
        return DetectBookingIntentResponse(
            has_meeting_intent=True,
            message=_build_missing_info_message(first_name, last_name, date_str, start_time, end_time),
            info_complete=False,
            extracted=extracted,
        )

    if not first_name or not last_name or not date_str or not start_time or not end_time:
        return DetectBookingIntentResponse(
            has_meeting_intent=True,
            message=_build_missing_info_message(first_name, last_name, date_str, start_time, end_time),
            info_complete=False,
            extracted=extracted,
        )

    # Info complete: check conflict only, do not insert
    try:
        conn = get_db_connection()
    except MySQLError as e:
        raise HTTPException(status_code=503, detail=f"Database connection error: {str(e)}") from e

    try:
        conflict = _check_booking_conflict(conn, first_name, last_name, date_str, start_time, end_time)
        if conflict:
            return DetectBookingIntentResponse(
                has_meeting_intent=True,
                message="There is a conflict with an existing meeting at this time. Please choose a different time or date.",
                info_complete=True,
                extracted=extracted,
                booking_conflict=True,
            )

        # No conflict: return extracted data only (no insert)
        return DetectBookingIntentResponse(
            has_meeting_intent=True,
            message="No conflict. Booking details are available below.",
            info_complete=True,
            extracted=extracted,
            booking_conflict=False,
        )
    except MySQLError as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}") from e
    finally:
        conn.close()

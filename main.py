"""
Meeting Intent Detection API

Accepts a list of messages and uses DeepSeek to determine whether
they show intent to book a meeting, and to extract meeting details when intent is detected.
"""

import json
import os
import re
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(
    title="Meeting Intent Detection API",
    description="Detects whether a conversation shows intent to book a meeting using DeepSeek. Supports English and Arabic.",
    version="1.0.0",
)

# DeepSeek client (OpenAI-compatible API)
def get_deepseek_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="DEEPSEEK_API_KEY is not set. Add it to your .env file.",
        )
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )


# --- Request / Response models ---

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1)


class DetectIntentRequest(BaseModel):
    messages: list[ChatMessage] = Field(
        ...,
        min_length=1,
        description="List of conversation messages to analyze for meeting booking intent. Messages may be in English or Arabic.",
    )


class DetectIntentResponse(BaseModel):
    has_meeting_intent: bool = Field(
        ...,
        description="True if the conversation shows intent to schedule or book a meeting.",
    )
    explanation: str | None = Field(
        default=None,
        description="When meeting intent is detected, a brief analysis in the same language as the conversation (English or Arabic). Omitted when false.",
    )


# --- detect_intent_info: extract meeting details when intent is detected ---

class DetectIntentInfoResponse(BaseModel):
    has_meeting_intent: bool = Field(
        ...,
        description="True if the conversation shows intent to schedule or book a meeting.",
    )
    info_complete: bool | None = Field(
        default=None,
        description="When has_meeting_intent is true: true if first name, last name, expected date, start time and end time are all present; false otherwise. Omitted when no meeting intent.",
    )
    first_name: str | None = Field(default=None, description="Extracted first name. Present when intent is detected.")
    last_name: str | None = Field(default=None, description="Extracted last name. Present when intent is detected.")
    expected_date: str | None = Field(default=None, description="Extracted expected/meeting date. Present when intent is detected.")
    start_time: str | None = Field(default=None, description="Extracted start time. Present when intent is detected.")
    end_time: str | None = Field(default=None, description="Extracted end time. Present when intent is detected.")
    missing_info_prompt: str | None = Field(
        default=None,
        description="When info_complete is false: prompt in the conversation language asking the user to provide the missing information. Omitted when info is complete or no intent.",
    )


# --- System prompt for meeting intent detection (English + Arabic) ---

MEETING_INTENT_SYSTEM_PROMPT = """You are an expert at analyzing conversations to detect whether participants show intent to book or schedule a meeting.

The conversation may be in ENGLISH or ARABIC. Analyze the messages in whichever language they use.

Consider it meeting intent when users:
- Propose or suggest scheduling a call, meeting, or appointment
- Ask about availability for a meeting
- Discuss times, dates, or places for getting together
- Confirm or negotiate meeting details (time, duration, location, format)
- English: "let's meet", "schedule a call", "book a slot", "set up a meeting", "find a time", "when are you free"
- Arabic: "نلتقي", "جدولة اجتماع/مكالمة", "حجز موعد", "متى تكون متاحاً", "نحدد وقتاً", "نرتب لقاء", أي ترتيب لوقت أو مكان للقاء

Do NOT consider it meeting intent when:
- They only make small talk or general discussion with no scheduling
- They mention "meeting" in a different sense (e.g. "nice meeting you" / "تشرفت بمقابلتك" only)
- There is no clear indication of wanting to schedule something

Respond with exactly two lines:
1. First line: either "YES" or "NO" (whether there is meeting booking intent). Always use these exact words.
2. Second line: A single short sentence explaining your reasoning. Use the SAME language as the conversation (English or Arabic).

Example (English):
YES
The user asked to schedule a call next week and suggested Tuesday afternoon.

Example (Arabic):
YES
المستخدم اقترح جدولة مكالمة الأسبوع القادم وذكر يوم الثلاثاء."""


# --- System prompt for detect_intent_info (detect intent + extract details) ---

DETECT_INTENT_INFO_SYSTEM_PROMPT = """You are an expert at analyzing conversations to detect meeting booking intent and extracting meeting details.

The conversation may be in ENGLISH or ARABIC. Analyze and respond in the same language for any user-facing text.

STEP 1 - Detect intent: Is there intent to book or schedule a meeting/call/appointment? (Same rules as before: "let's meet", "schedule a call", "جدولة اجتماع", etc. count; small talk or "nice meeting you" only do not.)

STEP 2 - If intent is YES: Extract from the messages whatever you can find for:
- first_name (person who wants the meeting or is being scheduled with)
- last_name
- expected_date (the date they proposed or mentioned for the meeting - use the exact phrase if vague e.g. "next Tuesday", or ISO date if clear e.g. "2025-03-15")
- start_time (e.g. "14:00", "2pm", "بعد الظهر")
- end_time

STEP 3 - Info complete? Set info_complete to true ONLY if all five fields (first_name, last_name, expected_date, start_time, end_time) are clearly present in the conversation. Otherwise set info_complete to false.

STEP 4 - If info_complete is false: Write a short missing_info_prompt in the SAME language as the conversation (English or Arabic), politely asking the user to provide the missing information. List only what is missing (e.g. "Please provide your last name, the meeting date, and the end time.").

You must respond with a single valid JSON object, no other text. Use this exact structure (use null for any missing or inapplicable value):

{
  "has_meeting_intent": true or false,
  "info_complete": true or false or null,
  "first_name": "string or null",
  "last_name": "string or null",
  "expected_date": "string or null",
  "start_time": "string or null",
  "end_time": "string or null",
  "missing_info_prompt": "string or null"
}

Rules: If has_meeting_intent is false, set info_complete to null and set first_name, last_name, expected_date, start_time, end_time, missing_info_prompt all to null. If has_meeting_intent is true, fill in every extracted field (use null for missing), set info_complete accordingly, and set missing_info_prompt only when info_complete is false."""


@app.get("/")
def root():
    return {
        "name": "meeting_intent_detection",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/detect-meeting-intent",
    response_model=DetectIntentResponse,
    response_model_exclude_none=True,
    summary="Detect meeting booking intent",
    description="Analyzes the given list of messages (English or Arabic). When no meeting intent is detected, returns only has_meeting_intent: false. When intent is detected, returns has_meeting_intent: true plus an analysis in the same language as the conversation.",
)
def detect_meeting_intent(request: DetectIntentRequest) -> DetectIntentResponse:
    client = get_deepseek_client()

    # Build messages for DeepSeek: system prompt + conversation
    api_messages = [
        {"role": "system", "content": MEETING_INTENT_SYSTEM_PROMPT},
        *[{"role": m.role, "content": m.content} for m in request.messages],
    ]

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=api_messages,
            temperature=0.2,
            max_tokens=256,
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"DeepSeek API error: {str(e)}",
        ) from e

    content = (response.choices[0].message.content or "").strip()
    lines = [line.strip() for line in content.split("\n") if line.strip()]

    has_intent = False
    explanation: str | None = None

    if lines:
        first = lines[0].upper()
        has_intent = first.startswith("YES")
        # Only include explanation when meeting intent was detected
        if has_intent:
            explanation = lines[1] if len(lines) > 1 else lines[0]

    return DetectIntentResponse(
        has_meeting_intent=has_intent,
        explanation=explanation,
    )


def _extract_json_from_response(content: str) -> dict | None:
    """Try to parse JSON from LLM response, including inside markdown code blocks."""
    content = content.strip()
    # Try raw parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    # Try to extract from ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try to find a JSON object in the text
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


@app.post(
    "/detect-intent-info",
    response_model=DetectIntentInfoResponse,
    response_model_exclude_none=True,
    summary="Detect intent and extract meeting info",
    description="Analyzes messages (English or Arabic) for meeting intent. If intent is detected, extracts first name, last name, expected date, start time, end time. Returns info_complete true only when all are present; otherwise returns missing_info_prompt to ask the user for the missing information.",
)
def detect_intent_info(request: DetectIntentRequest) -> DetectIntentInfoResponse:
    client = get_deepseek_client()

    api_messages = [
        {"role": "system", "content": DETECT_INTENT_INFO_SYSTEM_PROMPT},
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
        raise HTTPException(
            status_code=502,
            detail=f"DeepSeek API error: {str(e)}",
        ) from e

    content = (response.choices[0].message.content or "").strip()
    data = _extract_json_from_response(content)

    if not data:
        return DetectIntentInfoResponse(
            has_meeting_intent=False,
            info_complete=None,
        )

    has_intent = data.get("has_meeting_intent") is True
    info_complete = data.get("info_complete") if has_intent else None

    return DetectIntentInfoResponse(
        has_meeting_intent=has_intent,
        info_complete=info_complete,
        first_name=data.get("first_name") if has_intent else None,
        last_name=data.get("last_name") if has_intent else None,
        expected_date=data.get("expected_date") if has_intent else None,
        start_time=data.get("start_time") if has_intent else None,
        end_time=data.get("end_time") if has_intent else None,
        missing_info_prompt=data.get("missing_info_prompt") if has_intent and not info_complete else None,
    )

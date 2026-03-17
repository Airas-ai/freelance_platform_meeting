# Booking Intent Detection API

A FastAPI service that detects meeting booking intent from chat messages (English or Arabic), extracts booking details, checks for scheduling conflicts, and creates appointments in a MySQL database.

---

## Overview

The **final** implementation lives in **`final.py`**. It exposes a **single endpoint** that handles the full flow:

1. **Detect intent** – Is the user trying to book a meeting?
2. **Extract details** – First name, last name, date, start time, end time.
3. **Handle incomplete info** – Return a prompt asking for missing fields; the client sends more messages and calls the API again.
4. **Check for duplicates** – When info is complete, check the database for an existing booking for the same person at the same date and overlapping time.
5. **Create or conflict** – If no conflict, insert the booking and return success; if conflict, return a message and do not insert.

---

## Why a single API?

The flow is implemented as **one endpoint** so the client can use one pattern: send the conversation (including any follow-up messages), get back either “no intent”, “missing info + prompt”, “conflict”, or “booking created”. The client keeps appending messages and calling the same endpoint until the outcome is clear.

---

## Database

- **Database:** Set via `MYSQL_DATABASE` in `.env` (e.g. `giganeti_gym_management`).
- **Table:** `appointment_bookings` with columns:
  - `appointment_booking_id`, `gym_name`, `appointment_name`, `appointment_category`, `instructor`, `price`, `location`, `first_name`, `last_name`, `phone_number`, `date`, `start_time`, `end_time`, `duration`, `status`, `attended`, `added_by`

The API inserts rows with the extracted fields; other columns use defaults (e.g. `status='scheduled'`, `added_by='api'`).

---

## Setup

1. **Environment**

   Copy `.env.example` to `.env` and set:

   - `DEEPSEEK_API_KEY` – from [DeepSeek](https://platform.deepseek.com/api_keys)
   - `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE` (database that contains `appointment_bookings`)

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API (final)**

   ```bash
   uvicorn final:app --reload
   ```

   Docs: **http://127.0.0.1:8000/docs**

---

## Endpoint: `POST /detect-booking-intent`

**Request body**

```json
{
  "messages": [
    { "role": "user", "content": "I want to book a session next Tuesday at 2pm for an hour." },
    { "role": "assistant", "content": "Sure! May I have your first and last name?" },
    { "role": "user", "content": "John Smith." }
  ]
}
```

Messages can be in **English or Arabic**.

**Responses**

| Case | Response shape |
|------|----------------|
| No intent | `{ "has_meeting_intent": false, "message": "There is no intent of booking a meeting." }` |
| Intent, info incomplete | `{ "has_meeting_intent": true, "message": "<missing-info prompt>", "info_complete": false, "extracted": { "first_name", "last_name", "date", "start_time", "end_time" } }` |
| Intent, info complete, **conflict** | `{ "has_meeting_intent": true, "message": "There is a conflict with an existing meeting...", "info_complete": true, "extracted": {...}, "booking_conflict": true }` |
| Intent, info complete, **created** | `{ "has_meeting_intent": true, "message": "Booking created successfully.", "info_complete": true, "extracted": {...}, "booking_conflict": false, "booking_created": true, "booking_id": 123 }` |

---

## Flow for the client

1. Send the current conversation as `messages` to `POST /detect-booking-intent`.
2. If `has_meeting_intent` is `false` → show `message` and stop.
3. If `info_complete` is `false` → show `message` (missing-info prompt) to the user; after the user replies, append the new messages and call the API again.
4. If `booking_conflict` is `true` → show conflict message; do not insert again.
5. If `booking_created` is `true` → show success and use `booking_id` / `extracted` as needed.

---

## Other files

- **`main.py`** – Earlier version with separate endpoints for intent-only and intent+extraction (no DB).
- **`final.py`** – Full flow: intent, extraction, duplicate check, and insert into `appointment_bookings`.
- **`schema.sql`** / **`run_schema.py`** – For the `meeting_appointments` table in another database; the final API uses the client’s `appointment_bookings` table instead.

---

## Requirements

- Python 3.10+
- FastAPI, uvicorn, python-dotenv, openai (DeepSeek), mysql-connector-python

See **`requirements.txt`**.

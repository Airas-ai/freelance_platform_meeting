# AI Booking Intent Detection Service

A FastAPI-based NLP microservice that detects meeting booking intent in chat conversations and extracts user contact information in real time.

The service analyzes recent chat messages between users and freelancers to determine whether the conversation indicates an intent to schedule a meeting. When booking intent is detected, the system extracts key user details required for scheduling.

The system is designed to be **stateless**, lightweight, and easily integrated with external platforms.

---

## Features

- Real-time **meeting intent detection**
- Automatic extraction of:
  - First Name
  - Last Name
  - Email
- Identification of **missing user information**
- Generation of **suggested prompts** to request missing fields
- **Stateless architecture** (no database required)
- Designed for **high concurrency and scalability**
- Powered by **OpenAI API**

---

## How It Works

1. The client application sends the **latest N chat messages** (typically 10–15 messages) whenever a new message is received.
2. The service analyzes the conversation to detect **booking intent**.
3. If intent is detected, the system extracts available user details:
   - First name
   - Last name
   - Email
4. If required information is missing, the API returns:
   - Which fields are missing
   - A suggested prompt to request them
5. The client sends updated message windows as the conversation continues.

This loop continues until all required fields are collected.

---

## Example Request

```json
{
  "chat_id": "12345",
  "messages": [
    {
      "role": "user",
      "text": "Hi, I would like to schedule a meeting."
    },
    {
      "role": "freelancer",
      "text": "Sure! Could you share your details?"
    }
  ]
}
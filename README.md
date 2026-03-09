AI Booking Intent Detection Service

This project provides a FastAPI-based NLP microservice that detects meeting booking intent in chat conversations and extracts user contact information in real time.

The service analyzes recent chat messages between users and freelancers to determine whether the conversation indicates an intent to schedule a meeting. When booking intent is detected, the system extracts key user information required to proceed with scheduling.

The system is designed to operate as a stateless service and integrates easily with external applications (e.g., Laravel-based platforms) via a simple API.

Key Features

Real-time meeting intent detection

Automatic extraction of:

First Name

Last Name

Email

Detection of missing information

Generation of suggested prompts to request missing details from the user

Stateless architecture (no database required)

Designed for high concurrency and horizontal scalability

Uses OpenAI API for NLP processing

How It Works

The client application sends the latest N chat messages (e.g., last 10–15 messages) to the API whenever a new message is received.

The service analyzes the conversation to determine if there is intent to book a meeting.

If booking intent is detected, the system extracts any available user information:

First name

Last name

Email

If required fields are missing, the API returns a suggested prompt that can be sent to the user to collect the missing information.

The client continues sending updated message windows as the conversation progresses, allowing the system to extract additional details in real time.

Example Request
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
Example Response
{
  "booking_intent": true,
  "confidence": 0.91,
  "fields": {
    "first_name": "John",
    "last_name": null,
    "email": null
  },
  "missing_fields": ["last_name", "email"],
  "ask_user_message": "Great! Could you please share your last name and email so we can schedule the meeting?"
}
Architecture Overview

FastAPI for the API layer

OpenAI API for intent detection and information extraction

Stateless design — no persistent storage required

Sliding message window approach for efficient real-time processing

Deployment

The service is designed to be deployed on a lightweight cloud instance (e.g., AWS EC2) and scaled horizontally as needed.

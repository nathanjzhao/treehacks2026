import os
import json
import uuid
import asyncio
import logging
import re
import base64
import time
import httpx
from dataclasses import dataclass, field
from datetime import datetime, date, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from google.cloud import firestore  # type: ignore
except Exception:  # pragma: no cover
    firestore = None  # type: ignore

try:
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover
    storage = None  # type: ignore

try:
    from google.cloud import texttospeech  # type: ignore
except Exception:  # pragma: no cover
    texttospeech = None  # type: ignore

try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # type: ignore


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("realityhacks-backend")


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

EST = ZoneInfo("America/New_York")
GEMINI_MODEL = "gemini-3.0"


def _format_query_timestamp(ts: datetime, now: datetime) -> str:
    """Format timestamp with relative time for session differentiation."""
    delta = now - ts
    if delta.total_seconds() < 60:
        relative = "just now"
    elif delta.total_seconds() < 3600:
        mins = int(delta.total_seconds() / 60)
        relative = f"{mins}m ago"
    elif delta.total_seconds() < 86400:
        hours = int(delta.total_seconds() / 3600)
        relative = f"{hours}h ago"
    else:
        days = int(delta.total_seconds() / 86400)
        relative = f"{days}d ago"
    return f"{ts.strftime('%Y-%m-%d %H:%M:%S')} ({relative})"


MAX_QUERY_IMAGES = 16
DEFAULT_QUERY_IMAGES = 8

RAW_MEDIA_BUCKET = "reality-hack-2026-raw-media"
PROCESSED_MEDIA_BUCKET = "reality-hack-2026-processed-media"

# =============================================================================
# SYSTEM PROMPTS (well-labeled for easy modification)
# =============================================================================

CAPTURE_ANALYSIS_PROMPT = """You are analyzing a memory capture for a memory assistance application.
The user relies on this app to remember their daily experiences.

Given:
- An image (if provided)
- A transcription of audio (if provided)  
- The timestamp: {timestamp}

Analyze and return a JSON object with:
{{
  "imageSummary": "Brief description of what's shown in the image",
  "themes": ["theme1", "theme2"],
  "mood": "the emotional tone (e.g., happy, focused, relaxed, stressed)",
  "location": "best guess of location type (e.g., home, office, cafe, outdoors)",
  "detectedFaces": [
    {{
      "description": "physical description of person",
      "possibleName": "name if mentioned in transcription, otherwise null",
      "confidence": 0.0-1.0
    }}
  ],
  "mentionedNames": ["names mentioned in transcription"],
  "keyMoment": "one sentence capturing the essence of this memory"
}}

Transcription: {transcription}

Be concise but informative. Focus on details that would help someone recall this moment later.
Return ONLY valid JSON, no markdown formatting."""

HOURLY_SUMMARY_PROMPT = """You are creating an hourly summary for a memory assistance application.
The user relies on these summaries to remember their day.
NOTE: There may be 60+ memory captures per hour - this is normal. Synthesize them into a comprehensive narrative.

Memory captures from this hour:
{captures_json}

Queries made this hour (user was thinking about these topics):
{queries_json}

Create a detailed, comprehensive summary. Return a JSON object:
{{
  "summary": "A detailed 5-10 sentence narrative summary of everything that happened this hour. Include specific details, conversations, activities, and transitions between moments. Be thorough.",
  "themes": ["theme1", "theme2", "theme3", "theme4", "theme5"],
  "events": [
    {{"time": "HH:MM", "description": "what happened"}},
    {{"time": "HH:MM", "description": "what happened"}}
  ],
  "peoplePresent": ["names of people seen or mentioned"],
  "locations": ["places visited this hour"],
  "highlight": "the single most notable or meaningful moment from this hour",
  "mood": "overall emotional tone of the hour",
  "activities": ["list of distinct activities performed"]
}}

Be warm and personal. Include enough detail that someone could vividly recall this hour later.
Return ONLY valid JSON, no markdown formatting."""

DAILY_SUMMARY_PROMPT = """You are creating a daily summary for a memory assistance application.
This summary helps the user remember their entire day.
NOTE: Each hour may contain 60+ captures, so a full day could have hundreds of moments. Create a rich, detailed narrative.

Given the hourly summaries from today:
{hourly_json}

Create a comprehensive daily summary. Return a JSON object:
{{
  "summary": "A detailed 10-15 sentence narrative of the entire day. Describe the flow of the day from morning to evening, key activities, interactions, and how the day progressed. Be thorough and vivid.",
  "timeline": [
    {{"time": "HH:MM", "event": "description of significant event or activity"}},
    {{"time": "HH:MM", "event": "description of significant event or activity"}}
  ],
  "themes": ["theme1", "theme2", "theme3", "theme4", "theme5", "theme6"],
  "highlights": [
    {{"time": "HH:MM", "description": "first highlight of the day"}},
    {{"time": "HH:MM", "description": "second highlight of the day"}},
    {{"time": "HH:MM", "description": "third highlight of the day"}}
  ],
  "mood": "overall emotional arc of the day - describe how mood shifted throughout",
  "peopleInteractions": [
    {{"name": "person name", "context": "how/when they interacted"}}
  ],
  "locations": ["all locations visited today"],
  "accomplishments": ["things completed or achieved today"],
  "morningOverview": "2-3 sentences about the morning",
  "afternoonOverview": "2-3 sentences about the afternoon", 
  "eveningOverview": "2-3 sentences about the evening (if applicable)"
}}

Write in a warm, reflective tone. Help the user appreciate and vividly remember their entire day.
Return ONLY valid JSON, no markdown formatting."""

CONTACT_UPDATE_PROMPT = """You are managing a contacts database for a memory assistance application.

Current contacts:
{contacts_json}

New information from a memory capture:
- Detected faces: {detected_faces}
- Mentioned names: {mentioned_names}
- Transcription context: {transcription}

Determine what the complete JSON file should be.

Return ONLY valid JSON, no markdown formatting."""

QUERY_ROUTER_PROMPT = """You are a memory assistant helping a user recall their experiences.

User Profile:
{user_profile}

Known Contacts:
{contacts_summary}

Recent Context:
- Last hour summary: {last_hour_summary}
- Last day summary: {last_day_summary}

Recent Queries (with timestamps - use to identify session boundaries, queries hours/days apart are different sessions):
{recent_queries}

Current Query: "{query_text}"
Date range: {date_range}

Determine what data is needed to answer this query. Consider conversation context for follow-ups.
Return a JSON object:
{{
  "queryType": "person" | "event" | "location" | "time" | "general",
  "needsFaceImages": true/false,
  "relevantContacts": ["names of contacts that might be relevant"],
  "dataNeeded": {{
    "captures": true/false,
    "hourlySummaries": true/false,
    "dailySummaries": true/false,
    "specificDates": ["YYYY-MM-DD"]
  }},
  "needsMoreQueryContext": 0-20,
  "needsClarification": true/false,
  "clarificationQuestion": "question if clarification needed, else null"
}}

Note: Set "needsMoreQueryContext" to fetch additional conversation history (0 = use default 5, up to 20 max).

Return ONLY valid JSON, no markdown formatting."""

QUERY_ANSWER_PROMPT = """You are a memory assistant helping a user recall their experiences.
Be warm, helpful, and specific, although brief and natural. Reference actual details from the memories.
If this follows up on a recent query, maintain context. 

User Profile:
{user_profile}

Recent Queries (timestamps show session context)
{recent_queries}

Relevant Memories:
{memory_context}

Relevant Contacts:
{contacts_info}

Current Time: {current_time}
Current Query: "{query_text}"

Provide a helpful, conversational answer. 

Return a JSON object:
{{
  "answer": "Your conversational response",
  "confidence": 0.0-1.0,
  "sourceCaptureIds": ["ids of captures used"],
  "mentionedContacts": ["names mentioned in answer"],
  "suggestedFollowUp": "optional follow-up question or null"
}}

Return ONLY valid JSON, no markdown formatting."""

USER_INIT_PROMPT = """You are setting up a user profile for a memory assistance application.

Given this initial information about the user:
{user_info}

Extract and organize the information. Return a JSON object:
{{
  "name": "user's name",
  "occupation": "what they do",
  "familyMembers": ["family member names/relations"],
  "dailyRoutines": "description of typical day",
  "medicalNotes": "important health information",
  "preferences": {{"key": "value"}},
  "initialContacts": [
    {{
      "name": "contact name",
      "relationship": "relationship to user",
      "notes": "any relevant notes"
    }}
  ]
}}

Return ONLY valid JSON, no markdown formatting."""

# =============================================================================
# FASTAPI APP SETUP
# =============================================================================

app = FastAPI()

# CORS enabled for all origins (hackathon-friendly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


def _parse_iso_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO8601 string")
    s = value.strip()
    # Support trailing Z
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _date_key_from_dt(dt: datetime) -> str:
    # stored as YYYY-MM-DD (Unity request format)
    return dt.astimezone(timezone.utc).date().isoformat()


@dataclass
class InMemoryDB:
    captures: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    processed_memories: Dict[Tuple[str, str],
                             Dict[str, Any]] = field(default_factory=dict)
    user_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    contacts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    hourly_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    daily_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class FirestoreRepo:
    def __init__(self) -> None:
        self.project_id = os.environ.get("GCP_PROJECT_ID")
        self._mem = InMemoryDB()
        self._client = None

        if firestore is None:
            logger.warning(
                "google-cloud-firestore not available; using in-memory DB")
            return

        try:
            self._client = firestore.AsyncClient(project=self.project_id)
            logger.info("Firestore client initialized (project=%s)",
                        self.project_id)
        except Exception:
            logger.exception(
                "Failed to initialize Firestore; using in-memory DB")
            self._client = None

    @property
    def is_firestore(self) -> bool:
        return self._client is not None

    # -------------------------------------------------------------------------
    # Memory Captures
    # -------------------------------------------------------------------------
    async def create_capture(self, doc: Dict[str, Any]) -> None:
        if not self._client:
            self._mem.captures[doc["id"]] = doc
            return
        await self._client.collection("memory_captures").document(doc["id"]).set(doc)

    async def get_capture(self, capture_id: str) -> Optional[Dict[str, Any]]:
        if not self._client:
            return self._mem.captures.get(capture_id)
        snap = await self._client.collection("memory_captures").document(capture_id).get()
        return snap.to_dict() if snap.exists else None

    async def update_capture(self, capture_id: str, updates: Dict[str, Any]) -> None:
        if not self._client:
            if capture_id in self._mem.captures:
                self._mem.captures[capture_id].update(updates)
            return
        await self._client.collection("memory_captures").document(capture_id).update(updates)

    async def list_captures_for_date(self, user_id: str, day: date) -> List[Dict[str, Any]]:
        if not self._client:
            out: List[Dict[str, Any]] = []
            for doc in self._mem.captures.values():
                if doc.get("userId") != user_id:
                    continue
                ts = doc.get("timestamp")
                if isinstance(ts, datetime):
                    d = ts.astimezone(timezone.utc).date()
                else:
                    try:
                        d = _parse_iso_datetime(
                            ts).astimezone(timezone.utc).date()
                    except Exception:
                        continue
                if d == day:
                    out.append(doc)
            out.sort(key=lambda x: x.get("timestamp") or datetime.min)
            return out

        start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        end = start.replace(hour=23, minute=59, second=59, microsecond=999999)
        q = (
            self._client.collection("memory_captures")
            .where("userId", "==", user_id)
            .where("timestamp", ">=", start)
            .where("timestamp", "<=", end)
            .order_by("timestamp")
        )
        snaps = [doc async for doc in q.stream()]
        return [s.to_dict() for s in snaps]

    async def list_captures_in_range(self, user_id: str, start_dt: datetime, end_dt: datetime) -> List[Dict[str, Any]]:
        if not self._client:
            out: List[Dict[str, Any]] = []
            for doc in self._mem.captures.values():
                if doc.get("userId") != user_id:
                    continue
                ts = doc.get("timestamp")
                if isinstance(ts, datetime):
                    if start_dt <= ts <= end_dt:
                        out.append(doc)
            out.sort(key=lambda x: x.get("timestamp") or datetime.min)
            return out

        q = (
            self._client.collection("memory_captures")
            .where("userId", "==", user_id)
            .where("timestamp", ">=", start_dt)
            .where("timestamp", "<=", end_dt)
            .order_by("timestamp")
        )
        snaps = [doc async for doc in q.stream()]
        return [s.to_dict() for s in snaps]

    # -------------------------------------------------------------------------
    # User Profiles (lifestyle/general info)
    # -------------------------------------------------------------------------
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        doc_id = f"profile_{user_id}"
        if not self._client:
            return self._mem.user_profiles.get(doc_id)
        snap = await self._client.collection("user_profiles").document(doc_id).get()
        return snap.to_dict() if snap.exists else None

    async def upsert_user_profile(self, user_id: str, doc: Dict[str, Any]) -> None:
        doc_id = f"profile_{user_id}"
        doc["userId"] = user_id
        doc["updatedAt"] = datetime.now(timezone.utc)
        if not self._client:
            self._mem.user_profiles[doc_id] = doc
            return
        await self._client.collection("user_profiles").document(doc_id).set(doc, merge=True)

    # -------------------------------------------------------------------------
    # Contacts
    # -------------------------------------------------------------------------
    async def get_contacts(self, user_id: str) -> Optional[Dict[str, Any]]:
        doc_id = f"contacts_{user_id}"
        if not self._client:
            return self._mem.contacts.get(doc_id)
        snap = await self._client.collection("contacts").document(doc_id).get()
        return snap.to_dict() if snap.exists else None

    async def upsert_contacts(self, user_id: str, doc: Dict[str, Any]) -> None:
        doc_id = f"contacts_{user_id}"
        doc["userId"] = user_id
        doc["updatedAt"] = datetime.now(timezone.utc)
        if not self._client:
            self._mem.contacts[doc_id] = doc
            return
        await self._client.collection("contacts").document(doc_id).set(doc, merge=True)

    # -------------------------------------------------------------------------
    # Hourly Summaries
    # -------------------------------------------------------------------------
    async def get_hourly_summary(self, user_id: str, date_str: str, hour: int) -> Optional[Dict[str, Any]]:
        doc_id = f"hourly_{user_id}_{date_str}_{hour:02d}"
        if not self._client:
            return self._mem.hourly_summaries.get(doc_id)
        snap = await self._client.collection("hourly_summaries").document(doc_id).get()
        return snap.to_dict() if snap.exists else None

    async def create_hourly_summary(self, user_id: str, date_str: str, hour: int, doc: Dict[str, Any]) -> None:
        doc_id = f"hourly_{user_id}_{date_str}_{hour:02d}"
        doc["userId"] = user_id
        doc["date"] = date_str
        doc["hour"] = hour
        doc["createdAt"] = datetime.now(timezone.utc)
        if not self._client:
            self._mem.hourly_summaries[doc_id] = doc
            return
        await self._client.collection("hourly_summaries").document(doc_id).set(doc)

    async def list_hourly_summaries_for_date(self, user_id: str, date_str: str) -> List[Dict[str, Any]]:
        if not self._client:
            out = []
            prefix = f"hourly_{user_id}_{date_str}_"
            for k, v in self._mem.hourly_summaries.items():
                if k.startswith(prefix):
                    out.append(v)
            out.sort(key=lambda x: x.get("hour", 0))
            return out

        q = (
            self._client.collection("hourly_summaries")
            .where("userId", "==", user_id)
            .where("date", "==", date_str)
            .order_by("hour")
        )
        snaps = [doc async for doc in q.stream()]
        return [s.to_dict() for s in snaps]

    # -------------------------------------------------------------------------
    # Daily Summaries
    # -------------------------------------------------------------------------
    async def get_daily_summary(self, user_id: str, date_str: str) -> Optional[Dict[str, Any]]:
        doc_id = f"daily_{user_id}_{date_str}"
        if not self._client:
            return self._mem.daily_summaries.get(doc_id)
        snap = await self._client.collection("daily_summaries").document(doc_id).get()
        return snap.to_dict() if snap.exists else None

    async def create_daily_summary(self, user_id: str, date_str: str, doc: Dict[str, Any]) -> None:
        doc_id = f"daily_{user_id}_{date_str}"
        doc["userId"] = user_id
        doc["date"] = date_str
        doc["createdAt"] = datetime.now(timezone.utc)
        if not self._client:
            self._mem.daily_summaries[doc_id] = doc
            return
        await self._client.collection("daily_summaries").document(doc_id).set(doc)

    # -------------------------------------------------------------------------
    # Recent Queries
    # -------------------------------------------------------------------------
    async def get_recent_queries(self, user_id: str) -> Optional[Dict[str, Any]]:
        doc_id = f"queries_{user_id}"
        if not self._client:
            return self._mem.processed_memories.get(("recent_queries", user_id))
        snap = await self._client.collection("recent_queries").document(doc_id).get()
        return snap.to_dict() if snap.exists else None

    async def add_recent_query(self, user_id: str, query_record: Dict[str, Any]) -> None:
        doc_id = f"queries_{user_id}"
        now = datetime.now(timezone.utc)
        query_record["timestamp"] = now

        if not self._client:
            existing = self._mem.processed_memories.get(
                ("recent_queries", user_id), {"queries": []})
            existing["queries"].insert(0, query_record)
            existing["queries"] = existing["queries"][:50]  # Keep last 50
            existing["updatedAt"] = now
            self._mem.processed_memories[(
                "recent_queries", user_id)] = existing
            return

        doc_ref = self._client.collection("recent_queries").document(doc_id)
        snap = await doc_ref.get()
        if snap.exists:
            data = snap.to_dict()
            queries = data.get("queries", [])
        else:
            queries = []

        queries.insert(0, query_record)
        queries = queries[:50]  # Keep last 50 queries

        await doc_ref.set({
            "userId": user_id,
            "queries": queries,
            "updatedAt": now
        })

    # -------------------------------------------------------------------------
    # Legacy processed_memories (kept for backwards compatibility)
    # -------------------------------------------------------------------------
    async def get_processed_memory(self, user_id: str, date_key: str) -> Optional[Dict[str, Any]]:
        if not self._client:
            return self._mem.processed_memories.get((user_id, date_key))
        snap = await self._client.collection("processed_memories").document(f"{user_id}_{date_key}").get()
        return snap.to_dict() if snap.exists else None

    async def upsert_processed_memory(self, user_id: str, date_key: str, doc: Dict[str, Any]) -> None:
        if not self._client:
            self._mem.processed_memories[(user_id, date_key)] = doc
            return
        await self._client.collection("processed_memories").document(f"{user_id}_{date_key}").set(doc)


repo = FirestoreRepo()


class ConnectionManager:
    def __init__(self) -> None:
        self._ios: Dict[str, Set[WebSocket]] = {}
        self._unity: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, kind: str, user_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            bucket = self._ios if kind == "ios" else self._unity
            bucket.setdefault(user_id, set()).add(websocket)
        logger.info("WS connected kind=%s user=%s", kind, user_id)

    async def disconnect(self, kind: str, user_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            bucket = self._ios if kind == "ios" else self._unity
            if user_id in bucket:
                bucket[user_id].discard(websocket)
                if not bucket[user_id]:
                    del bucket[user_id]
        logger.info("WS disconnected kind=%s user=%s", kind, user_id)

    async def send_json(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        await websocket.send_text(json.dumps(data))

    async def broadcast_unity(self, user_id: str, data: Dict[str, Any]) -> None:
        async with self._lock:
            conns = list(self._unity.get(user_id, set()))
        if not conns:
            return
        payload = json.dumps(data)
        dead: List[WebSocket] = []
        for ws in conns:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._unity.get(user_id, set()).discard(ws)


manager = ConnectionManager()


# =============================================================================
# TEXT-TO-SPEECH SERVICE
# =============================================================================

class TTSService:
    """Google Cloud Text-to-Speech service for generating audio from query responses."""

    MAX_TEXT_LENGTH = 5000

    def __init__(self):
        self._client = None
        self._voice = None
        self._audio_config = None

        if texttospeech is None:
            logger.warning(
                "google-cloud-texttospeech not available; TTS disabled")
            return

        try:
            self._client = texttospeech.TextToSpeechClient()
            self._voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name="en-US-Neural2-F",
                ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
            )
            self._audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
            logger.info("TTS service initialized with voice en-US-Neural2-F")
        except Exception:
            logger.exception("Failed to initialize TTS client")
            self._client = None

    @property
    def is_available(self) -> bool:
        return self._client is not None

    async def generate_speech(self, text: str, query_id: str, user_id: str) -> Optional[str]:
        """
        Generate speech from text and upload to Cloud Storage.

        Returns the public URL of the audio file, or None on failure.
        """
        if not self.is_available:
            logger.debug("TTS not available, skipping audio generation")
            return None

        if not text or not text.strip():
            return None

        try:
            # Truncate text if too long
            if len(text) > self.MAX_TEXT_LENGTH:
                text = text[:self.MAX_TEXT_LENGTH]
                logger.info("TTS text truncated to %d chars for query_id=%s",
                            self.MAX_TEXT_LENGTH, query_id)

            # Synthesize speech (sync call wrapped in to_thread)
            def _synthesize():
                synthesis_input = texttospeech.SynthesisInput(text=text)
                response = self._client.synthesize_speech(
                    input=synthesis_input,
                    voice=self._voice,
                    audio_config=self._audio_config
                )
                return response.audio_content

            audio_content = await asyncio.to_thread(_synthesize)

            if not audio_content:
                logger.warning(
                    "TTS returned empty audio for query_id=%s", query_id)
                return None

            # Upload to Cloud Storage
            if storage is None:
                logger.warning(
                    "Storage not available, cannot upload TTS audio")
                return None

            bucket_name = PROCESSED_MEDIA_BUCKET
            object_name = f"query-responses/{user_id}/{query_id}/response.mp3"

            def _upload():
                import io
                client = storage.Client(
                    project=os.environ.get("GCP_PROJECT_ID"))
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(object_name)
                # Use BytesIO to avoid any ACL operations with uniform bucket access
                audio_file = io.BytesIO(audio_content)
                blob.upload_from_file(
                    audio_file, content_type="audio/mpeg", rewind=True)
                return f"https://storage.googleapis.com/{bucket_name}/{object_name}"

            public_url = await asyncio.to_thread(_upload)

            logger.info("TTS audio uploaded: query_id=%s url=%s",
                        query_id, public_url)
            return public_url

        except Exception as e:
            logger.exception(
                "TTS generation failed for query_id=%s: %s", query_id, e)
            return None


tts_service = TTSService()


@app.get("/")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/test-firestore")
async def test_firestore() -> Dict[str, Any]:
    try:
        if firestore is None:
            raise RuntimeError("google-cloud-firestore is not installed")

        client = firestore.AsyncClient(project="mit-reality26cam-1526")
        docs = [d async for d in client.collection("memory_captures").limit(1).stream()]
        sample = [docs[0].to_dict()] if docs else []

        return {
            "status": "success",
            "message": "Firestore working!",
            "sample_data": sample,
        }
    except Exception as e:
        logger.exception("/test-firestore failed")
        return {"status": "error", "error": str(e)}


@app.get("/test-gemini")
async def test_gemini() -> Dict[str, Any]:
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {"status": "error", "error": "GEMINI_API_KEY is missing"}
        if genai is None:
            raise RuntimeError("google-generativeai is not installed")

        def _call() -> str:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash-exp")
            resp = model.generate_content("Say hello in exactly 5 words")
            text = getattr(resp, "text", None)
            if not text:
                text = str(resp)
            return text.strip()

        text = await asyncio.to_thread(_call)

        return {
            "status": "success",
            "message": "Gemini working!",
            "response": text,
        }
    except Exception as e:
        logger.exception("/test-gemini failed")
        return {"status": "error", "error": str(e)}


@app.post("/upload/{capture_id}")
async def upload_capture_media(capture_id: str, file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        if storage is None:
            raise RuntimeError("google-cloud-storage is not installed")

        bucket_name = "reality-hack-2026-raw-media"
        object_name = f"memories/{capture_id}/photo.jpg"

        content = await file.read()
        content_type = file.content_type or "image/jpeg"

        def _upload() -> None:
            client = storage.Client(project=os.environ.get("GCP_PROJECT_ID"))
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(object_name)
            blob.upload_from_string(content, content_type=content_type)

        await asyncio.to_thread(_upload)

        url = f"https://storage.googleapis.com/{bucket_name}/{object_name}"
        return {"status": "success", "url": url, "captureId": capture_id}
    except Exception as e:
        logger.exception("Upload failed capture_id=%s", capture_id)
        return {"status": "error", "error": str(e)}


@app.post("/query-upload/{query_id}")
async def upload_query_image(query_id: str, file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload an image to be used with a query. Returns URL to reference in WebSocket query."""
    try:
        if storage is None:
            raise RuntimeError("google-cloud-storage is not installed")

        bucket_name = "reality-hack-2026-raw-media"
        object_name = f"queries/{query_id}/image.jpg"

        content = await file.read()
        content_type = file.content_type or "image/jpeg"

        def _upload() -> None:
            client = storage.Client(project=os.environ.get("GCP_PROJECT_ID"))
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(object_name)
            blob.upload_from_string(content, content_type=content_type)

        await asyncio.to_thread(_upload)

        url = f"https://storage.googleapis.com/{bucket_name}/{object_name}"
        logger.info(
            "[QUERY_DATA] Uploaded query image query_id=%s url=%s", query_id, url)
        return {"status": "success", "url": url, "queryId": query_id}
    except Exception as e:
        logger.exception("Query image upload failed query_id=%s", query_id)
        return {"status": "error", "error": str(e)}


@app.get("/memories/{user_id}/{date}")
async def get_memories_for_date(user_id: str, date: str) -> Dict[str, Any]:
    try:
        doc = await repo.get_processed_memory(user_id, date)
        if doc is None:
            daily = await repo.get_daily_summary(user_id, date)
            if daily:
                return {"status": "success", "data": daily}
            return {"status": "not_found", "message": "No memories for this date"}
        return {"status": "success", "data": doc}
    except Exception as e:
        logger.exception(
            "Failed to fetch processed memories user=%s date=%s", user_id, date)
        return {"status": "error", "error": str(e)}


# =============================================================================
# USER INITIALIZATION ENDPOINT
# =============================================================================

class UserInitRequest(BaseModel):
    summary: str


@app.post("/init-user/{user_id}")
async def init_user(user_id: str, request: UserInitRequest) -> Dict[str, Any]:
    try:
        existing = await repo.get_user_profile(user_id)
        if existing:
            return {
                "status": "exists",
                "message": "User profile already exists",
                "profile": existing
            }

        prompt = USER_INIT_PROMPT.format(user_info=request.summary)
        parsed = await _call_gemini_text(prompt)

        profile_doc = {
            "name": parsed.get("name", ""),
            "occupation": parsed.get("occupation", ""),
            "familyMembers": parsed.get("familyMembers", []),
            "dailyRoutines": parsed.get("dailyRoutines", ""),
            "medicalNotes": parsed.get("medicalNotes", ""),
            "preferences": parsed.get("preferences", {}),
            "lastDaySummaryDate": None,
            "lastDaySummary": None,
            "lastHourSummaryTime": None,
            "lastHourSummary": None,
            "lastCondensationTime": None,
            "createdAt": datetime.now(timezone.utc),
        }
        await repo.upsert_user_profile(user_id, profile_doc)

        initial_contacts = parsed.get("initialContacts", [])
        if initial_contacts:
            now_iso = datetime.now(timezone.utc).isoformat()
            contacts_list = []
            for c in initial_contacts:
                contacts_list.append({
                    "name": c.get("name", "Unknown"),
                    "relationship": c.get("relationship", "unknown"),
                    "notes": c.get("notes", ""),
                    "bestFacePhotoURL": None,
                    "firstSeen": now_iso,
                    "lastSeen": now_iso,
                    "mentionCount": 0
                })
            await repo.upsert_contacts(user_id, {"contacts": contacts_list})
        else:
            await repo.upsert_contacts(user_id, {"contacts": []})

        logger.info("Initialized user profile for user_id=%s", user_id)
        return {
            "status": "success",
            "message": "User profile created",
            "profile": profile_doc,
            "contactsCount": len(initial_contacts)
        }
    except Exception as e:
        logger.exception("Failed to initialize user user_id=%s", user_id)
        return {"status": "error", "error": str(e)}


@app.get("/user/{user_id}/profile")
async def get_user_profile(user_id: str) -> Dict[str, Any]:
    try:
        profile = await repo.get_user_profile(user_id)
        if not profile:
            return {"status": "not_found", "message": "User profile not found"}
        return {"status": "success", "data": profile}
    except Exception as e:
        logger.exception("Failed to get user profile user_id=%s", user_id)
        return {"status": "error", "error": str(e)}


@app.get("/user/{user_id}/contacts")
async def get_user_contacts(user_id: str) -> Dict[str, Any]:
    try:
        contacts = await repo.get_contacts(user_id)
        if not contacts:
            return {"status": "success", "data": {"contacts": []}}
        return {"status": "success", "data": contacts}
    except Exception as e:
        logger.exception("Failed to get contacts user_id=%s", user_id)
        return {"status": "error", "error": str(e)}


# =============================================================================
# GEMINI HELPER FUNCTIONS
# =============================================================================

def _get_gemini_model():
    """Configure the genai SDK and return the configured model identifier or
    raise a helpful error if configuration is missing.

    Note: newer versions of the `google.generativeai` SDK expose different
    idioms for calling models. Callers should use the returned model name and
    an adapter-based call helper implemented below.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    if genai is None:
        raise RuntimeError("google-generativeai not installed")
    # Ensure SDK is configured (safe to call repeatedly)
    try:
        genai.configure(api_key=api_key)
    except Exception:
        # Older/newer SDKs may raise on configure; ignore and let adapter handle it
        pass
    return GEMINI_MODEL


def _parse_json_response(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


async def _download_image_as_base64(url: str) -> Optional[str]:
    """Download image and return as base64. Uses GCS SDK for private bucket URLs."""
    try:
        # Check if this is a GCS URL for our private bucket
        gcs_prefix = f"https://storage.googleapis.com/{RAW_MEDIA_BUCKET}/"
        if url.startswith(gcs_prefix):
            # Use authenticated GCS SDK for private bucket
            if storage is None:
                logger.error(
                    "google-cloud-storage not installed, cannot download from private bucket")
                return None

            object_name = url[len(gcs_prefix):]

            def _download_blob() -> bytes:
                client = storage.Client(
                    project=os.environ.get("GCP_PROJECT_ID"))
                bucket = client.bucket(RAW_MEDIA_BUCKET)
                blob = bucket.blob(object_name)
                return blob.download_as_bytes()

            content = await asyncio.to_thread(_download_blob)
            return base64.b64encode(content).decode("utf-8")
        else:
            # External URL - use httpx (unauthenticated)
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return base64.b64encode(resp.content).decode("utf-8")
    except Exception as e:
        logger.warning("Failed to download image %s: %s", url, e)
        return None


async def _call_gemini_with_image(prompt: str, image_base64: Optional[str] = None) -> Dict[str, Any]:
    # Reuse the unified adapter which attempts to call either the old
    # `GenerativeModel` API or the newer `genai.models.generate` / `genai.generate`
    text = await _call_gemini_raw(prompt, image_base64)
    return _parse_json_response(text)


async def _call_gemini_text(prompt: str) -> Dict[str, Any]:
    text = await _call_gemini_raw(prompt, None)
    return _parse_json_response(text)


async def _call_gemini_raw(prompt: str, image_base64: Optional[str] = None) -> str:
    """Call the installed `google.generativeai` SDK using a best-effort
    adapter that supports both older and newer SDK call patterns.

    Returns the raw text response (not JSON-parsed). The caller will parse
    JSON as needed.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    if genai is None:
        raise RuntimeError("google-generativeai not installed")

    def _call():
        # Ensure SDK configured
        try:
            genai.configure(api_key=api_key)
        except Exception:
            pass

        # 1) Old style: genai.GenerativeModel(...).generate_content(...)
        if hasattr(genai, "GenerativeModel"):
            try:
                model_obj = genai.GenerativeModel(GEMINI_MODEL)
                if image_base64:
                    image_part = {"mime_type": "image/jpeg",
                                  "data": image_base64}
                    resp = model_obj.generate_content([prompt, image_part])
                else:
                    resp = model_obj.generate_content(prompt)
                text = getattr(resp, "text", None)
                return text if text is not None else str(resp)
            except Exception:
                # Fall through to other adapters
                pass

        # 2) Newer SDK: genai.models.generate(...)
        models_mod = getattr(genai, "models", None)
        if models_mod and hasattr(models_mod, "generate"):
            try:
                if image_base64:
                    # Compose multimodal input as a list of parts when supported
                    inputs = [{"content": prompt}, {
                        "image": {"mime_type": "image/jpeg", "data": image_base64}}]
                else:
                    inputs = {"content": prompt}

                resp = models_mod.generate(model=GEMINI_MODEL, input=inputs)
                # Try common response accessors
                #  - resp.output_text
                #  - resp.output (list)
                ot = getattr(resp, "output_text", None)
                if ot:
                    return ot

                out = getattr(resp, "output", None)
                if out:
                    try:
                        first = out[0]
                        if isinstance(first, dict):
                            # content may be list of blocks
                            content = first.get("content")
                            if isinstance(content, list):
                                for item in content:
                                    if isinstance(item, dict) and "text" in item:
                                        return item["text"]
                            # fallback to joining text-like fields
                            text_fields = []
                            for v in (first.get("content") or []):
                                if isinstance(v, dict) and v.get("text"):
                                    text_fields.append(v.get("text"))
                            if text_fields:
                                return "\n".join(text_fields)
                    except Exception:
                        pass

                # Last resort
                return str(resp)
            except Exception:
                pass

        # 3) Another possible shape: genai.generate(...)
        if hasattr(genai, "generate"):
            try:
                if image_base64:
                    resp = genai.generate(model=GEMINI_MODEL, prompt=[
                                          prompt, {"image": image_base64}])
                else:
                    resp = genai.generate(model=GEMINI_MODEL, prompt=prompt)
                text = getattr(resp, "text", None) or getattr(
                    resp, "output_text", None)
                return text if text is not None else str(resp)
            except Exception:
                pass

        raise RuntimeError(
            "Unsupported google.generativeai SDK API shape; please ensure the SDK is up-to-date")


# =============================================================================
# CAPTURE ANALYSIS WITH GEMINI VISION
# =============================================================================

async def _analyze_capture_with_gemini(capture: Dict[str, Any]) -> Dict[str, Any]:
    ts = capture.get("timestamp")
    if isinstance(ts, datetime):
        ts_str = ts.isoformat()
    else:
        ts_str = str(ts)

    transcription = capture.get(
        "transcription") or "No transcription available"
    photo_url = capture.get("photoURL")

    prompt = CAPTURE_ANALYSIS_PROMPT.format(
        timestamp=ts_str,
        transcription=transcription
    )

    image_b64 = None
    if photo_url:
        image_b64 = await _download_image_as_base64(photo_url)

    try:
        analysis = await _call_gemini_with_image(prompt, image_b64)
        return analysis
    except Exception as e:
        logger.exception("Gemini analysis failed, using fallback")
        return {
            "imageSummary": "Analysis unavailable",
            "themes": [],
            "mood": "unknown",
            "location": "unknown",
            "detectedFaces": [],
            "mentionedNames": [],
            "keyMoment": transcription[:100] if transcription else "No details",
            "error": str(e)
        }


# =============================================================================
# CONTACTS MANAGEMENT
# =============================================================================

async def _update_contacts_from_analysis(user_id: str, analysis: Dict[str, Any], capture: Dict[str, Any]) -> None:
    mentioned_names = analysis.get("mentionedNames") or []
    detected_faces = analysis.get("detectedFaces") or []

    if not mentioned_names and not detected_faces:
        return

    contacts_doc = await repo.get_contacts(user_id)
    if not contacts_doc:
        contacts_doc = {"contacts": []}

    contacts_list = contacts_doc.get("contacts", [])
    contacts_by_name = {c.get("name", "").lower(): c for c in contacts_list}
    now_iso = datetime.now(timezone.utc).isoformat()

    for name in mentioned_names:
        name_lower = name.lower()
        if name_lower in contacts_by_name:
            contacts_by_name[name_lower]["lastSeen"] = now_iso
            contacts_by_name[name_lower]["mentionCount"] = contacts_by_name[name_lower].get(
                "mentionCount", 0) + 1
        else:
            new_contact = {
                "name": name,
                "relationship": "unknown",
                "notes": f"First mentioned in capture {capture.get('id')}",
                "bestFacePhotoURL": None,
                "firstSeen": now_iso,
                "lastSeen": now_iso,
                "mentionCount": 1
            }
            contacts_list.append(new_contact)
            contacts_by_name[name_lower] = new_contact

    for face in detected_faces:
        face_name = face.get("possibleName")
        if face_name:
            name_lower = face_name.lower()
            if name_lower in contacts_by_name:
                contact = contacts_by_name[name_lower]
                if not contact.get("bestFacePhotoURL") and capture.get("photoURL"):
                    contact["bestFacePhotoURL"] = capture.get("photoURL")
                contact["lastSeen"] = now_iso

    contacts_doc["contacts"] = list(contacts_by_name.values())
    await repo.upsert_contacts(user_id, contacts_doc)
    logger.info("Updated contacts for user=%s, total=%d",
                user_id, len(contacts_doc["contacts"]))


# =============================================================================
# HOURLY CONDENSATION
# =============================================================================

async def _check_and_run_hourly_condensation(user_id: str, capture_ts: datetime) -> None:
    profile = await repo.get_user_profile(user_id)
    if not profile:
        return

    last_condensation = profile.get("lastCondensationTime")
    if last_condensation:
        if isinstance(last_condensation, str):
            last_condensation = _parse_iso_datetime(last_condensation)
        time_since = capture_ts - last_condensation
        if time_since < timedelta(hours=1):
            return

    now_est = capture_ts.astimezone(EST)
    hour_start = now_est.replace(
        minute=0, second=0, microsecond=0) - timedelta(hours=1)
    hour_end = hour_start + timedelta(hours=1) - timedelta(seconds=1)

    hour_start_utc = hour_start.astimezone(timezone.utc)
    hour_end_utc = hour_end.astimezone(timezone.utc)

    captures = await repo.list_captures_in_range(user_id, hour_start_utc, hour_end_utc)

    if not captures:
        await repo.upsert_user_profile(user_id, {"lastCondensationTime": capture_ts})
        return

    captures_for_prompt = []
    for c in captures:
        ts = c.get("timestamp")
        if isinstance(ts, datetime):
            ts = ts.isoformat()
        captures_for_prompt.append({
            "timestamp": ts,
            "transcription": c.get("transcription"),
            "analysis": c.get("geminiAnalysis", {})
        })

    # Fetch queries made during this hour
    queries_for_prompt = []
    recent_queries_doc = await repo.get_recent_queries(user_id)
    if recent_queries_doc:
        for q in recent_queries_doc.get("queries", []):
            q_ts = q.get("timestamp")
            if isinstance(q_ts, datetime):
                if hour_start_utc <= q_ts <= hour_end_utc:
                    queries_for_prompt.append({
                        "timestamp": q_ts.isoformat(),
                        "query": q.get("query"),
                        "answer": q.get("answer", "")[:300]
                    })

    try:
        prompt = HOURLY_SUMMARY_PROMPT.format(
            captures_json=json.dumps(captures_for_prompt, indent=2),
            queries_json=json.dumps(
                queries_for_prompt, indent=2) if queries_for_prompt else "No queries this hour"
        )
        summary_result = await _call_gemini_text(prompt)

        date_str = hour_start.date().isoformat()
        hour_num = hour_start.hour

        await repo.create_hourly_summary(user_id, date_str, hour_num, {
            "summary": summary_result.get("summary", ""),
            "themes": summary_result.get("themes", []),
            "events": summary_result.get("events", []),
            "peoplePresent": summary_result.get("peoplePresent", []),
            "locations": summary_result.get("locations", []),
            "highlight": summary_result.get("highlight", ""),
            "mood": summary_result.get("mood", ""),
            "activities": summary_result.get("activities", []),
            "captureIds": [c.get("id") for c in captures],
            "captureCount": len(captures)
        })

        await repo.upsert_user_profile(user_id, {
            "lastCondensationTime": capture_ts,
            "lastHourSummaryTime": capture_ts.isoformat(),
            "lastHourSummary": summary_result.get("summary", "")
        })

        logger.info("Created hourly summary for user=%s date=%s hour=%d",
                    user_id, date_str, hour_num)
    except Exception:
        logger.exception("Hourly condensation failed for user=%s", user_id)
        await repo.upsert_user_profile(user_id, {"lastCondensationTime": capture_ts})


# =============================================================================
# DAILY CONDENSATION (midnight EST check)
# =============================================================================

async def _check_and_run_daily_condensation(user_id: str, capture_ts: datetime) -> None:
    now_est = capture_ts.astimezone(EST)
    yesterday_est = (now_est - timedelta(days=1)).date()
    yesterday_str = yesterday_est.isoformat()

    existing = await repo.get_daily_summary(user_id, yesterday_str)
    if existing:
        return

    if now_est.hour < 1:
        return

    hourly_summaries = await repo.list_hourly_summaries_for_date(user_id, yesterday_str)

    if not hourly_summaries:
        return

    try:
        hourly_for_prompt = []
        for h in hourly_summaries:
            hourly_for_prompt.append({
                "hour": h.get("hour"),
                "summary": h.get("summary"),
                "themes": h.get("themes"),
                "events": h.get("events", []),
                "peoplePresent": h.get("peoplePresent", []),
                "locations": h.get("locations", []),
                "activities": h.get("activities", []),
                "mood": h.get("mood"),
                "captureCount": h.get("captureCount", 0)
            })

        prompt = DAILY_SUMMARY_PROMPT.format(
            hourly_json=json.dumps(hourly_for_prompt, indent=2))
        daily_result = await _call_gemini_text(prompt)

        await repo.create_daily_summary(user_id, yesterday_str, {
            "summary": daily_result.get("summary", ""),
            "timeline": daily_result.get("timeline", []),
            "themes": daily_result.get("themes", []),
            "highlights": daily_result.get("highlights", []),
            "mood": daily_result.get("mood", ""),
            "peopleInteractions": daily_result.get("peopleInteractions", []),
            "locations": daily_result.get("locations", []),
            "accomplishments": daily_result.get("accomplishments", []),
            "morningOverview": daily_result.get("morningOverview", ""),
            "afternoonOverview": daily_result.get("afternoonOverview", ""),
            "eveningOverview": daily_result.get("eveningOverview", ""),
            "hourlyIds": [f"hourly_{user_id}_{yesterday_str}_{h.get('hour', 0):02d}" for h in hourly_summaries],
            "totalCaptures": sum(h.get("captureCount", 0) for h in hourly_summaries)
        })

        await repo.upsert_user_profile(user_id, {
            "lastDaySummaryDate": yesterday_str,
            "lastDaySummary": daily_result.get("summary", "")
        })

        logger.info("Created daily summary for user=%s date=%s",
                    user_id, yesterday_str)
    except Exception:
        logger.exception("Daily condensation failed for user=%s", user_id)


# =============================================================================
# MAIN CAPTURE PROCESSING PIPELINE
# =============================================================================

async def _process_capture_async(user_id: str, capture_id: str, capture_ts: datetime) -> None:
    try:
        capture_doc = await repo.get_capture(capture_id)
        if not capture_doc:
            logger.error("Capture not found: %s", capture_id)
            return

        analysis = await _analyze_capture_with_gemini(capture_doc)

        await repo.update_capture(capture_id, {
            "processed": True,
            "geminiAnalysis": analysis,
        })

        await _update_contacts_from_analysis(user_id, analysis, capture_doc)

        await _check_and_run_hourly_condensation(user_id, capture_ts)
        await _check_and_run_daily_condensation(user_id, capture_ts)

        day_key = _date_key_from_dt(capture_ts)
        await manager.broadcast_unity(user_id, {
            "type": "memory_processed",
            "date": day_key,
            "captureId": capture_id,
        })

        logger.info("Processed capture=%s user=%s", capture_id, user_id)
    except Exception:
        logger.exception(
            "Processing failed for capture=%s user=%s", capture_id, user_id)
        try:
            await repo.update_capture(capture_id, {
                "processed": False,
                "geminiAnalysis": {"error": "processing_failed"}
            })
        except Exception:
            logger.exception("Failed to mark capture processing failure")


@app.websocket("/ws/ios/{user_id}")
async def ws_ios(websocket: WebSocket, user_id: str) -> None:
    await manager.connect("ios", user_id, websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(
                    "[SEND_DATA] user=%s INVALID_JSON raw=%s", user_id, raw[:500])
                await manager.send_json(websocket, {"ok": False, "error": "invalid_json"})
                continue

            try:
                # Log raw message structure for debugging
                msg_keys = list(msg.keys())
                transcription = msg.get("transcription") or ""
                photo_url = msg.get("photoURL") or ""
                audio_url = msg.get("audioURL") or ""

                logger.info(
                    "[SEND_DATA] user=%s type=%s keys=%s transcription_len=%d photo=%s audio=%s",
                    user_id,
                    msg.get("type"),
                    msg_keys,
                    len(transcription),
                    "yes" if photo_url else "no",
                    "yes" if audio_url else "no"
                )

                # Log transcription content if present
                if transcription:
                    logger.info(
                        "[SEND_DATA] user=%s transcription_preview=%s", user_id, transcription[:300])
                else:
                    logger.warning(
                        "[SEND_DATA] user=%s NO_TRANSCRIPTION in message", user_id)

                if msg.get("type") != "memory_capture":
                    await manager.send_json(
                        websocket,
                        {
                            "type": "error",
                            "status": "invalid_type",
                            "detail": "Expected type=memory_capture",
                        },
                    )
                    continue

                capture_id = str(msg.get("id") or uuid.uuid4())
                ts = _parse_iso_datetime(msg.get("timestamp"))

                doc = {
                    "id": capture_id,
                    "userId": user_id,
                    "timestamp": ts,
                    "photoURL": msg.get("photoURL"),
                    "audioURL": msg.get("audioURL"),
                    "transcription": msg.get("transcription"),
                    "processed": False,
                    "geminiAnalysis": None,
                }

                await repo.create_capture(doc)

                ack_ts = ts.astimezone(
                    timezone.utc).isoformat().replace("+00:00", "Z")
                await manager.send_json(
                    websocket,
                    {
                        "type": "ack",
                        "status": "received",
                        "captureId": capture_id,
                        "timestamp": ack_ts,
                    },
                )

                # Non-blocking processing
                asyncio.create_task(
                    _process_capture_async(user_id, capture_id, ts))

            except Exception as e:
                logger.exception(
                    "Failed to handle iOS message user=%s", user_id)
                await manager.send_json(
                    websocket,
                    {
                        "ok": False,
                        "error": "failed_to_save",
                        "detail": str(e),
                    },
                )

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("iOS websocket error user=%s", user_id)
    finally:
        await manager.disconnect("ios", user_id, websocket)


def _serialize_capture(doc: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(doc)
    ts = out.get("timestamp")
    if isinstance(ts, datetime):
        out["timestamp"] = ts.astimezone(
            timezone.utc).isoformat().replace("+00:00", "Z")
    return out


# =============================================================================
# UNITY QUERY PIPELINE
# =============================================================================

async def _process_unity_query(user_id: str, query_text: str, date_range: Optional[Dict], include_faces: bool, max_images: int, user_image: Optional[str] = None) -> Dict[str, Any]:
    profile = await repo.get_user_profile(user_id)
    contacts_doc = await repo.get_contacts(user_id)
    recent_queries_doc = await repo.get_recent_queries(user_id)

    profile_str = json.dumps(
        profile, default=str) if profile else "No profile available"
    contacts_list = contacts_doc.get("contacts", []) if contacts_doc else []
    contacts_summary = ", ".join(
        [f"{c.get('name')} ({c.get('relationship')})" for c in contacts_list[:20]]) or "No contacts"

    # Format recent queries for conversation context with formatted timestamps
    now = datetime.now(timezone.utc)
    recent_queries_list = []
    if recent_queries_doc:
        for q in recent_queries_doc.get("queries", [])[:5]:
            q_ts = q.get("timestamp")
            if isinstance(q_ts, datetime):
                formatted_ts = _format_query_timestamp(q_ts, now)
            else:
                formatted_ts = str(q_ts)
            recent_queries_list.append({
                "query": q.get("query"),
                "answer": q.get("answer", "")[:200],
                "timestamp": formatted_ts
            })
    recent_queries_str = json.dumps(
        recent_queries_list, default=str, indent=2) if recent_queries_list else "No recent queries"

    last_hour = profile.get(
        "lastHourSummary", "No recent hourly summary") if profile else "No data"
    last_day = profile.get(
        "lastDaySummary", "No recent daily summary") if profile else "No data"

    date_range_str = "Not specified"
    if date_range:
        date_range_str = f"{date_range.get('start', 'any')} to {date_range.get('end', 'any')}"

    router_prompt = QUERY_ROUTER_PROMPT.format(
        user_profile=profile_str,
        contacts_summary=contacts_summary,
        last_hour_summary=last_hour,
        last_day_summary=last_day,
        recent_queries=recent_queries_str,
        query_text=query_text,
        date_range=date_range_str
    )

    try:
        router_result = await _call_gemini_text(router_prompt)
    except Exception as e:
        logger.exception("Query router failed")
        return {"ok": False, "error": "query_routing_failed", "detail": str(e)}

    if router_result.get("needsClarification"):
        # Return clarification as a regular response - user can follow up with new query
        # Recent queries context will maintain conversation continuity
        return {
            "type": "query_answer",
            "ok": True,
            "answer": router_result.get("clarificationQuestion", "Could you please clarify your question?"),
            "confidence": 0.5,
            "sources": [],
            "relatedContacts": [],
            "attachedImages": []
        }

    # Check if router needs more query context
    more_context = router_result.get("needsMoreQueryContext", 0)
    if more_context and more_context > 5 and recent_queries_doc:
        extended_count = min(more_context, 20)
        recent_queries_list = []
        for q in recent_queries_doc.get("queries", [])[:extended_count]:
            q_ts = q.get("timestamp")
            if isinstance(q_ts, datetime):
                formatted_ts = _format_query_timestamp(q_ts, now)
            else:
                formatted_ts = str(q_ts)
            recent_queries_list.append({
                "query": q.get("query"),
                "answer": q.get("answer", "")[:200],
                "timestamp": formatted_ts
            })
        recent_queries_str = json.dumps(
            recent_queries_list, default=str, indent=2)
        logger.info("[QUERY_DATA] Extended query context to %d queries", len(
            recent_queries_list))

    memory_context = []

    data_needed = router_result.get("dataNeeded", {})
    specific_dates = data_needed.get("specificDates", [])

    if date_range:
        try:
            start_date = datetime.fromisoformat(
                date_range.get("start", "")).date()
            end_date = datetime.fromisoformat(date_range.get("end", "")).date()
            current = start_date
            while current <= end_date:
                specific_dates.append(current.isoformat())
                current += timedelta(days=1)
        except Exception:
            pass

    if not specific_dates:
        today = datetime.now(EST).date()
        specific_dates = [
            today.isoformat(), (today - timedelta(days=1)).isoformat()]

    specific_dates = list(set(specific_dates))[:7]

    if data_needed.get("dailySummaries"):
        for date_str in specific_dates:
            summary = await repo.get_daily_summary(user_id, date_str)
            if summary:
                memory_context.append(
                    {"type": "daily_summary", "date": date_str, "data": summary})

    if data_needed.get("hourlySummaries"):
        for date_str in specific_dates:
            hourly = await repo.list_hourly_summaries_for_date(user_id, date_str)
            for h in hourly:
                memory_context.append(
                    {"type": "hourly_summary", "date": date_str, "hour": h.get("hour"), "data": h})

    if data_needed.get("captures"):
        for date_str in specific_dates:
            try:
                day = datetime.fromisoformat(date_str).date()
                captures = await repo.list_captures_for_date(user_id, day)
                for c in captures[:10]:
                    analysis = c.get("geminiAnalysis")
                    if analysis:
                        memory_context.append({
                            "type": "capture",
                            "id": c.get("id"),
                            "timestamp": str(c.get("timestamp")),
                            "transcription": c.get("transcription"),
                            "analysis": analysis
                        })
                    else:
                        # Unprocessed capture - use raw transcription
                        memory_context.append({
                            "type": "recent_capture",
                            "id": c.get("id"),
                            "timestamp": str(c.get("timestamp")),
                            "transcription": c.get("transcription") or "(no audio)",
                            "status": "processing"
                        })
            except Exception:
                pass

    relevant_contacts = router_result.get("relevantContacts", [])
    contacts_info = []
    attached_images = []

    if include_faces and router_result.get("needsFaceImages"):
        for contact_name in relevant_contacts[:max_images]:
            for c in contacts_list:
                if c.get("name", "").lower() == contact_name.lower():
                    contacts_info.append(c)
                    if c.get("bestFacePhotoURL"):
                        attached_images.append(c.get("bestFacePhotoURL"))
                    break

    attached_images = attached_images[:max_images]

    answer_prompt = QUERY_ANSWER_PROMPT.format(
        user_profile=profile_str,
        recent_queries=recent_queries_str,
        memory_context=json.dumps(memory_context, default=str, indent=2),
        contacts_info=json.dumps(contacts_info, default=str),
        current_time=now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        query_text=query_text
    )

    try:
        # Use vision model if user provided an image with their query
        if user_image:
            logger.info(
                "[QUERY_DATA] Processing query with user-provided image")
            answer_result = await _call_gemini_with_image(answer_prompt, user_image)
        else:
            answer_result = await _call_gemini_text(answer_prompt)
    except Exception as e:
        logger.exception("Query answer generation failed")
        return {"ok": False, "error": "answer_generation_failed", "detail": str(e)}

    source_ids = answer_result.get("sourceCaptureIds", [])
    sources = []
    for sid in source_ids[:5]:
        for mc in memory_context:
            if mc.get("id") == sid:
                sources.append({
                    "captureId": sid,
                    "timestamp": mc.get("timestamp"),
                    "summary": mc.get("analysis", {}).get("keyMoment", "")
                })
                break

    return {
        "type": "response",
        "ok": True,
        "answer": answer_result.get("answer", "I couldn't find relevant information."),
        "confidence": answer_result.get("confidence", 0.5),
        "sources": sources,
        "relatedContacts": [{"name": c.get("name"), "relationship": c.get("relationship"), "faceImageURL": c.get("bestFacePhotoURL")} for c in contacts_info],
        "attachedImages": attached_images,
        "suggestedFollowUp": answer_result.get("suggestedFollowUp")
    }


# =============================================================================
# /ws/query/{user_id} - Dedicated Query WebSocket
# =============================================================================
# Image Flow (optional):
#   1. Client uploads image: POST /query-upload/{query_id} → returns {url}
#   2. Client sends query: {"text": "...", "imageURL": url, ...}
#
# Response Flow:
#   1. Client sends: {"text": "query", "imageURL": "https://...", "dateRange": {...}}
#   2. Server processes with Gemini (vision model if imageURL provided)
#   3. Server sends back ONE of:
#      - {"type": "response", "ok": true, "answer": "...", ...}  (final answer)
#      - {"type": "clarification_needed", "ok": true, "message": "..."}  (needs more info)
#      - {"type": "error", "ok": false, "error": "...", "detail": "..."}  (error)
#   4. If clarification was requested, client sends: {"text": "clarification response"}
#   5. Server sends final response
# =============================================================================

@app.websocket("/ws/query/{user_id}")
async def ws_query(websocket: WebSocket, user_id: str) -> None:
    await manager.connect("unity", user_id, websocket)
    logger.info("Query WebSocket connected user=%s", user_id)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                req = json.loads(raw)
            except json.JSONDecodeError:
                await manager.send_json(websocket, {"type": "error", "ok": False, "error": "invalid_json"})
                continue

            query_text = req.get("text", "")
            if not query_text:
                await manager.send_json(websocket, {"type": "error", "ok": False, "error": "missing_text"})
                continue

            try:
                date_range = req.get("dateRange")
                include_faces = req.get("includeFaces", True)
                max_images = min(
                    req.get("maxImages", DEFAULT_QUERY_IMAGES), MAX_QUERY_IMAGES)
                # Optional: URL from POST /query-upload
                image_url = req.get("imageURL")

                # Generate unique query ID
                query_id = str(uuid.uuid4())[:8]

                # Download image if URL provided
                user_image = None
                if image_url:
                    user_image = await _download_image_as_base64(image_url)

                # Log incoming query content
                logger.info(
                    "[QUERY_DATA] user=%s query_id=%s query=%s dateRange=%s includeFaces=%s imageURL=%s",
                    user_id,
                    query_id,
                    query_text[:300],
                    json.dumps(date_range) if date_range else "none",
                    include_faces,
                    image_url[:80] if image_url else "none"
                )

                start_time = time.time()
                result = await _process_unity_query(user_id, query_text, date_range, include_faces, max_images, user_image)
                elapsed_ms = (time.time() - start_time) * 1000

                # Add queryId to result
                result["queryId"] = query_id

                # Generate TTS audio for successful responses (non-blocking on failure)
                audio_url = None
                if result.get("ok") and result.get("answer"):
                    try:
                        audio_url = await tts_service.generate_speech(
                            result["answer"],
                            query_id,
                            user_id
                        )
                    except Exception as tts_err:
                        logger.warning(
                            "TTS failed for query_id=%s: %s", query_id, tts_err)

                # Add audioURL to result (null if TTS failed or unavailable)
                result["audioURL"] = audio_url

                # Log response summary
                logger.info(
                    "[QUERY_DATA] user=%s query_id=%s response_ok=%s confidence=%s elapsed_ms=%.0f audioURL=%s answer_preview=%s",
                    user_id,
                    query_id,
                    result.get("ok"),
                    result.get("confidence", "n/a"),
                    elapsed_ms,
                    "yes" if audio_url else "no",
                    (result.get("answer") or "")[:150]
                )

                # Save query to recent queries
                await repo.add_recent_query(user_id, {
                    "query": query_text,
                    "queryId": query_id,
                    "imageURL": image_url,
                    "dateRange": date_range,
                    "responseType": result.get("type"),
                    # Truncate for storage
                    "answer": result.get("answer", "")[:500],
                    "audioURL": audio_url,
                    "confidence": result.get("confidence"),
                    "elapsed_ms": round(elapsed_ms),
                    "ok": result.get("ok", False)
                })

                await manager.send_json(websocket, result)

            except Exception as e:
                logger.exception("Query failed user=%s", user_id)
                await manager.send_json(websocket, {
                    "type": "error",
                    "ok": False,
                    "error": "query_failed",
                    "detail": str(e)
                })

    except WebSocketDisconnect:
        logger.info("Query WebSocket disconnected user=%s", user_id)
    except Exception:
        logger.exception("Query websocket error user=%s", user_id)
    finally:
        await manager.disconnect("unity", user_id, websocket)


# =============================================================================
# /ws/unity/{user_id} - Legacy Unity WebSocket (fetch daily memories)
# =============================================================================

@app.websocket("/ws/unity/{user_id}")
async def ws_unity(websocket: WebSocket, user_id: str) -> None:
    await manager.connect("unity", user_id, websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                req = json.loads(raw)
            except json.JSONDecodeError:
                await manager.send_json(websocket, {"ok": False, "error": "invalid_json"})
                continue

            rtype = req.get("type")

            if rtype == "fetch_daily_memories":
                try:
                    date_str = req.get("date")
                    if not isinstance(date_str, str):
                        raise ValueError("date must be a string YYYY-MM-DD")
                    day = datetime.fromisoformat(date_str).date()
                    date_key = day.isoformat()

                    captures = await repo.list_captures_for_date(user_id, day)

                    daily = await repo.get_daily_summary(user_id, date_key)
                    if daily:
                        summary = daily.get("summary", "")
                        themes = daily.get("themes", [])
                        timeline = daily.get("timeline", [])
                    else:
                        processed = await repo.get_processed_memory(user_id, date_key)
                        if processed:
                            summary = processed.get("summary", "")
                            themes = processed.get("themes", [])
                            timeline = []
                        else:
                            summary = f"You had {len(captures)} captures on this day."
                            themes = []
                            timeline = []

                    await manager.send_json(websocket, {
                        "ok": True,
                        "type": "daily_memories",
                        "date": date_key,
                        "summary": summary,
                        "themes": themes,
                        "timeline": timeline,
                        "captures": [_serialize_capture(c) for c in captures],
                        "totalCaptures": len(captures),
                    })
                except Exception as e:
                    logger.exception(
                        "Unity fetch_daily_memories failed user=%s", user_id)
                    await manager.send_json(websocket, {
                        "ok": False,
                        "type": "daily_memories",
                        "error": "fetch_failed",
                        "detail": str(e),
                    })
            else:
                await manager.send_json(websocket, {
                    "ok": False,
                    "error": "unknown_request_type",
                    "detail": f"type={rtype}. Use /ws/query for queries.",
                })

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Unity websocket error user=%s", user_id)
    finally:
        await manager.disconnect("unity", user_id, websocket)

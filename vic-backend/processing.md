# Gemini Processing Pipeline

This document explains how memory captures are analyzed and processed using Gemini AI.

---

## Overview

When a memory capture arrives, it triggers an async processing pipeline that:
1. Analyzes the image with Gemini Vision
2. Extracts themes and mood from transcription + image
3. Detects and identifies faces (matching to known contacts)
4. Updates the contacts database with new/updated people
5. Checks if hourly condensation should run
6. Checks if daily condensation should run (midnight EST)

---

## System Prompts

### CAPTURE_ANALYSIS_PROMPT
Used when analyzing individual memory captures.

```
You are analyzing a memory capture for a memory assistance application.
The user relies on this app to remember their daily experiences.

Given:
- An image (if provided)
- A transcription of audio (if provided)
- The timestamp of when this was captured

Analyze and return a JSON object with:
{
  "imageSummary": "Brief description of what's shown in the image",
  "themes": ["theme1", "theme2"],  // 2-5 relevant themes
  "mood": "the emotional tone (e.g., happy, focused, relaxed, stressed)",
  "location": "best guess of location type (e.g., home, office, cafe, outdoors)",
  "detectedFaces": [
    {
      "description": "physical description of person",
      "possibleName": "name if mentioned in transcription, otherwise null",
      "confidence": 0.0-1.0
    }
  ],
  "mentionedNames": ["names mentioned in transcription"],
  "keyMoment": "one sentence capturing the essence of this memory"
}

Be concise but informative. Focus on details that would help someone recall this moment later.
```

### HOURLY_SUMMARY_PROMPT
Used when condensing captures from the past hour. Handles 60+ captures per hour.

```
You are creating an hourly summary for a memory assistance application.
NOTE: There may be 60+ memory captures per hour - this is normal.

Return a JSON object with:
{
  "summary": "5-10 sentence detailed narrative of the hour",
  "themes": ["theme1", "theme2", "theme3", "theme4", "theme5"],
  "events": [
    {"time": "HH:MM", "description": "what happened"}
  ],
  "peoplePresent": ["names of people seen or mentioned"],
  "locations": ["places visited this hour"],
  "highlight": "the most notable moment from this hour",
  "mood": "overall emotional tone of the hour",
  "activities": ["list of distinct activities performed"]
}

Be warm and personal. Include enough detail to vividly recall the hour.
```

### DAILY_SUMMARY_PROMPT
Used when condensing hourly summaries into a daily summary at midnight EST. Handles hundreds of captures per day.

```
You are creating a daily summary for a memory assistance application.
NOTE: Each hour may contain 60+ captures, so a full day could have hundreds of moments.

Return a JSON object with:
{
  "summary": "10-15 sentence detailed narrative of the entire day",
  "timeline": [
    {"time": "HH:MM", "event": "significant event or activity"}
  ],
  "themes": ["theme1", "theme2", "theme3", "theme4", "theme5", "theme6"],
  "highlights": [
    {"time": "HH:MM", "description": "highlight of the day"}
  ],
  "mood": "emotional arc - how mood shifted throughout the day",
  "peopleInteractions": [
    {"name": "person name", "context": "how/when they interacted"}
  ],
  "locations": ["all locations visited today"],
  "accomplishments": ["things completed or achieved today"],
  "morningOverview": "2-3 sentences about the morning",
  "afternoonOverview": "2-3 sentences about the afternoon",
  "eveningOverview": "2-3 sentences about the evening"
}

Write in a warm, reflective tone. Help the user vividly remember their entire day.
```

### CONTACT_UPDATE_PROMPT
Used when updating the contacts database with detected/mentioned people.

```
You are managing a contacts database for a memory assistance application.

Given:
- Current contacts list
- New information from a memory capture (detected faces, mentioned names)

Determine what updates to make:
1. If a name is mentioned and matches an existing contact, update lastSeen
2. If a new name is mentioned, add a new contact entry
3. If a face is detected with a name, associate them

Return a JSON object with:
{
  "updates": [
    {
      "action": "update" or "create",
      "name": "contact name",
      "changes": { fields to update }
    }
  ],
  "needsMoreInfo": false,  // true if face detected but no name
  "unknownFaceDescription": "description if needsMoreInfo is true"
}
```

---

## Processing Flow

### Step 1: Capture Analysis
```python
async def _analyze_capture_with_gemini(capture: Dict) -> Dict:
    # 1. Download image from GCS if photoURL exists
    # 2. Build prompt with image + transcription + timestamp
    # 3. Call Gemini Vision API
    # 4. Parse JSON response
    # 5. Return analysis dict
```

### Step 2: Contact Detection & Update
```python
async def _update_contacts_from_analysis(user_id: str, analysis: Dict, capture: Dict):
    # 1. Get current contacts document
    # 2. Check mentionedNames against existing contacts
    # 3. Check detectedFaces for matches
    # 4. Call Gemini with CONTACT_UPDATE_PROMPT if needed
    # 5. Apply updates to contacts document
    # 6. If face detected with no name, store for later association
```

### Step 3: Hourly Condensation Check
```python
async def _check_hourly_condensation(user_id: str, capture_ts: datetime):
    # 1. Get user profile
    # 2. Check lastCondensationTime
    # 3. If > 1 hour since last condensation:
    #    a. Fetch all captures from last hour
    #    b. Call Gemini with HOURLY_SUMMARY_PROMPT
    #    c. Create hourly_summaries document
    #    d. Update user profile with lastHourSummary
    #    e. Update lastCondensationTime
```

### Step 4: Daily Condensation Check
```python
async def _check_daily_condensation(user_id: str, capture_ts: datetime):
    # 1. Convert to EST timezone
    # 2. Check if it's a new day (past midnight EST)
    # 3. Check if yesterday's daily summary exists
    # 4. If not:
    #    a. Fetch all hourly summaries for yesterday
    #    b. Call Gemini with DAILY_SUMMARY_PROMPT
    #    c. Create daily_summaries document
    #    d. Update user profile with lastDaySummary
```

---

## Gemini API Configuration

```python
# Model used for all processing
MODEL_NAME = "gemini-2.0-flash-exp"

# Configuration
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "max_output_tokens": 2048,
}

# For vision tasks (image analysis)
# Images are passed as base64 or GCS URIs
```

---

## Error Handling

| Error | Handling |
|-------|----------|
| Gemini API timeout | Retry up to 3 times with exponential backoff |
| Invalid JSON response | Log error, use fallback analysis |
| Image download failed | Proceed with transcription-only analysis |
| Rate limit exceeded | Queue for later processing |

On failure, the capture is marked:
```json
{
  "processed": false,
  "geminiAnalysis": {"error": "processing_failed", "detail": "..."}
}
```

---

## Face Handling

### When a face is detected with a name:
1. Check if contact exists → update `lastSeen`
2. If new contact → create entry
3. If image quality is good → save as `bestFacePhotoURL`

### When a face is detected without a name:
1. Store face description in analysis
2. Mark as "unknown face" 
3. If later a name is associated (via transcription), link them

### Best Face Photo Selection:
- Prefer frontal, well-lit faces
- Store in `reality-hack-2026-processed-media/faces/{user_id}/{contact_name}.jpg`
- Update contact's `bestFacePhotoURL` field

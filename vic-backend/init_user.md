# User Initialization Guide

Initialize a new user profile via a single `curl` command.

```bash
curl -X POST "${API_URL:-http://localhost:8000}/init-user/<user_id>" \
  -H "Content-Type: application/json" \
  -d '{"summary":"<freeform user summary>"}'
```
## Example (cass)

```bash
curl -X POST "https://memory-backend-328251955578.us-east1.run.app/init-user/cass" \
  -H "Content-Type: application/json" \
  -d '{"summary":"Cass Manchado is a 72-year-old retired software developer living with early-stage Alzheimer’s. He is a widower; his wife passed away three years ago. He lives alone, and his daughter and granddaughter visit weekly. He is gentle, thoughtful, and slightly humorous, and enjoys telling small stories and teasing younger people. He loves chocolate, especially anything with nuts, and jokes that ‘chocolate is better than medicine’. He always hides chocolate in the second kitchen drawer. His most precious object is a book about Alzheimer’s given by his late wife, with a note that said, ‘We’ll face this together.’ He calls his wife ‘my clever girl’ when remembering her, and sometimes talks to her quietly when he is alone. He often leaves his keys on the bookshelf instead of the key hook (8 recorded times this year). When anxious, he rereads the same page. He drinks tea every afternoon at 4pm."}'
```

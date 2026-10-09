# Saborea frontend

Next.js, React, and TypeScript interface for `POST /agents/restaurant-search`.
The chat uses assistant-ui with a local runtime; shadcn/ui components and
Tailwind CSS v4 provide the visual foundation. Restaurant candidates and
credited photographs remain rendered from the response's structured fields.

## Run locally

Start the backend from `backend`:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

The backend needs its normal database and model configuration. Search requests
can call Gemini and, when configured, the OpenAI fallback; they may incur
provider charges. The frontend does not make a model request until a chat
message is submitted.

From `frontend`, install dependencies and start Next.js:

```powershell
npm install
Copy-Item .env.example .env.local
npm run dev
```

Open `http://localhost:3000`. `BACKEND_API_URL` defaults to
`http://127.0.0.1:8000`; change it in `.env.local` when the backend runs at a
different address.

The Next.js route handler proxies search requests server-side, so the browser
does not need cross-origin access to FastAPI and the backend CORS policy remains
unchanged. The proxy validates the request size and passes the backend response
status through without logging query content.

Each chat turn sends only the latest user message as an independent search.
Assistant-ui keeps the visible conversation in memory for the current page;
messages are not persisted, and previous turns are not sent to the backend.
Agent answers are Markdown rendered with `react-markdown`; raw HTML is not
enabled.

`components.json` configures shadcn/ui aliases and Tailwind CSS. Reusable
shadcn-style components live in `src/components/ui`. The chat UI is grouped in
`src/components/restaurant-chat`, result cards in
`src/components/restaurant-results`, the assistant-ui runtime adapter in
`src/hooks/use-restaurant-search-chat.ts`, and API validation/request handling
in `src/lib/restaurant-search.ts`.

Run the frontend checks with:

```powershell
npm run lint
npx tsc --noEmit --incremental false
npm run build
```

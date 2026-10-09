# Saborea frontend

Next.js, React, and TypeScript interface for `POST /agents/restaurant-search`.
It shows the Agent's answer, restaurant candidates, and credited photographs.

## Run locally

Start the backend from `backend`:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

The backend needs its normal database and model configuration. Search requests
can call Gemini and, when configured, the OpenAI fallback; they may incur
provider charges. The frontend does not make a model request until the user
submits the form.

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

Agent answers are Markdown and are rendered with `react-markdown`; raw HTML is
not enabled. Restaurant candidates and photo credits are rendered from their
separate structured response fields.

Run the frontend checks with:

```powershell
npm run lint
npm run build
```

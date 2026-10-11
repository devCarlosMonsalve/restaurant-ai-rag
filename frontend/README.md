# Saborea — Frontend

The web interface for Restaurant AI RAG, built with Next.js, React, and
TypeScript. It provides a restaurant-search chat backed by the FastAPI
`POST /agents/restaurant-search` endpoint.

## Features

- assistant-ui chat runtime with in-page conversation state.
- Structured rendering of restaurant candidates and credited photographs.
- A server-side Next.js route that validates requests and proxies them to the
  backend.
- Markdown rendering with `react-markdown`; raw HTML is not enabled.
- shadcn/ui-style components and Tailwind CSS v4.

Each submitted turn is sent as a standalone search containing only the latest
user message. The visible conversation is kept in memory for the current page;
it is not persisted, and earlier turns are not sent to the backend.

## Requirements

- Node.js and npm versions compatible with the installed Next.js release.
- A running backend for search requests. See the
  [backend setup guide](../backend/README.md).

Search requests may call Gemini and, if configured, the OpenAI fallback. These
requests may incur provider charges; no model request is made until the user
submits a chat message.

## Run locally

From this directory:

```powershell
npm ci
if (-not (Test-Path .env.local)) { Copy-Item .env.example .env.local }
npm run dev
```

Open `http://localhost:3000`. `BACKEND_API_URL` defaults to
`http://127.0.0.1:8000`; set it in `.env.local` when the backend runs elsewhere.

The proxy passes the backend response status through and does not log query
content. Because the browser calls the same-origin Next.js route, it does not
need direct cross-origin access to FastAPI and the backend CORS policy remains
unchanged.

## Project structure

| Path | Responsibility |
| --- | --- |
| `src/app/` | Next.js pages, layout, and route handlers. |
| `src/app/api/restaurant-search/` | Input validation and server-side backend proxy. |
| `src/components/restaurant-chat/` | Chat interface. |
| `src/components/restaurant-results/` | Structured restaurant and photo result cards. |
| `src/components/ui/` | Shared UI components. |
| `src/hooks/use-restaurant-search-chat.ts` | assistant-ui runtime adapter. |
| `src/lib/restaurant-search.ts` | Frontend request and response validation. |

`components.json` configures the component aliases and Tailwind integration.

## Quality checks

```powershell
npm run lint
npx tsc --noEmit --incremental false
npm run build
```

The package currently defines lint, development, build, and production-start
scripts; it does not define a separate test script.

## Production

Build and start the production server with:

```powershell
npm run build
npm run start
```

Set `BACKEND_API_URL` in the server environment to the reachable FastAPI URL.
Keep provider credentials on the backend; the frontend only needs the backend
base URL.

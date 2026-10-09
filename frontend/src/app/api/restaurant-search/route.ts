function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function jsonError(detail: string, status: number): Response {
  return Response.json({ detail }, { status });
}

export async function POST(request: Request): Promise<Response> {
  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return jsonError("El cuerpo de la solicitud debe ser JSON válido.", 400);
  }

  if (!isRecord(payload) || typeof payload.query !== "string") {
    return jsonError("La solicitud debe incluir una consulta de texto.", 400);
  }

  const query = payload.query.trim();
  if (!query || query.length > 2000) {
    return jsonError(
      "La consulta debe tener entre 1 y 2000 caracteres.",
      400,
    );
  }

  try {
    const backendUrl = process.env.BACKEND_API_URL ?? "http://127.0.0.1:8000";
    const backendResponse = await fetch(
      new URL("/agents/restaurant-search", backendUrl),
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
        cache: "no-store",
      },
    );

    return new Response(backendResponse.body, {
      status: backendResponse.status,
      headers: {
        "Content-Type":
          backendResponse.headers.get("Content-Type") ?? "application/json",
      },
    });
  } catch (error) {
    console.error(
      "Restaurant search backend request failed",
      error instanceof Error ? error.name : "UnknownError",
    );
    return jsonError(
      "No se pudo conectar con el backend de búsqueda.",
      502,
    );
  }
}

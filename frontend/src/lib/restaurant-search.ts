export type RestaurantCandidate = {
  name: string;
  city: string;
  cuisine: string | null;
  location: string | null;
  latitude: number | null;
  longitude: number | null;
  features: string[];
  source_url: string;
  attribution: string;
  attribution_url: string;
  similarity: number;
};

export type RestaurantPhoto = {
  image_url: string;
  source_url: string | null;
  license_name: string | null;
  license_url: string | null;
  attribution: string | null;
  restaurant_name: string | null;
  restaurant_location: string | null;
  restaurant_cuisine: string | null;
  restaurant_source_url: string | null;
  restaurant_attribution: string | null;
  restaurant_attribution_url: string | null;
};

export type RestaurantSearchResponse = {
  answer: string;
  restaurants: RestaurantCandidate[];
  photos: RestaurantPhoto[];
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNullableString(value: unknown): value is string | null {
  return value === null || typeof value === "string";
}

function isRestaurantCandidate(value: unknown): value is RestaurantCandidate {
  if (!isRecord(value)) {
    return false;
  }
  return (
    typeof value.name === "string" &&
    typeof value.city === "string" &&
    isNullableString(value.cuisine) &&
    isNullableString(value.location) &&
    (typeof value.latitude === "number" || value.latitude === null) &&
    (typeof value.longitude === "number" || value.longitude === null) &&
    Array.isArray(value.features) &&
    value.features.every((feature) => typeof feature === "string") &&
    typeof value.source_url === "string" &&
    typeof value.attribution === "string" &&
    typeof value.attribution_url === "string" &&
    typeof value.similarity === "number"
  );
}

function isRestaurantPhoto(value: unknown): value is RestaurantPhoto {
  if (!isRecord(value)) {
    return false;
  }
  return (
    typeof value.image_url === "string" &&
    isNullableString(value.source_url) &&
    isNullableString(value.license_name) &&
    isNullableString(value.license_url) &&
    isNullableString(value.attribution) &&
    isNullableString(value.restaurant_name) &&
    isNullableString(value.restaurant_location) &&
    isNullableString(value.restaurant_cuisine) &&
    isNullableString(value.restaurant_source_url) &&
    isNullableString(value.restaurant_attribution) &&
    isNullableString(value.restaurant_attribution_url)
  );
}

function isRestaurantSearchResponse(
  value: unknown,
): value is RestaurantSearchResponse {
  return (
    isRecord(value) &&
    typeof value.answer === "string" &&
    Array.isArray(value.restaurants) &&
    value.restaurants.every(isRestaurantCandidate) &&
    Array.isArray(value.photos) &&
    value.photos.every(isRestaurantPhoto)
  );
}

function errorDetail(value: unknown): string | null {
  if (isRecord(value) && typeof value.detail === "string") {
    return value.detail;
  }
  return null;
}

export async function searchRestaurants(
  query: string,
  signal: AbortSignal,
): Promise<RestaurantSearchResponse> {
  let response: Response;
  try {
    response = await fetch("/api/restaurant-search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
      signal,
    });
  } catch (cause) {
    if (signal.aborted) {
      throw cause;
    }
    throw new Error(
      "No se pudo conectar con el backend. Comprueba que esté iniciado.",
      { cause },
    );
  }

  let payload: unknown;
  try {
    payload = await response.json();
  } catch (cause) {
    if (signal.aborted) {
      throw cause;
    }
    throw new Error("La API devolvió una respuesta que no se pudo leer.", {
      cause,
    });
  }

  if (!response.ok) {
    throw new Error(
      errorDetail(payload) ??
        `La búsqueda no pudo completarse (HTTP ${response.status}).`,
    );
  }
  if (!isRestaurantSearchResponse(payload)) {
    throw new Error("La API devolvió resultados con un formato inesperado.");
  }

  return payload;
}

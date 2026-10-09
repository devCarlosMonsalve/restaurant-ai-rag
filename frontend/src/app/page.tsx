"use client";

import Image from "next/image";
import Link from "next/link";
import { type FormEvent, useState } from "react";

type RestaurantCandidate = {
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

type RestaurantPhoto = {
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

type RestaurantSearchResponse = {
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

export default function Home() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<RestaurantSearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const normalizedQuery = query.trim();
    if (!normalizedQuery) {
      setError("Escribe qué te apetece antes de buscar.");
      return;
    }

    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const response = await fetch("/api/restaurant-search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: normalizedQuery }),
      });

      let payload: unknown;
      try {
        payload = await response.json();
      } catch {
        setError("La API devolvió una respuesta que no se pudo leer.");
        return;
      }

      if (!response.ok) {
        setError(
          errorDetail(payload) ??
            `La búsqueda no pudo completarse (HTTP ${response.status}).`,
        );
        return;
      }
      if (!isRestaurantSearchResponse(payload)) {
        setError("La API devolvió resultados con un formato inesperado.");
        return;
      }
      setResult(payload);
    } catch {
      setError(
        "No se pudo conectar con el backend. Comprueba que esté iniciado.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page-shell">
      <header className="hero">
        <Link className="brand" href="/" aria-label="Saborea, inicio">
          <span className="brand-mark" aria-hidden="true">
            S
          </span>
          <span>saborea</span>
        </Link>
        <p className="eyebrow">BUEN COMER, CON BUENAS FUENTES</p>
        <h1>
          ¿Qué te apetece
          <br />
          <span>hoy?</span>
        </h1>
        <p className="intro">
          Cuéntanos el plato, el ambiente o el tipo de cocina que buscas.
          Encontraremos restaurantes reales y fotos con su atribución.
        </p>
        <form className="search-form" onSubmit={handleSubmit}>
          <label className="sr-only" htmlFor="restaurant-query">
            Describe lo que buscas
          </label>
          <input
            id="restaurant-query"
            name="query"
            type="search"
            maxLength={2000}
            placeholder="Ej. ramen acogedor para cenar en Madrid"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            disabled={loading}
          />
          <button type="submit" disabled={loading}>
            {loading ? "Buscando…" : "Buscar"}
            {!loading && <span aria-hidden="true">↗</span>}
          </button>
        </form>
        <p className="search-note">
          Búsqueda gastronómica con evidencia de OpenStreetMap y fotografías
          atribuidas.
        </p>
      </header>

      <section className="results-section" aria-live="polite">
        {error && (
          <div className="notice notice-error" role="alert">
            <strong>No pudimos completar la búsqueda.</strong>
            <p>{error}</p>
          </div>
        )}

        {loading && (
          <div className="loading-state" role="status">
            <span className="loading-indicator" aria-hidden="true" />
            <p>Buscando opciones y revisando la evidencia disponible…</p>
          </div>
        )}

        {result && (
          <>
            <section className="answer-card" aria-labelledby="answer-title">
              <p className="eyebrow">TU BÚSQUEDA</p>
              <h2 id="answer-title">Una selección para ti</h2>
              <p className="answer-text">{result.answer}</p>
            </section>

            {result.photos.length > 0 && (
              <section
                className="results-group"
                aria-labelledby="photos-title"
              >
                <div className="section-heading">
                  <div>
                    <p className="eyebrow">CON FOTOGRAFÍA</p>
                    <h2 id="photos-title">Lugares que entran por los ojos</h2>
                  </div>
                  <span className="result-count">
                    {result.photos.length}{" "}
                    {result.photos.length === 1 ? "foto" : "fotos"}
                  </span>
                </div>
                <div className="card-grid">
                  {result.photos.map((photo, index) => (
                    <article className="photo-card" key={`${photo.image_url}-${index}`}>
                      <div className="photo-image">
                        <Image
                          src={photo.image_url}
                          alt={
                            photo.restaurant_name
                              ? `Foto de ${photo.restaurant_name}`
                              : "Fotografía de restaurante"
                          }
                          fill
                          sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw"
                        />
                      </div>
                      <div className="card-content">
                        <h3>{photo.restaurant_name ?? "Restaurante"}</h3>
                        <p className="card-meta">
                          {[photo.restaurant_cuisine, photo.restaurant_location]
                            .filter(Boolean)
                            .join(" · ") || "Madrid"}
                        </p>
                        <div className="attribution">
                          {photo.attribution && <span>{photo.attribution}</span>}
                          {photo.license_url ? (
                            <a
                              href={photo.license_url}
                              target="_blank"
                              rel="noreferrer"
                            >
                              {photo.license_name ?? "Licencia de la foto"}
                            </a>
                          ) : (
                            photo.license_name && <span>{photo.license_name}</span>
                          )}
                          {photo.source_url && (
                            <a
                              href={photo.source_url}
                              target="_blank"
                              rel="noreferrer"
                            >
                              Fuente de la fotografía
                            </a>
                          )}
                          {photo.restaurant_source_url && (
                            <a
                              href={photo.restaurant_source_url}
                              target="_blank"
                              rel="noreferrer"
                            >
                              Ver ficha del lugar
                            </a>
                          )}
                        </div>
                      </div>
                    </article>
                  ))}
                </div>
              </section>
            )}

            {result.restaurants.length > 0 && (
              <section
                className="results-group"
                aria-labelledby="restaurants-title"
              >
                <div className="section-heading">
                  <div>
                    <p className="eyebrow">MÁS OPCIONES</p>
                    <h2 id="restaurants-title">Restaurantes para explorar</h2>
                  </div>
                  <span className="result-count">
                    {result.restaurants.length}{" "}
                    {result.restaurants.length === 1 ? "lugar" : "lugares"}
                  </span>
                </div>
                <div className="restaurant-list">
                  {result.restaurants.map((restaurant) => (
                    <article
                      className="restaurant-card"
                      key={`${restaurant.name}-${restaurant.source_url}`}
                    >
                      <div>
                        <p className="card-meta">{restaurant.city}</p>
                        <h3>{restaurant.name}</h3>
                        <p className="restaurant-description">
                          {[restaurant.cuisine, restaurant.location]
                            .filter(Boolean)
                            .join(" · ") || "Detalles del lugar no disponibles"}
                        </p>
                        {restaurant.features.length > 0 && (
                          <ul className="feature-list">
                            {restaurant.features.map((feature) => (
                              <li key={feature}>{feature}</li>
                            ))}
                          </ul>
                        )}
                      </div>
                      <a
                        className="source-link"
                        href={restaurant.source_url}
                        target="_blank"
                        rel="noreferrer"
                      >
                        Ver lugar <span aria-hidden="true">↗</span>
                      </a>
                    </article>
                  ))}
                </div>
              </section>
            )}

            {result.photos.length === 0 && result.restaurants.length === 0 && (
              <div className="notice">
                No encontramos resultados para esta búsqueda. Prueba con otro
                plato o tipo de cocina.
              </div>
            )}
          </>
        )}
      </section>

      <footer className="site-footer">
        <span>Hecho para descubrir la ciudad, un plato a la vez.</span>
        <span>
          Datos de lugares ©{" "}
          <a
            href="https://www.openstreetmap.org/copyright"
            target="_blank"
            rel="noreferrer"
          >
            OpenStreetMap contributors
          </a>
          .
        </span>
      </footer>
    </main>
  );
}

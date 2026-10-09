import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import type { RestaurantSearchResponse } from "@/lib/restaurant-search";
import { RestaurantCard } from "./restaurant-card";
import { RestaurantPhotoCard } from "./restaurant-photo-card";

export function RestaurantResults({
  result,
}: {
  result: RestaurantSearchResponse;
}) {
  const placesCount = result.restaurants.length;
  const photosCount = result.photos.length;

  return (
    <section className="space-y-5 pt-2" aria-label="Resultados de la búsqueda">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.17em] text-primary/75">
            Resultados
          </p>
          <h2 className="mt-1 font-serif text-2xl font-medium tracking-tight sm:text-3xl">
            Lugares para explorar
          </h2>
        </div>
        <div className="flex gap-2">
          <Badge variant="secondary">
            {placesCount} {placesCount === 1 ? "lugar" : "lugares"}
          </Badge>
          <Badge variant="outline">
            {photosCount} {photosCount === 1 ? "foto" : "fotos"}
          </Badge>
        </div>
      </div>

      {photosCount > 0 && (
        <div className="grid gap-4 sm:grid-cols-2">
          {result.photos.map((photo, index) => (
            <RestaurantPhotoCard
              key={`${photo.image_url}-${index}`}
              photo={photo}
            />
          ))}
        </div>
      )}

      {placesCount > 0 ? (
        <div className="grid gap-3">
          {result.restaurants.map((restaurant) => (
            <RestaurantCard
              key={`${restaurant.name}-${restaurant.source_url}`}
              restaurant={restaurant}
            />
          ))}
        </div>
      ) : photosCount === 0 ? (
        <Card>
          <CardContent className="py-5 text-sm text-muted-foreground">
            No encontramos resultados para esta búsqueda. Prueba con otro plato
            o tipo de cocina.
          </CardContent>
        </Card>
      ) : null}
    </section>
  );
}

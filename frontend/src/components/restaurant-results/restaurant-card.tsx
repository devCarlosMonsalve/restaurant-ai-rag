import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import type { RestaurantCandidate } from "@/lib/restaurant-search";

export function RestaurantCard({
  restaurant,
}: {
  restaurant: RestaurantCandidate;
}) {
  return (
    <Card className="flex flex-col gap-4 p-5 sm:flex-row sm:items-center sm:justify-between sm:p-6">
      <div className="min-w-0">
        <p className="text-xs font-medium text-muted-foreground">
          {restaurant.city}
        </p>
        <h3 className="mt-1 font-serif text-xl font-semibold tracking-tight">
          {restaurant.name}
        </h3>
        <p className="mt-1.5 text-sm text-muted-foreground">
          {[restaurant.cuisine, restaurant.location]
            .filter(Boolean)
            .join(" · ") || "Detalles del lugar no disponibles"}
        </p>
        {restaurant.features.length > 0 && (
          <ul className="mt-3 flex flex-wrap gap-1.5">
            {restaurant.features.map((feature) => (
              <li key={feature}>
                <Badge variant="secondary">{feature}</Badge>
              </li>
            ))}
          </ul>
        )}
      </div>
      <a
        className="w-fit shrink-0 text-sm font-semibold text-primary underline-offset-4 hover:underline"
        href={restaurant.source_url}
        target="_blank"
        rel="noreferrer"
      >
        Ver lugar ↗
      </a>
    </Card>
  );
}

import Image from "next/image";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import type { RestaurantPhoto } from "@/lib/restaurant-search";

export function RestaurantPhotoCard({ photo }: { photo: RestaurantPhoto }) {
  return (
    <Card className="overflow-hidden">
      <div className="relative aspect-[16/10] bg-secondary">
        <Image
          src={photo.image_url}
          alt={
            photo.restaurant_name
              ? `Foto de ${photo.restaurant_name}`
              : "Fotografía de restaurante"
          }
          fill
          sizes="(max-width: 640px) 100vw, 50vw"
          className="object-cover"
        />
      </div>
      <CardHeader className="pb-3">
        <CardTitle className="font-serif text-xl">
          {photo.restaurant_name ?? "Restaurante"}
        </CardTitle>
        <CardDescription>
          {[photo.restaurant_cuisine, photo.restaurant_location]
            .filter(Boolean)
            .join(" · ") || "Madrid"}
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-1.5 text-xs leading-5 text-muted-foreground">
        {photo.attribution && <span>{photo.attribution}</span>}
        {photo.license_url ? (
          <a
            className="w-fit underline underline-offset-4 hover:text-foreground"
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
            className="w-fit underline underline-offset-4 hover:text-foreground"
            href={photo.source_url}
            target="_blank"
            rel="noreferrer"
          >
            Fuente de la fotografía
          </a>
        )}
        {photo.restaurant_source_url && (
          <a
            className="w-fit underline underline-offset-4 hover:text-foreground"
            href={photo.restaurant_source_url}
            target="_blank"
            rel="noreferrer"
          >
            Ver ficha del lugar
          </a>
        )}
      </CardContent>
    </Card>
  );
}

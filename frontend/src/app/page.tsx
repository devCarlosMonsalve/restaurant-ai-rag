import { RestaurantChat } from "@/components/restaurant-chat";
import { UtensilsCrossed } from "lucide-react";
import Link from "next/link";

export default function Home() {
  return (
    <main className="mx-auto flex min-h-svh w-full max-w-6xl flex-col px-4 pb-8 pt-5 sm:px-6 lg:px-8">
      <header className="flex items-center justify-between">
        <Link
          className="inline-flex items-center gap-2.5 text-lg font-semibold tracking-tight text-primary"
          href="/"
          aria-label="Saborea, inicio"
        >
          <span className="grid size-9 place-items-center rounded-xl bg-primary text-primary-foreground">
            <UtensilsCrossed aria-hidden="true" className="size-[18px]" />
          </span>
          saborea
        </Link>
        <span className="hidden text-sm text-muted-foreground sm:inline">
          Descubre sitios que merecen la pena
        </span>
      </header>

      <section className="mx-auto flex w-full max-w-3xl flex-1 flex-col justify-center py-10 sm:py-14">
        <div className="mb-6 text-center">
          <p className="mb-3 text-xs font-semibold uppercase tracking-[0.2em] text-primary/75">
            Buen comer, con buenas fuentes
          </p>
          <h1 className="font-serif text-4xl font-medium leading-tight tracking-tight text-foreground sm:text-6xl">
            ¿Qué te apetece <span className="italic text-primary">hoy?</span>
          </h1>
          <p className="mx-auto mt-4 max-w-xl text-base leading-7 text-muted-foreground sm:text-lg">
            Cuéntanos el plato, el ambiente o el tipo de cocina que buscas.
            Encontraremos restaurantes reales y fotos con su atribución.
          </p>
        </div>

        <RestaurantChat />
      </section>

      <footer className="flex flex-col gap-2 border-t pt-5 text-xs leading-5 text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
        <span>Hecho para descubrir la ciudad, un plato a la vez.</span>
        <span>
          Datos de lugares ©{" "}
          <a
            className="underline underline-offset-4 hover:text-foreground"
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

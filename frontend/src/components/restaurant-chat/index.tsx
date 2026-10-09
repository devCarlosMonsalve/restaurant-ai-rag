"use client";

import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { Sparkles } from "lucide-react";
import { RestaurantResults } from "@/components/restaurant-results";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useRestaurantSearchChat } from "@/hooks/use-restaurant-search-chat";
import { RestaurantThread } from "./thread";

export function RestaurantChat() {
  const { result, runtime } = useRestaurantSearchChat();

  return (
    <div className="space-y-5">
      <Card className="overflow-hidden border-border/80 shadow-[0_18px_60px_-36px_oklch(0.3_0.04_130_/_0.35)]">
        <CardHeader className="gap-2 pb-0">
          <div className="flex items-center gap-2">
            <span className="grid size-8 place-items-center rounded-full bg-secondary text-primary">
              <Sparkles aria-hidden="true" className="size-4" />
            </span>
            <div>
              <CardTitle className="text-base">Tu guía gastronómica</CardTitle>
              <CardDescription className="mt-1">
                Cada mensaje inicia una búsqueda independiente.
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="pt-4">
          <AssistantRuntimeProvider runtime={runtime}>
            <RestaurantThread />
          </AssistantRuntimeProvider>
        </CardContent>
      </Card>

      {result && <RestaurantResults result={result} />}
    </div>
  );
}

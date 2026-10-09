"use client";

import {
  useLocalRuntime,
  type ChatModelAdapter,
  type ThreadMessage,
} from "@assistant-ui/react";
import { useMemo, useState } from "react";
import {
  searchRestaurants,
  type RestaurantSearchResponse,
} from "@/lib/restaurant-search";

function latestUserText(messages: readonly ThreadMessage[]): string {
  const latestMessage = [...messages]
    .reverse()
    .find((message) => message.role === "user");

  return (
    latestMessage?.content
      .flatMap((part) => (part.type === "text" ? [part.text] : []))
      .join("")
      .trim() ?? ""
  );
}

export function useRestaurantSearchChat() {
  const [result, setResult] = useState<RestaurantSearchResponse | null>(null);

  const adapter = useMemo<ChatModelAdapter>(
    () => ({
      async run({ messages, abortSignal }) {
        const query = latestUserText(messages);
        if (!query) {
          throw new Error("Escribe qué te apetece antes de buscar.");
        }
        if (query.length > 2000) {
          throw new Error("La búsqueda no puede superar los 2000 caracteres.");
        }

        setResult(null);
        const response = await searchRestaurants(query, abortSignal);
        setResult(response);

        return { content: [{ type: "text", text: response.answer }] };
      },
    }),
    [],
  );
  const runtime = useLocalRuntime(adapter);

  return { result, runtime };
}

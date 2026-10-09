"use client";

import {
  AuiIf,
  ThreadPrimitive,
} from "@assistant-ui/react";
import { LoaderCircle, MapPin } from "lucide-react";
import { ChatComposer } from "./composer";
import { AssistantMessage, UserMessage } from "./message";

export function RestaurantThread() {
  return (
    <ThreadPrimitive.Root className="flex h-[min(30rem,65vh)] min-h-[20rem] flex-col">
      <ThreadPrimitive.Viewport className="flex min-h-0 flex-1 flex-col gap-5 overflow-y-auto px-1 py-4 sm:px-2">
        <AuiIf condition={(state) => state.thread.isEmpty}>
          <div className="flex flex-1 flex-col items-center justify-center px-5 pb-5 text-center">
            <span className="mb-4 grid size-12 place-items-center rounded-2xl bg-secondary text-primary">
              <MapPin aria-hidden="true" className="size-5" />
            </span>
            <p className="font-medium text-foreground">
              ¿Dónde te gustaría comer?
            </p>
            <p className="mt-2 max-w-sm text-sm leading-6 text-muted-foreground">
              Prueba con un plato, una zona o el ambiente que buscas. Te
              mostraremos opciones con fuentes y atribuciones.
            </p>
          </div>
        </AuiIf>

        <ThreadPrimitive.Messages>
          {({ message }) => {
            if (message.role === "user") {
              return <UserMessage />;
            }
            if (message.role === "assistant") {
              return (
                <AssistantMessage
                  hasError={message.status.type === "incomplete"}
                />
              );
            }
            return null;
          }}
        </ThreadPrimitive.Messages>

        <AuiIf condition={(state) => state.thread.isRunning}>
          <div
            className="flex items-center gap-2 pl-1 text-sm text-muted-foreground"
            role="status"
          >
            <LoaderCircle
              aria-hidden="true"
              className="size-4 animate-spin"
            />
            Buscando opciones y comprobando la evidencia…
          </div>
        </AuiIf>

        <ThreadPrimitive.ViewportFooter className="sticky bottom-0 z-10 mt-auto bg-card pt-2">
          <ChatComposer />
        </ThreadPrimitive.ViewportFooter>
      </ThreadPrimitive.Viewport>
    </ThreadPrimitive.Root>
  );
}

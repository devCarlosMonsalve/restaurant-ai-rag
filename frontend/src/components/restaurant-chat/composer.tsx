"use client";

import { ComposerPrimitive } from "@assistant-ui/react";
import { ArrowUp } from "lucide-react";
import { Button } from "@/components/ui/button";

export function ChatComposer() {
  return (
    <ComposerPrimitive.Root className="rounded-xl border bg-background p-2 shadow-sm transition focus-within:border-ring focus-within:ring-2 focus-within:ring-ring/20">
      <ComposerPrimitive.Input
        aria-label="Describe lo que te apetece"
        maxLength={2000}
        placeholder="Ej. ramen acogedor para cenar en Madrid"
        className="min-h-14 w-full resize-none bg-transparent px-3 py-2 text-sm leading-6 outline-none placeholder:text-muted-foreground/75"
        rows={2}
      />
      <div className="flex items-center justify-between gap-3 px-1 pb-1">
        <span className="text-xs text-muted-foreground">
          Enter para enviar · hasta 2000 caracteres
        </span>
        <ComposerPrimitive.Send asChild>
          <Button
            aria-label="Enviar búsqueda"
            size="icon"
            className="size-9 rounded-lg"
          >
            <ArrowUp aria-hidden="true" className="size-4" />
          </Button>
        </ComposerPrimitive.Send>
      </div>
    </ComposerPrimitive.Root>
  );
}

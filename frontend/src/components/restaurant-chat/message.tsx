"use client";

import {
  ErrorPrimitive,
  MessagePartPrimitive,
  MessagePrimitive,
} from "@assistant-ui/react";
import { Sparkles } from "lucide-react";
import ReactMarkdown from "react-markdown";

export function UserMessage() {
  return (
    <MessagePrimitive.Root className="flex justify-end">
      <div className="max-w-[88%] rounded-2xl rounded-br-md bg-primary px-4 py-3 text-sm leading-6 text-primary-foreground sm:max-w-[80%]">
        <MessagePrimitive.Parts>
          {({ part }) =>
            part.type === "text" ? (
              <MessagePartPrimitive.Text className="whitespace-pre-wrap" />
            ) : null
          }
        </MessagePrimitive.Parts>
      </div>
    </MessagePrimitive.Root>
  );
}

export function AssistantMessage({ hasError }: { hasError: boolean }) {
  return (
    <MessagePrimitive.Root className="flex justify-start">
      <div className="flex max-w-full items-start gap-3 sm:max-w-[92%]">
        <span className="mt-0.5 grid size-8 shrink-0 place-items-center rounded-full bg-secondary text-primary">
          <Sparkles aria-hidden="true" className="size-4" />
        </span>
        <div className="min-w-0 flex-1 pt-1">
          <p className="mb-1 text-xs font-semibold text-foreground">Saborea</p>
          <div className="text-sm leading-7 text-foreground/85 [&_a]:font-medium [&_a]:text-primary [&_a]:underline [&_a]:underline-offset-4 [&_li+li]:mt-1.5 [&_ol]:my-3 [&_ol]:list-decimal [&_ol]:space-y-1 [&_ol]:pl-5 [&_p+p]:mt-3 [&_ul]:my-3 [&_ul]:list-disc [&_ul]:space-y-1 [&_ul]:pl-5">
            <MessagePrimitive.Parts>
              {({ part }) =>
                part.type === "text" ? (
                  <ReactMarkdown
                    components={{
                      a: ({ href, children }) => (
                        <a href={href} target="_blank" rel="noreferrer">
                          {children}
                        </a>
                      ),
                    }}
                  >
                    {part.text}
                  </ReactMarkdown>
                ) : null
              }
            </MessagePrimitive.Parts>
          </div>
          {hasError && (
            <ErrorPrimitive.Root className="mt-2 text-sm text-destructive">
              <ErrorPrimitive.Message />
            </ErrorPrimitive.Root>
          )}
        </div>
      </div>
    </MessagePrimitive.Root>
  );
}

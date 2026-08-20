"use client";

import { Sparkles } from "lucide-react";

export default function TypingIndicator() {
  return (
    <div className="animate-fade-in flex gap-3" role="status" aria-label="Assistant is thinking">
      <span
        className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-accent-muted/50 bg-accent-subtle"
        aria-hidden
      >
        <Sparkles className="h-3.5 w-3.5 text-accent-fg" strokeWidth={1.75} />
      </span>

      <div className="flex h-7 items-center gap-1.5" aria-hidden>
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="animate-typing-bounce block h-1.5 w-1.5 rounded-full bg-accent-fg"
            style={{ animationDelay: `${i * 0.18}s` }}
          />
        ))}
      </div>
    </div>
  );
}

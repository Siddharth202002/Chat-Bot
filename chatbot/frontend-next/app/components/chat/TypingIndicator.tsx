"use client";

import Logo from "../ui/Logo";

export default function TypingIndicator() {
  return (
    <div className="animate-fade-in flex gap-3" role="status" aria-label="Assistant is thinking">
      <Logo size={30} className="mt-0.5 hidden rounded-lg shadow-e2 sm:block" />

      <div
        className="surface-card flex h-11 items-center gap-2.5 rounded-3xl rounded-tl-lg px-4 shadow-e1"
        aria-hidden
      >
        <span className="flex items-center gap-1">
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              className="animate-typing-bounce bg-brand block h-1.5 w-1.5 rounded-full"
              style={{ animationDelay: `${i * 0.18}s` }}
            />
          ))}
        </span>
        <span className="shimmer-text text-small font-medium">Thinking</span>
      </div>
    </div>
  );
}

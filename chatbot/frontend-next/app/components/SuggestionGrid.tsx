"use client";

import { cn } from "@/app/lib/utils";
import { BookOpen, Bug, Code2, CloudSun, Lightbulb, PenLine } from "lucide-react";

const SUGGESTIONS = [
  { icon: Code2, text: "Explain Python decorators" },
  { icon: Lightbulb, text: "Give me a project idea" },
  { icon: PenLine, text: "Write something creative" },
  { icon: BookOpen, text: "Summarise a concept" },
  { icon: Bug, text: "Debug my code" },
  { icon: CloudSun, text: "What's the weather today?" },
];

/**
 * Starter prompts shown under the composer on an empty conversation. Each chip
 * sends its text straight through the normal send path.
 */
export default function SuggestionGrid({
  onSelect,
  disabled,
}: {
  onSelect: (text: string) => void;
  disabled?: boolean;
}) {
  return (
    <div className="shrink-0 px-4 pb-5 pt-1 sm:px-6 lg:px-8">
      <ul className="mx-auto grid w-full max-w-2xl grid-cols-1 gap-2 sm:grid-cols-2">
        {SUGGESTIONS.map(({ icon: Icon, text }, i) => (
          <li key={text}>
            <button
              type="button"
              disabled={disabled}
              onClick={() => onSelect(text)}
              style={{ animationDelay: `${120 + i * 35}ms` }}
              className={cn(
                "animate-rise group flex w-full items-center gap-2.5 rounded-md",
                "border border-line bg-raised px-3 py-2.5 text-left",
                "text-small text-fg-muted",
                "transition-colors duration-150 ease-standard",
                "hover:border-line-strong hover:bg-hover hover:text-fg",
                "active:scale-[0.99] disabled:pointer-events-none disabled:opacity-50"
              )}
            >
              <Icon
                className="h-4 w-4 shrink-0 text-fg-subtle transition-colors group-hover:text-accent-fg"
                strokeWidth={1.75}
                aria-hidden
              />
              <span className="truncate">{text}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

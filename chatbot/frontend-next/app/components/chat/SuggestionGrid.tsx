"use client";

import { cn } from "@/app/lib/utils";
import { Code2, CloudSun, Globe, Lightbulb, PenLine, TrendingUp } from "lucide-react";
import { TONE_CLASSES, type ToolTone } from "../tools/toolMeta";

// Each one runs through the normal send path; several exercise the real
// tools (live weather, web search, stock prices).
// `label` is what the chip shows; `text` is the question it sends.
const SUGGESTIONS: { icon: typeof Globe; label: string; text: string; tone: ToolTone }[] = [
  { icon: CloudSun, label: "Today's weather", text: "What's the weather today?", tone: "sky" },
  { icon: Globe, label: "Latest AI news", text: "What's the latest news in AI this week?", tone: "indigo" },
  { icon: Code2, label: "Python decorators", text: "Explain Python decorators", tone: "violet" },
  { icon: Lightbulb, label: "A project idea", text: "Give me a project idea", tone: "amber" },
  { icon: TrendingUp, label: "Apple stock price", text: "What's Apple's stock price?", tone: "emerald" },
  { icon: PenLine, label: "Write something creative", text: "Write something creative", tone: "pink" },
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
    <div className="shrink-0 px-4 pb-4 pt-1 sm:px-6 lg:px-8">
      <ul className="mx-auto grid w-full max-w-xs grid-cols-1 gap-2 sm:max-w-lg sm:grid-cols-2">
        {SUGGESTIONS.map(({ icon: Icon, label, text, tone }, i) => (
          // Phones get the first four; six stacked chips push the composer off-screen.
          <li key={text} className={cn(i >= 4 && "max-sm:hidden")}>
            <button
              type="button"
              disabled={disabled}
              onClick={() => onSelect(text)}
              title={text}
              style={{ animationDelay: `${120 + i * 35}ms` }}
              className={cn(
                "animate-rise group flex w-full items-center gap-2 rounded-full",
                "surface-card py-1.5 pl-1.5 pr-3.5 text-left shadow-e1",
                "text-small font-medium text-fg-muted",
                "transition-[color,box-shadow,transform] duration-150 ease-standard",
                "hover:-translate-y-px hover:text-fg hover:shadow-e2",
                "active:translate-y-0 disabled:pointer-events-none disabled:opacity-50"
              )}
            >
              <span
                className={cn(
                  "flex h-6.5 w-6.5 shrink-0 items-center justify-center rounded-full",
                  TONE_CLASSES[tone]
                )}
                aria-hidden
              >
                <Icon className="h-3.5 w-3.5" strokeWidth={2} />
              </span>
              <span className="truncate">{label}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

"use client";

import type { ToolActivity } from "@/app/lib/tools";
import { cn } from "@/app/lib/utils";
import { AlertCircle, Check } from "lucide-react";
import { TONE_CLASSES, toolErrorMessage, toolMeta } from "./toolMeta";
import WeatherCard, { hasWeatherReading } from "./WeatherCard";

/** One tool call: a compact status card while running, a result when done. */
function ToolCard({ tool }: { tool: ToolActivity }) {
  // A finished weather call has real numbers worth showing properly.
  if (tool.name === "get_weather" && tool.status === "done" && hasWeatherReading(tool.data)) {
    return <WeatherCard data={tool.data} />;
  }

  const meta = toolMeta(tool.name);
  const Icon = meta.icon;
  const subject = meta.subject(tool);
  const isRunning = tool.status === "running";
  const isError = tool.status === "error";
  const errorText = isError ? toolErrorMessage(tool) : null;

  return (
    <div
      className={cn(
        "animate-pop-in surface-card relative flex w-full max-w-md items-center gap-3 overflow-hidden",
        "rounded-2xl py-2.5 pl-2.5 pr-3.5 shadow-e1"
      )}
      role="status"
      aria-live="polite"
    >
      <span
        className={cn(
          "flex h-9 w-9 shrink-0 items-center justify-center rounded-xl",
          TONE_CLASSES[meta.tone]
        )}
        aria-hidden
      >
        <Icon className="h-4.5 w-4.5" strokeWidth={1.9} />
      </span>

      <div className="min-w-0 flex-1">
        <p className="flex items-center gap-1.5 text-small font-semibold text-fg">
          {meta.label}
        </p>
        <p
          className={cn(
            "truncate text-micro",
            isRunning ? "shimmer-text" : isError ? "text-danger" : "text-fg-subtle"
          )}
          title={subject ?? undefined}
        >
          {isRunning
            ? subject
              ? `${meta.running} · ${subject}`
              : `${meta.running}…`
            : isError
              ? (errorText ?? "Couldn't complete this step")
              : (subject ?? "Completed")}
        </p>
      </div>

      <span className="shrink-0" aria-hidden>
        {isRunning ? (
          <span className="flex items-center gap-1">
            {[0, 1, 2].map((i) => (
              <span
                key={i}
                className="animate-typing-bounce block h-1.5 w-1.5 rounded-full bg-accent"
                style={{ animationDelay: `${i * 0.16}s` }}
              />
            ))}
          </span>
        ) : isError ? (
          <span className="flex h-6 w-6 items-center justify-center rounded-full bg-danger-subtle text-danger">
            <AlertCircle className="h-3.5 w-3.5" strokeWidth={2} />
          </span>
        ) : (
          <span className="flex h-6 w-6 items-center justify-center rounded-full bg-success-subtle text-success">
            <Check className="h-3.5 w-3.5" strokeWidth={2.5} />
          </span>
        )}
      </span>

      {isRunning && (
        <span className="progress-indeterminate absolute inset-x-0 bottom-0 h-0.5 bg-transparent" aria-hidden />
      )}
    </div>
  );
}

/** Every tool call of one assistant turn, in the order they ran. */
export default function ToolActivityList({ tools }: { tools: ToolActivity[] }) {
  if (tools.length === 0) return null;
  return (
    <div className="flex flex-col gap-2">
      {tools.map((tool) => (
        <ToolCard key={tool.id} tool={tool} />
      ))}
    </div>
  );
}

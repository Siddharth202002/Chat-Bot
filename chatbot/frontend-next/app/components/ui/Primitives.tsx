"use client";

import { cn } from "@/app/lib/utils";
import type { ComponentType, ReactNode } from "react";

/* ── Skeleton ─────────────────────────────────────────────────── */

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("skeleton", className)} aria-hidden />;
}

/* ── Badge ────────────────────────────────────────────────────── */

type BadgeTone = "neutral" | "accent" | "success" | "danger";

const TONES: Record<BadgeTone, string> = {
  neutral: "border-line bg-hover text-fg-muted",
  accent: "border-accent-muted/60 bg-accent-subtle text-accent-fg",
  success: "border-success/25 bg-success-subtle text-success",
  danger: "border-danger/25 bg-danger-subtle text-danger",
};

export function Badge({
  tone = "neutral",
  dot = false,
  pulse = false,
  children,
  className,
}: {
  tone?: BadgeTone;
  /** Show a leading status dot. */
  dot?: boolean;
  /** Softly pulse the dot — reserved for genuinely live states. */
  pulse?: boolean;
  children: ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5",
        "text-micro font-medium whitespace-nowrap",
        TONES[tone],
        className
      )}
    >
      {dot && (
        <span
          className={cn("h-1.5 w-1.5 rounded-full bg-current", pulse && "animate-pulse")}
          aria-hidden
        />
      )}
      {children}
    </span>
  );
}

/* ── Empty state ──────────────────────────────────────────────── */

export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
  className,
}: {
  icon: ComponentType<{ className?: string; strokeWidth?: number }>;
  title: string;
  description?: string;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-col items-center px-4 py-10 text-center", className)}>
      <Icon className="mb-3 h-6 w-6 text-fg-faint" strokeWidth={1.5} />
      <p className="text-small font-medium text-fg-muted">{title}</p>
      {description && <p className="mt-1 text-micro text-fg-subtle">{description}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

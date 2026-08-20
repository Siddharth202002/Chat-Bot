"use client";

import { cn } from "@/app/lib/utils";
import { forwardRef, type ButtonHTMLAttributes } from "react";

type Variant = "primary" | "secondary" | "ghost" | "danger" | "accentGhost";
type Size = "sm" | "md";

const VARIANTS: Record<Variant, string> = {
  primary:
    "bg-accent text-white shadow-e1 hover:bg-accent-hi active:bg-accent disabled:bg-accent/35 disabled:text-white/50",
  secondary:
    "bg-raised text-fg border border-line hover:bg-hover hover:border-line-strong active:bg-active disabled:text-fg-faint disabled:border-line-subtle disabled:bg-transparent",
  ghost:
    "bg-transparent text-fg-muted hover:bg-hover hover:text-fg active:bg-active disabled:text-fg-faint disabled:hover:bg-transparent",
  accentGhost:
    "bg-accent-subtle text-accent-fg border border-accent-muted/60 hover:bg-accent-muted/30 hover:border-accent-muted active:bg-accent-muted/40 disabled:text-fg-faint",
  danger:
    "bg-danger/90 text-[#1a0d0d] font-semibold hover:bg-danger active:bg-danger/90 disabled:bg-danger/30",
};

const SIZES: Record<Size, string> = {
  sm: "h-8 px-3 text-small gap-1.5",
  md: "h-10 px-4 text-small gap-2",
};

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  /** Stretch to the full width of the parent. */
  block?: boolean;
}

/**
 * The single source of truth for button styling. Every clickable label in the
 * app routes through here so hover/active/disabled/focus states stay identical.
 */
const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "secondary", size = "md", block, className, type = "button", ...props },
  ref
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cn(
        "inline-flex shrink-0 items-center justify-center rounded-md font-medium",
        "transition-colors duration-150 ease-standard",
        "active:scale-[0.985] disabled:pointer-events-none",
        SIZES[size],
        VARIANTS[variant],
        block && "w-full",
        className
      )}
      {...props}
    />
  );
});

export default Button;

/* ────────────────────────────────────────────────────────────── */

export interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  /** Required — icon-only controls have no text for assistive tech to read. */
  label: string;
  size?: Size;
  variant?: Variant;
  /** Render a native tooltip in addition to the accessible name. */
  tooltip?: boolean;
}

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  function IconButton(
    { label, size = "md", variant = "ghost", tooltip = true, className, type = "button", ...props },
    ref
  ) {
    return (
      <button
        ref={ref}
        type={type}
        aria-label={label}
        title={tooltip ? label : undefined}
        className={cn(
          "inline-flex shrink-0 items-center justify-center rounded-md",
          "transition-colors duration-150 ease-standard",
          "active:scale-95 disabled:pointer-events-none",
          size === "sm" ? "h-8 w-8" : "h-9 w-9",
          VARIANTS[variant],
          className
        )}
        {...props}
      />
    );
  }
);

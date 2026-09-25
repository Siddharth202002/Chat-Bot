"use client";

import { cn } from "@/app/lib/utils";
import { forwardRef, type ButtonHTMLAttributes } from "react";

type Variant = "primary" | "secondary" | "ghost" | "danger" | "accentGhost";
type Size = "sm" | "md";

const VARIANTS: Record<Variant, string> = {
  primary:
    "bg-brand text-white shadow-glow hover:brightness-[1.06] hover:shadow-e3 active:brightness-95 disabled:opacity-45 disabled:shadow-none",
  secondary:
    "bg-raised/80 text-fg border border-line shadow-e1 hover:bg-raised hover:border-line-strong active:bg-active disabled:text-fg-faint disabled:border-line-subtle disabled:bg-transparent disabled:shadow-none",
  ghost:
    "bg-transparent text-fg-muted hover:bg-hover hover:text-fg active:bg-active disabled:text-fg-faint disabled:hover:bg-transparent",
  accentGhost:
    "bg-accent-subtle text-accent-fg border border-accent-muted/60 hover:bg-accent-muted/40 hover:border-accent-muted active:bg-accent-muted/50 disabled:text-fg-faint",
  danger:
    "bg-danger text-white font-semibold shadow-e1 hover:brightness-110 active:brightness-95 disabled:opacity-40",
};

const SIZES: Record<Size, string> = {
  sm: "h-8 px-3.5 text-small gap-1.5",
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
        "inline-flex shrink-0 items-center justify-center rounded-full font-medium",
        "transition-[background-color,border-color,color,box-shadow,filter,transform] duration-150 ease-standard",
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
          "inline-flex shrink-0 items-center justify-center rounded-full",
          "transition-[background-color,color,box-shadow,transform] duration-150 ease-standard",
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

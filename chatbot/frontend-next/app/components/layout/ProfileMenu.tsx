"use client";

import type { Theme } from "@/app/lib/theme";
import { cn } from "@/app/lib/utils";
import { ChevronsUpDown, LogOut, Moon, Sun } from "lucide-react";
import { useEffect, useRef, useState, type CSSProperties } from "react";
import { createPortal } from "react-dom";

interface ProfileMenuProps {
  userEmail: string;
  onLogout: () => void;
  theme: Theme;
  onThemeChange: (theme: Theme) => void;
  /** "full" is the sidebar footer card; "compact" is the avatar in the rail. */
  variant: "full" | "compact";
}

function Avatar({ email, size = 36 }: { email: string; size?: number }) {
  const initial = email.trim().charAt(0).toUpperCase() || "?";
  return (
    <span
      className="bg-brand flex shrink-0 items-center justify-center rounded-full font-semibold text-white shadow-glow"
      style={{ width: size, height: size, fontSize: size * 0.4 }}
      aria-hidden
    >
      {initial}
    </span>
  );
}

export default function ProfileMenu({
  userEmail,
  onLogout,
  theme,
  onThemeChange,
  variant,
}: ProfileMenuProps) {
  const [open, setOpen] = useState(false);
  /** Where the rail's menu floats; it is portalled, so it needs coordinates. */
  const [floating, setFloating] = useState<CSSProperties | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  function toggle() {
    if (!open && variant === "compact" && ref.current) {
      const rect = ref.current.getBoundingClientRect();
      setFloating({
        position: "fixed",
        left: rect.right + 12,
        bottom: Math.max(12, window.innerHeight - rect.bottom),
      });
    }
    setOpen((v) => !v);
  }

  useEffect(() => {
    if (!open) return;
    function onPointerDown(event: MouseEvent) {
      const target = event.target as Node;
      if (ref.current?.contains(target) || menuRef.current?.contains(target)) return;
      setOpen(false);
    }
    function onKeyDown(event: globalThis.KeyboardEvent) {
      if (event.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  const menu = (
    <div
      role="menu"
      aria-label="Account"
      ref={menuRef}
      style={variant === "compact" ? (floating ?? undefined) : undefined}
      className={cn(
        "animate-pop-in glass-strong z-[80] w-64 rounded-2xl p-1.5 shadow-e3",
        variant === "full" && "absolute bottom-full left-0 mb-2"
      )}
    >
      <div className="flex items-center gap-2.5 px-2.5 pb-2.5 pt-1.5">
        <Avatar email={userEmail} size={30} />
        <p className="min-w-0 truncate text-small font-medium text-fg" title={userEmail}>
          {userEmail}
        </p>
      </div>

      <div className="mx-1 mb-1.5 border-t border-line-subtle" />

      <p className="px-2.5 pb-1.5 text-micro font-semibold uppercase tracking-wider text-fg-subtle">
        Theme
      </p>
      <div
        role="radiogroup"
        aria-label="Theme"
        className="mx-1 mb-1.5 grid grid-cols-2 gap-1 rounded-xl bg-active p-1"
      >
        {(
          [
            { value: "light", label: "Light", icon: Sun },
            { value: "dark", label: "Dark", icon: Moon },
          ] as const
        ).map(({ value, label, icon: Icon }) => (
          <button
            key={value}
            type="button"
            role="radio"
            aria-checked={theme === value}
            onClick={() => onThemeChange(value)}
            className={cn(
              "flex h-8 items-center justify-center gap-1.5 rounded-lg text-small font-medium",
              "transition-[background-color,color,box-shadow] duration-150",
              theme === value
                ? "bg-raised text-fg shadow-e1"
                : "text-fg-subtle hover:text-fg"
            )}
          >
            <Icon className="h-3.5 w-3.5" strokeWidth={2} aria-hidden />
            {label}
          </button>
        ))}
      </div>

      <div className="mx-1 mb-1 border-t border-line-subtle" />

      <button
        type="button"
        role="menuitem"
        onClick={() => {
          setOpen(false);
          onLogout();
        }}
        className={cn(
          "flex w-full items-center gap-2.5 rounded-xl px-2.5 py-2 text-left text-small text-fg-muted",
          "transition-colors duration-150 hover:bg-danger-subtle hover:text-danger"
        )}
      >
        <LogOut className="h-4 w-4" strokeWidth={1.9} aria-hidden />
        Sign out
      </button>
    </div>
  );

  return (
    <div className="relative" ref={ref}>
      {variant === "full" ? (
        <button
          type="button"
          onClick={toggle}
          aria-haspopup="menu"
          aria-expanded={open}
          className={cn(
            "surface-card flex w-full items-center gap-2.5 rounded-2xl p-2 pr-2.5 text-left shadow-e1",
            "transition-shadow duration-150 ease-standard hover:shadow-e2",
            open && "shadow-e2"
          )}
        >
          <Avatar email={userEmail} />
          <span className="min-w-0 flex-1">
            <span className="block truncate text-small font-semibold text-fg" title={userEmail}>
              {userEmail}
            </span>
            <span className="block text-micro text-fg-subtle">Account &amp; theme</span>
          </span>
          <ChevronsUpDown className="h-4 w-4 shrink-0 text-fg-subtle" strokeWidth={1.9} aria-hidden />
        </button>
      ) : (
        <button
          type="button"
          onClick={toggle}
          aria-haspopup="menu"
          aria-expanded={open}
          aria-label="Account menu"
          title={userEmail}
          className="rounded-full transition-transform duration-150 hover:scale-105 active:scale-95"
        >
          <Avatar email={userEmail} size={38} />
        </button>
      )}

      {open &&
        (variant === "compact"
          ? // The rail sits inside the glass frame, whose backdrop-filter and
            // overflow clip anything fixed-positioned beneath it.
            createPortal(menu, document.body)
          : menu)}
    </div>
  );
}

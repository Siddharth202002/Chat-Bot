"use client";

import { cn } from "@/app/lib/utils";
import { Check, ChevronDown, Sparkles } from "lucide-react";
import { useEffect, useRef, useState } from "react";

export interface ModelOption {
  id: string;
  /** The model itself, e.g. "gpt-oss-120b" — what tells two Groq entries apart. */
  label: string;
  /** Provider family, e.g. "Groq". */
  family: string;
  model: string;
  default: boolean;
}

interface ModelPickerProps {
  /** Empty until /api/models answers, or when no provider has a key. */
  models: ModelOption[];
  selectedModel: string | null;
  onSelectModel: (id: string) => void;
}

/**
 * Model picker. Hidden when the deployment has one model or none — a menu
 * with a single entry is just noise. Stays usable mid-generation, since the
 * natural next move after a slow reply is to switch model and retry.
 */
export default function ModelPicker({ models, selectedModel, onSelectModel }: ModelPickerProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Fall back to the chain head if nothing is picked yet, so the pill always
  // names the model that will actually answer.
  const activeModel =
    models.find((m) => m.id === selectedModel) ?? models.find((m) => m.default) ?? null;

  useEffect(() => {
    if (!open) return;

    function onPointerDown(event: MouseEvent) {
      if (!ref.current?.contains(event.target as Node)) setOpen(false);
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

  if (models.length <= 1 || !activeModel) return null;

  return (
    <div className="relative min-w-0" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        title={`${activeModel.family} · ${activeModel.model}`}
        className={cn(
          "glass-panel flex h-9 max-w-[14rem] items-center gap-1.5 rounded-full pl-2 pr-3 shadow-e1",
          "text-small font-medium text-fg",
          "transition-shadow duration-150 ease-standard hover:shadow-e2",
          open && "shadow-e2"
        )}
      >
        <span className="bg-brand flex h-5.5 w-5.5 shrink-0 items-center justify-center rounded-full text-white">
          <Sparkles className="h-3 w-3" strokeWidth={2.25} aria-hidden />
        </span>
        <span className="truncate">{activeModel.label}</span>
        <ChevronDown
          className={cn(
            "h-3.5 w-3.5 shrink-0 text-fg-subtle transition-transform duration-150",
            open && "rotate-180"
          )}
          strokeWidth={2}
          aria-hidden
        />
      </button>

      {open && (
        <div
          role="menu"
          aria-label="Choose a model"
          className={cn(
            "animate-pop-in glass-strong absolute left-0 top-full z-40 mt-2 w-72",
            "overflow-hidden rounded-2xl p-1.5 shadow-e3"
          )}
        >
          <p className="px-2.5 pb-1.5 pt-1 text-micro font-semibold uppercase tracking-wider text-fg-subtle">
            Model
          </p>
          {models.map((model) => {
            const isActive = model.id === activeModel.id;
            return (
              <button
                key={model.id}
                type="button"
                role="menuitemradio"
                aria-checked={isActive}
                onClick={() => {
                  setOpen(false);
                  onSelectModel(model.id);
                }}
                className={cn(
                  "flex w-full items-center gap-2.5 rounded-xl px-2.5 py-2 text-left",
                  "transition-colors duration-150 ease-standard",
                  isActive ? "bg-accent-subtle text-fg" : "text-fg-muted hover:bg-hover hover:text-fg"
                )}
              >
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-small font-medium">{model.label}</span>
                  <span className="block truncate text-micro text-fg-subtle">
                    {model.family}
                    {model.default && " · default"}
                  </span>
                </span>
                <Check
                  className={cn("h-4 w-4 shrink-0", isActive ? "text-accent-fg" : "opacity-0")}
                  strokeWidth={2.25}
                  aria-hidden
                />
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

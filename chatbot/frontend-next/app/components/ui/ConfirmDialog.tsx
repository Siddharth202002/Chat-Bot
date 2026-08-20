"use client";

import { useFocusTrap } from "@/app/lib/useFocusTrap";
import { AlertTriangle } from "lucide-react";
import { useCallback, useRef } from "react";
import Button from "./Button";

interface ConfirmDialogProps {
  open: boolean;
  title: string;
  description: string;
  confirmLabel?: string;
  cancelLabel?: string;
  /** Styles the confirm action as destructive and shows a warning glyph. */
  destructive?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export default function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  destructive = false,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const handleEscape = useCallback(() => onCancel(), [onCancel]);

  useFocusTrap(panelRef, open, handleEscape);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-90 flex items-center justify-center p-4">
      <div
        className="animate-fade-in absolute inset-0 bg-black/65 backdrop-blur-[2px]"
        onClick={onCancel}
        aria-hidden
      />
      <div
        ref={panelRef}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
        aria-describedby="confirm-description"
        className="animate-dialog-in relative w-full max-w-sm rounded-xl border border-line-strong bg-overlay p-6 shadow-e3"
      >
        <div className="flex items-start gap-3">
          {destructive && (
            <span
              className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-danger-subtle text-danger"
              aria-hidden
            >
              <AlertTriangle className="h-4.5 w-4.5" strokeWidth={1.75} />
            </span>
          )}
          <div className="min-w-0">
            <h2 id="confirm-title" className="text-h2 text-fg">
              {title}
            </h2>
            <p id="confirm-description" className="mt-1.5 text-small leading-relaxed text-fg-muted">
              {description}
            </p>
          </div>
        </div>

        <div className="mt-6 flex justify-end gap-3">
          <Button variant="secondary" size="sm" onClick={onCancel} data-autofocus>
            {cancelLabel}
          </Button>
          <Button
            variant={destructive ? "danger" : "primary"}
            size="sm"
            onClick={onConfirm}
          >
            {confirmLabel}
          </Button>
        </div>
      </div>
    </div>
  );
}

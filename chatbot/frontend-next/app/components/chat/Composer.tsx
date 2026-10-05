"use client";

import { cn } from "@/app/lib/utils";
import { ArrowUp, FileText, MapPin, MapPinOff, Plus, Square } from "lucide-react";
import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type KeyboardEvent,
  type ReactNode,
  type RefObject,
} from "react";

interface ComposerProps {
  input: string;
  isLoading: boolean;
  isUploadingPdf: boolean;
  /** "welcome" centres the composer under the hero; "docked" pins it to the bottom. */
  variant: "welcome" | "docked";
  onInputChange: (value: string) => void;
  onSend: (text: string) => void;
  onStopGenerating: () => void;
  onAttach: () => void;
  onDropPdf: (file: File) => void;
  onRequestLocation: () => void;
  onClearLocation: () => void;
  /** The resolved place, shown next to the location item once it is known. */
  locationLabel?: string | null;
  textareaRef: RefObject<HTMLTextAreaElement | null>;
}

const MAX_TEXTAREA_HEIGHT = 168;

function MenuItem({
  icon: Icon,
  onClick,
  children,
}: {
  icon: typeof MapPin;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      role="menuitem"
      onClick={onClick}
      className={cn(
        "flex w-full items-center gap-3 rounded-xl px-2.5 py-2 text-left",
        "text-small text-fg-muted",
        "transition-colors duration-150 ease-standard",
        "hover:bg-hover hover:text-fg"
      )}
    >
      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-accent-subtle text-accent-fg">
        <Icon className="h-4 w-4" strokeWidth={1.9} aria-hidden />
      </span>
      <span className="min-w-0 flex-1">{children}</span>
    </button>
  );
}

/**
 * The one and only message composer. Mounted for the whole session — the
 * welcome screen and the active conversation share this exact instance, so
 * focus, draft text and keyboard behaviour never reset between the two.
 */
export default function Composer({
  input,
  isLoading,
  isUploadingPdf,
  variant,
  onInputChange,
  onSend,
  onStopGenerating,
  onAttach,
  onDropPdf,
  onRequestLocation,
  onClearLocation,
  locationLabel,
  textareaRef,
}: ComposerProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const dragDepth = useRef(0);
  const menuRef = useRef<HTMLDivElement>(null);

  const canSend = input.trim().length > 0 && !isLoading && !isUploadingPdf;

  // Auto-grow the textarea up to a cap, then scroll internally.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_TEXTAREA_HEIGHT)}px`;
  }, [input, textareaRef]);

  useEffect(() => {
    if (!menuOpen) return;

    function onPointerDown(event: MouseEvent) {
      if (!menuRef.current?.contains(event.target as Node)) setMenuOpen(false);
    }
    function onKeyDown(event: globalThis.KeyboardEvent) {
      if (event.key === "Escape") setMenuOpen(false);
    }

    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [menuOpen]);

  function runMenuAction(action: () => void) {
    setMenuOpen(false);
    action();
  }

  function handleSend() {
    if (!canSend) return;
    onSend(input);
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function handleDragEnter(e: DragEvent<HTMLDivElement>) {
    if (!e.dataTransfer.types.includes("Files")) return;
    e.preventDefault();
    dragDepth.current += 1;
    setIsDragging(true);
  }

  function handleDragLeave(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    dragDepth.current -= 1;
    if (dragDepth.current <= 0) {
      dragDepth.current = 0;
      setIsDragging(false);
    }
  }

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    dragDepth.current = 0;
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) onDropPdf(file);
  }

  const placeholder = isLoading
    ? "Type your next message…"
    : variant === "welcome"
      ? "Ask me anything…"
      : "Ask anything…";

  return (
    <div className="relative z-10 w-full shrink-0">
      <div
        className={cn(
          "mx-auto w-full px-3 sm:px-6 lg:px-10",
          variant === "welcome" ? "max-w-2xl pb-3 pt-1" : "max-w-3xl pb-3 pt-2 sm:pb-4"
        )}
      >
        <div
          className="relative"
          onDragEnter={handleDragEnter}
          onDragOver={(e) => e.preventDefault()}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {/* Drag-and-drop overlay */}
          {isDragging && (
            <div className="animate-fade-in pointer-events-none absolute inset-0 z-20 flex items-center justify-center rounded-[1.75rem] border-2 border-dashed border-accent bg-raised/90">
              <span className="flex items-center gap-2 text-small font-semibold text-accent-fg">
                <FileText className="h-4 w-4" strokeWidth={2} aria-hidden />
                Drop a PDF to index it
              </span>
            </div>
          )}

          {/* Input surface */}
          <div
            className={cn(
              "glass-strong flex items-end gap-2 rounded-[1.75rem] p-2 shadow-e2",
              "transition-[box-shadow,border-color] duration-200 ease-standard",
              "focus-within:border-accent-muted focus-within:shadow-focus",
              variant === "welcome" && "sm:p-2.5"
            )}
          >
            <div className="relative shrink-0" ref={menuRef}>
              <button
                type="button"
                aria-label={isUploadingPdf ? "Uploading PDF…" : "Add location or a PDF"}
                title={isUploadingPdf ? "Uploading PDF…" : "Add location or a PDF"}
                disabled={isUploadingPdf}
                onClick={() => setMenuOpen((open) => !open)}
                aria-haspopup="menu"
                aria-expanded={menuOpen}
                className={cn(
                  "flex h-10 w-10 items-center justify-center rounded-full",
                  "bg-raised text-fg-muted shadow-e1 ring-1 ring-line-subtle",
                  "transition-[color,box-shadow,transform] duration-150 ease-standard",
                  "hover:text-accent-fg hover:shadow-e2 active:scale-95",
                  "disabled:cursor-not-allowed disabled:opacity-50",
                  menuOpen && "text-accent-fg shadow-e2"
                )}
              >
                {isUploadingPdf ? (
                  <span
                    className="h-4 w-4 animate-spin rounded-full border-2 border-accent-muted border-t-accent"
                    aria-hidden
                  />
                ) : (
                  <Plus
                    className={cn("h-5 w-5 transition-transform duration-200", menuOpen && "rotate-45")}
                    strokeWidth={2}
                  />
                )}
              </button>

              {menuOpen && (
                <div
                  role="menu"
                  aria-label="Attach"
                  className={cn(
                    "animate-pop-in glass-strong absolute bottom-full left-0 z-30 mb-3 w-64",
                    "overflow-hidden rounded-2xl p-1.5 shadow-e3"
                  )}
                >
                  <MenuItem icon={MapPin} onClick={() => runMenuAction(onRequestLocation)}>
                    Use my location
                    {locationLabel && (
                      <span className="block truncate text-micro text-fg-subtle">
                        {locationLabel}
                      </span>
                    )}
                  </MenuItem>

                  {/* The only way to revoke sharing. Offered only when there is
                      something to revoke. */}
                  {locationLabel && (
                    <MenuItem icon={MapPinOff} onClick={() => runMenuAction(onClearLocation)}>
                      Stop sharing location
                    </MenuItem>
                  )}

                  <MenuItem icon={FileText} onClick={() => runMenuAction(onAttach)}>
                    Upload a PDF
                    <span className="block truncate text-micro text-fg-subtle">
                      Ask questions about any document
                    </span>
                  </MenuItem>
                </div>
              )}
            </div>

            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e: ChangeEvent<HTMLTextAreaElement>) => onInputChange(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              rows={1}
              aria-label="Message"
              className={cn(
                "flex-1 resize-none self-center bg-transparent px-1 py-2",
                "text-body-lg text-fg placeholder:text-fg-subtle",
                "outline-none focus-visible:outline-none"
              )}
              style={{ maxHeight: MAX_TEXTAREA_HEIGHT }}
            />

            {/* One control, two jobs. While a reply is in flight this is the
                progress indicator *and* the way to cancel it, so the arrow
                turns into a stop square inside a spinning ring — there is no
                separate button to hunt for, and no moment where the composer
                looks idle while the model is working. Live for the whole
                in-flight window, so a slow first token is still cancellable. */}
            <button
              type="button"
              onClick={isLoading ? onStopGenerating : handleSend}
              disabled={!isLoading && !canSend}
              aria-label={isLoading ? "Stop generating" : "Send message"}
              title={isLoading ? "Stop generating" : "Send message (Enter)"}
              className={cn(
                "relative flex h-10 w-10 shrink-0 items-center justify-center rounded-full",
                "transition-[filter,box-shadow,transform,background-color] duration-150 ease-standard",
                isLoading || canSend
                  ? "bg-brand text-white shadow-glow hover:brightness-110 active:scale-95"
                  : "cursor-not-allowed bg-active text-fg-faint"
              )}
            >
              {isLoading ? (
                <>
                  <span
                    className="absolute -inset-0.5 animate-spin rounded-full border-2 border-accent-muted border-t-accent"
                    aria-hidden
                  />
                  <Square className="h-3 w-3 fill-current" strokeWidth={0} aria-hidden />
                </>
              ) : (
                <ArrowUp className="h-4.5 w-4.5" strokeWidth={2.25} />
              )}
            </button>
          </div>
        </div>

        <p className="mt-2 text-center text-micro text-fg-subtle">
          {isUploadingPdf ? (
            <>Indexing PDF. You can send your question when it finishes.</>
          ) : isLoading ? (
            // There is no separate "Stop generating" pill, so this is where
            // the user learns the send button now cancels.
            <>Generating. Click the stop button to cancel.</>
          ) : (
            <>Zeno AI can make mistakes. Verify important information.</>
          )}
        </p>
      </div>
    </div>
  );
}

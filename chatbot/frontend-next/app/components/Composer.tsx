"use client";

import { cn } from "@/app/lib/utils";
import { ArrowUp, Paperclip, Square } from "lucide-react";
import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type KeyboardEvent,
  type RefObject,
} from "react";
import { IconButton } from "./ui/Button";

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
  textareaRef: RefObject<HTMLTextAreaElement | null>;
}

const MAX_TEXTAREA_HEIGHT = 168;

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
  textareaRef,
}: ComposerProps) {
  const [isDragging, setIsDragging] = useState(false);
  const dragDepth = useRef(0);

  const canSend = input.trim().length > 0 && !isLoading && !isUploadingPdf;

  // Auto-grow the textarea up to a cap, then scroll internally.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_TEXTAREA_HEIGHT)}px`;
  }, [input, textareaRef]);

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
    <div
      className={cn(
        "relative z-10 w-full shrink-0",
        variant === "docked" && "border-t border-line-subtle bg-canvas/80 backdrop-blur-md"
      )}
    >
      <div
        className={cn(
          "mx-auto w-full px-4 sm:px-6 lg:px-8",
          variant === "welcome" ? "max-w-2xl pb-2 pt-1" : "max-w-3xl pb-3 pt-3 sm:pb-4"
        )}
      >
        <div
          className="relative"
          onDragEnter={handleDragEnter}
          onDragOver={(e) => e.preventDefault()}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {/* Stop generating — overlays rather than pushing the composer down.
              Shown for the whole in-flight window, not just once tokens arrive,
              so a slow first response is still cancellable. */}
          {isLoading && (
            <div className="pointer-events-none absolute -top-11 left-0 right-0 flex justify-center">
              <button
                type="button"
                onClick={onStopGenerating}
                className={cn(
                  "pointer-events-auto inline-flex items-center gap-2 rounded-full",
                  "border border-line-strong bg-overlay px-3.5 py-1.5",
                  "text-small font-medium text-fg-muted shadow-e2",
                  "transition-colors duration-150 hover:border-accent-muted hover:text-fg active:scale-[0.98]"
                )}
              >
                <Square className="h-3 w-3 fill-current" strokeWidth={0} aria-hidden />
                Stop generating
              </button>
            </div>
          )}

          {/* Drag-and-drop overlay */}
          {isDragging && (
            <div className="animate-fade-in pointer-events-none absolute inset-0 z-20 flex items-center justify-center rounded-xl border-2 border-dashed border-accent-muted bg-overlay/95">
              <span className="text-small font-medium text-accent-fg">Drop a PDF to index it</span>
            </div>
          )}

          {/* Input surface */}
          <div
            className={cn(
              "flex items-end gap-1.5 rounded-xl border border-line bg-raised p-2 pl-2.5",
              "transition-[border-color,box-shadow] duration-200 ease-standard",
              "focus-within:border-focus focus-within:shadow-focus"
            )}
          >
            <IconButton
              label={isUploadingPdf ? "Uploading PDF…" : "Attach a PDF"}
              size="sm"
              disabled={isUploadingPdf}
              onClick={onAttach}
              className="mb-0.5 rounded-full"
            >
              <Paperclip className="h-4 w-4" strokeWidth={1.75} />
            </IconButton>

            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e: ChangeEvent<HTMLTextAreaElement>) => onInputChange(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              rows={1}
              aria-label="Message"
              className={cn(
                "flex-1 resize-none self-center bg-transparent py-1.5",
                "text-body-lg text-fg placeholder:text-fg-subtle",
                "outline-none focus-visible:outline-none"
              )}
              style={{ maxHeight: MAX_TEXTAREA_HEIGHT }}
            />

            <button
              type="button"
              onClick={handleSend}
              disabled={!canSend}
              aria-label="Send message"
              title="Send message"
              className={cn(
                "mb-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
                "transition-colors duration-150 ease-standard",
                canSend
                  ? "bg-accent text-white hover:bg-accent-hi active:scale-95"
                  : "cursor-not-allowed bg-hover text-fg-faint"
              )}
            >
              <ArrowUp className="h-4 w-4" strokeWidth={2.25} />
            </button>
          </div>
        </div>

        <p className="mt-2 text-center text-micro text-fg-subtle">
          {isUploadingPdf ? (
            <>Indexing PDF. You can send your question when it finishes.</>
          ) : isLoading ? (
            <>Press Enter after the reply finishes to send your next message.</>
          ) : (
            <>Zeno AI can make mistakes. Verify important information.</>
          )}
        </p>
      </div>
    </div>
  );
}

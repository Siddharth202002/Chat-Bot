"use client";

import { cn } from "@/app/lib/utils";
import { closeDanglingMarkup } from "@/app/lib/streamingMarkdown";
import { Check, Copy, Pencil, RotateCcw, Sparkles } from "lucide-react";
import { memo, useEffect, useRef, useState, type KeyboardEvent } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import Button from "./ui/Button";

export interface Message {
  id: string;
  /**
   * The id this message has in the stored conversation, once it is known.
   *
   * `id` is a client-side key that exists from the moment a bubble is
   * rendered; this is the handle the server uses to locate the turn, and it
   * only arrives when the turn is committed. Editing and retrying are
   * unavailable until then.
   */
  serverId?: string | null;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

interface MessageBubbleProps {
  message: Message;
  index: number;
  isStreaming?: boolean;
  /** False while a turn is in flight, or before the message has a server id. */
  canBranch?: boolean;
  isEditing?: boolean;
  onStartEdit?: (message: Message) => void;
  onCancelEdit?: () => void;
  onSubmitEdit?: (message: Message, text: string) => void;
  onRetry?: (message: Message) => void;
}

const LANGUAGE_LABELS: Record<string, string> = {
  js: "JavaScript",
  jsx: "JSX",
  ts: "TypeScript",
  tsx: "TSX",
  javascript: "JavaScript",
  typescript: "TypeScript",
  py: "Python",
  python: "Python",
  sh: "Shell",
  bash: "Bash",
  css: "CSS",
  html: "HTML",
  json: "JSON",
  sql: "SQL",
  yaml: "YAML",
};

function labelFor(lang: string) {
  return LANGUAGE_LABELS[lang] ?? lang.charAt(0).toUpperCase() + lang.slice(1);
}

/* ── Copy control, shared by code blocks and whole responses ────── */

function CopyButton({
  text,
  label = "Copy",
  className,
}: {
  text: string;
  label?: string;
  className?: string;
}) {
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // Async Clipboard API is unavailable on insecure origins and in some
      // embedded webviews — fall back so the button is never a silent no-op.
      const scratch = document.createElement("textarea");
      scratch.value = text;
      scratch.setAttribute("readonly", "");
      scratch.style.cssText = "position:fixed;top:-9999px;opacity:0";
      document.body.appendChild(scratch);
      scratch.select();
      try {
        document.execCommand("copy");
      } catch {
        /* Nothing left to try. */
      }
      scratch.remove();
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <button
      type="button"
      onClick={handleCopy}
      aria-label={copied ? "Copied" : label}
      title={copied ? "Copied" : label}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-sm px-1.5 py-1 text-micro font-medium",
        "text-fg-subtle transition-colors duration-150 hover:bg-hover hover:text-fg",
        className
      )}
    >
      {copied ? (
        <Check className="h-3.5 w-3.5 text-success" strokeWidth={2} />
      ) : (
        <Copy className="h-3.5 w-3.5" strokeWidth={1.75} />
      )}
    </button>
  );
}

/** Edit and Retry, styled to sit beside CopyButton without looking bolted on. */
function ActionButton({
  label,
  icon: Icon,
  onClick,
  disabled,
}: {
  label: string;
  icon: typeof Pencil;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={label}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-sm px-1.5 py-1 text-micro font-medium",
        "text-fg-subtle transition-colors duration-150 hover:bg-hover hover:text-fg",
        "disabled:pointer-events-none disabled:text-fg-faint"
      )}
    >
      <Icon className="h-3.5 w-3.5" strokeWidth={1.75} />
    </button>
  );
}

/* ── Inline editor for a user message ───────────────────────────── */

function MessageEditor({
  initialValue,
  onCancel,
  onSubmit,
}: {
  initialValue: string;
  onCancel: () => void;
  onSubmit: (text: string) => void;
}) {
  const [value, setValue] = useState(initialValue);
  const [submitted, setSubmitted] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // Open with the caret at the end of the existing text, the way a rename
  // field does — selecting all of it invites an accidental overwrite.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.focus();
    el.setSelectionRange(el.value.length, el.value.length);
  }, []);

  // Grow with the content rather than scrolling a three-line box.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 240)}px`;
  }, [value]);

  const trimmed = value.trim();
  const canSubmit = trimmed.length > 0 && !submitted;

  function submit() {
    if (!canSubmit) return;
    // A double Enter, or Enter racing the Save click, would otherwise fork
    // the conversation twice.
    setSubmitted(true);
    onSubmit(trimmed);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Escape") {
      event.preventDefault();
      onCancel();
      return;
    }
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  }

  return (
    <div className="w-full rounded-lg border border-accent-muted/40 bg-accent-subtle px-3 py-2.5">
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={handleKeyDown}
        disabled={submitted}
        rows={1}
        aria-label="Edit your message"
        className={cn(
          "w-full resize-none bg-transparent text-body text-fg outline-none",
          "placeholder:text-fg-faint disabled:text-fg-muted"
        )}
      />
      <div className="mt-2 flex items-center justify-end gap-2">
        <Button variant="ghost" size="sm" onClick={onCancel} disabled={submitted}>
          Cancel
        </Button>
        <Button variant="primary" size="sm" onClick={submit} disabled={!canSubmit}>
          {submitted ? "Sending…" : "Send"}
        </Button>
      </div>
    </div>
  );
}

/* ── Markdown renderer ──────────────────────────────────────────── */

// GitHub-Flavoured Markdown: tables, strikethrough, task lists, autolinks.
// react-markdown is CommonMark-only without this, so model-authored tables
// render as a run-on paragraph of pipes.
const REMARK_PLUGINS = [remarkGfm];

const markdownComponents = {
  // Wide tables scroll inside their own box instead of stretching the message.
  table({ children }: { children?: React.ReactNode }) {
    return (
      <div className="md-table-wrap">
        <table>{children}</table>
      </div>
    );
  },
  code({ className, children, ...props }: { className?: string; children?: React.ReactNode }) {
    const match = /language-(\w+)/.exec(className || "");
    const codeString = String(children).replace(/\n$/, "");

    if (!match) {
      return <code {...props}>{children}</code>;
    }

    return (
      <div className="my-4 overflow-hidden rounded-lg border border-line bg-[#12111a]">
        <div className="flex items-center justify-between border-b border-line bg-hover/60 px-3 py-1.5">
          <span className="text-micro font-medium tracking-wide text-fg-muted">
            {labelFor(match[1])}
          </span>
          <CopyButton text={codeString} label="Copy code" />
        </div>
        <div className="overflow-x-auto">
          <SyntaxHighlighter
            style={vscDarkPlus}
            language={match[1]}
            PreTag="pre"
            customStyle={{
              margin: 0,
              borderRadius: 0,
              fontSize: "13px",
              lineHeight: 1.65,
              padding: "14px 16px",
              background: "transparent",
            }}
            // vscDarkPlus paints its own slab on the <code> element, which shows
            // through as a lighter box inside our container — flatten it.
            codeTagProps={{
              style: {
                background: "transparent",
                textShadow: "none",
                fontFamily: "var(--font-mono, ui-monospace, SFMono-Regular, Menlo, monospace)",
              },
            }}
          >
            {codeString}
          </SyntaxHighlighter>
        </div>
      </div>
    );
  },
};

/* ── Message ────────────────────────────────────────────────────── */

function MessageBubbleInner({
  message,
  index,
  isStreaming = false,
  canBranch = false,
  isEditing = false,
  onStartEdit,
  onCancelEdit,
  onSubmitEdit,
  onRetry,
}: MessageBubbleProps) {
  const isUser = message.role === "user";
  const contentStr = String(message.content || "");

  const time = message.timestamp.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });

  if (isUser) {
    if (isEditing) {
      return (
        <div className="flex justify-end">
          <div className="flex w-full max-w-[85%] flex-col items-end sm:max-w-[75%]">
            <MessageEditor
              initialValue={contentStr}
              onCancel={() => onCancelEdit?.()}
              onSubmit={(text) => onSubmitEdit?.(message, text)}
            />
          </div>
        </div>
      );
    }

    return (
      <div
        className="animate-rise group flex justify-end"
        style={{ animationDelay: `${Math.min(index, 6) * 30}ms` }}
      >
        <div className="flex max-w-[85%] flex-col items-end sm:max-w-[75%]">
          <div className="md md-on-user rounded-lg rounded-br-sm border border-accent-muted/40 bg-accent-subtle px-4 py-2.5 text-fg">
            <ReactMarkdown remarkPlugins={REMARK_PLUGINS} components={markdownComponents}>
              {contentStr}
            </ReactMarkdown>
          </div>
          <div className="mt-1 flex items-center gap-1 opacity-0 transition-opacity duration-150 group-hover:opacity-100 focus-within:opacity-100">
            <time className="pr-1 text-micro text-fg-faint">{time}</time>
            <CopyButton text={contentStr} label="Copy message" />
            {onStartEdit && (
              <ActionButton
                label="Edit message"
                icon={Pencil}
                disabled={!canBranch}
                onClick={() => onStartEdit(message)}
              />
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className="animate-rise group flex gap-3"
      style={{ animationDelay: `${Math.min(index, 6) * 30}ms` }}
    >
      <span
        className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-accent-muted/50 bg-accent-subtle"
        aria-hidden
      >
        <Sparkles className="h-3.5 w-3.5 text-accent-fg" strokeWidth={1.75} />
      </span>

      <div className="min-w-0 flex-1">
        <div
          className={cn("md message-body", isStreaming && "streaming-cursor")}
        >
          {contentStr ? (
            <ReactMarkdown remarkPlugins={REMARK_PLUGINS} components={markdownComponents}>
              {/* Mid-stream the buffer is usually cut mid-token, so the
                  half-written markup is closed for display only -- see
                  closeDanglingMarkup. The stored message is never touched. */}
              {isStreaming ? closeDanglingMarkup(contentStr) : contentStr}
            </ReactMarkdown>
          ) : null}
        </div>

        {/* Action row — revealed on hover, and always present for keyboard users */}
        {!isStreaming && contentStr && (
          <div className="mt-1.5 flex items-center gap-1 opacity-0 transition-opacity duration-150 group-hover:opacity-100 focus-within:opacity-100">
            <CopyButton text={contentStr} label="Copy response" />
            {onRetry && (
              <ActionButton
                label="Regenerate response"
                icon={RotateCcw}
                disabled={!canBranch}
                onClick={() => onRetry(message)}
              />
            )}
            <time className="text-micro text-fg-faint">{time}</time>
          </div>
        )}
      </div>
    </div>
  );
}

// Skip re-rendering settled messages while tokens stream into the newest one.
const MessageBubble = memo(MessageBubbleInner, (prev, next) => {
  if (prev.isStreaming !== next.isStreaming) return false;
  if (next.isStreaming) return false;
  // Both drive what the action row renders, so a stale bubble would keep
  // offering Edit during a generation, or keep the editor open after cancel.
  if (prev.isEditing !== next.isEditing) return false;
  if (prev.canBranch !== next.canBranch) return false;
  return prev.message.content === next.message.content && prev.message.id === next.message.id;
});

MessageBubble.displayName = "MessageBubble";

export default MessageBubble;

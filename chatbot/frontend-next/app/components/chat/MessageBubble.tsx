"use client";

import { closeDanglingMarkup } from "@/app/lib/streamingMarkdown";
import type { ToolActivity } from "@/app/lib/tools";
import { cn } from "@/app/lib/utils";
import { Check, Copy, Pencil, RotateCcw } from "lucide-react";
import {
  isValidElement,
  memo,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactElement,
} from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import ToolActivityList from "../tools/ToolCard";
import Button from "../ui/Button";
import Logo from "../ui/Logo";

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
  /**
   * Set only when the model that answered was NOT the one picked, e.g. the
   * pick was rate-limited and the chain fell through. Live-stream only: a
   * reloaded conversation has no record of which model wrote an old reply.
   */
  answeredBy?: string | null;
  /** Tool calls made while producing this answer. Live-stream only. */
  tools?: ToolActivity[];
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
        "inline-flex h-7 w-7 items-center justify-center rounded-full text-micro font-medium",
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
        "inline-flex h-7 w-7 items-center justify-center rounded-full text-micro font-medium",
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
    <div className="glass-strong w-full rounded-3xl px-4 py-3 shadow-focus">
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
  // Every fenced block is rendered here, at the <pre>, so a block with no
  // language (typically program output) gets the same card as real code.
  // Inline `code` never reaches this path: it has no <pre> around it.
  pre({ children }: { children?: React.ReactNode }) {
    const child = isValidElement(children)
      ? (children as ReactElement<{ className?: string; children?: React.ReactNode }>)
      : null;
    const lang = /language-(\w+)/.exec(child?.props.className || "")?.[1] ?? null;
    const codeString = String(child?.props.children ?? "").replace(/\n$/, "");
    return <CodeBlock language={lang} code={codeString} />;
  },
};

// Languages that are really "what the program printed".
const OUTPUT_LANGS = new Set(["text", "txt", "plaintext", "output", "console", "log"]);

/**
 * A code or output block: the one place an answer gets a white card. Token
 * colours come from CSS (see `.code-block` in globals.css) rather than an
 * inline Prism theme, so they follow the light/dark switch.
 */
function CodeBlock({ language, code }: { language: string | null; code: string }) {
  const isOutput = !language || OUTPUT_LANGS.has(language.toLowerCase());
  return (
    <div className="code-block my-4 overflow-hidden rounded-2xl border border-line-subtle bg-raised shadow-e1">
      <div className="flex items-center justify-between border-b border-line-subtle bg-hover py-1 pl-4 pr-1.5">
        <span className="text-micro font-semibold tracking-wide text-fg-subtle">
          {isOutput ? "Output" : labelFor(language!)}
        </span>
        <CopyButton text={code} label={isOutput ? "Copy output" : "Copy code"} />
      </div>
      <div className="overflow-x-auto">
        {isOutput ? (
          <pre className="code-pre">
            <code>{code}</code>
          </pre>
        ) : (
          <SyntaxHighlighter
            language={language!}
            PreTag="pre"
            // An empty theme: the library otherwise falls back to its default
            // one and still inlines black text on the <code> element.
            style={{}}
            useInlineStyles={false}
            className="code-pre"
          >
            {code}
          </SyntaxHighlighter>
        )}
      </div>
    </div>
  );
}

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
        <div className="flex max-w-[88%] flex-col items-end sm:max-w-[75%]">
          <div className="md md-on-user bg-brand rounded-3xl rounded-br-lg px-4.5 py-3 shadow-glow">
            <ReactMarkdown remarkPlugins={REMARK_PLUGINS} components={markdownComponents}>
              {contentStr}
            </ReactMarkdown>
          </div>
          <div className="mt-1 flex items-center gap-0.5 opacity-0 transition-opacity duration-150 group-hover:opacity-100 focus-within:opacity-100">
            <time className="pr-1 text-micro text-fg-subtle">{time}</time>
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

  const tools = message.tools ?? [];

  return (
    <div
      className="animate-rise group flex gap-3"
      style={{ animationDelay: `${Math.min(index, 6) * 30}ms` }}
    >
      {/* rounded-lg follows the mark's own corner radius so the shadow does too */}
      <Logo size={30} className="mt-0.5 hidden rounded-lg shadow-e2 sm:block" />

      <div className="flex min-w-0 flex-1 flex-col gap-2.5">
        {/* Tool cards are progress, not content: they show while the turn is
            working and give way to the answer once its text arrives. A reset
            (the model going back for another tool) empties the text, so they
            come back for that round. */}
        {tools.length > 0 && !contentStr && <ToolActivityList tools={tools} />}

        {/* Answer text sits directly on the page; only code and output
            blocks inside it get a card (see CodeBlock). */}
        {contentStr && (
          <div className="pt-1">
            <div className={cn("md message-body", isStreaming && "streaming-cursor")}>
              <ReactMarkdown remarkPlugins={REMARK_PLUGINS} components={markdownComponents}>
                {/* Mid-stream the buffer is usually cut mid-token, so the
                    half-written markup is closed for display only -- see
                    closeDanglingMarkup. The stored message is never touched. */}
                {isStreaming ? closeDanglingMarkup(contentStr) : contentStr}
              </ReactMarkdown>
            </div>
          </div>
        )}

        {/* Action row — revealed on hover, and always present for keyboard users */}
        {!isStreaming && contentStr && (
          <div className="-mt-1 flex items-center gap-0.5 pl-1 opacity-0 transition-opacity duration-150 group-hover:opacity-100 focus-within:opacity-100">
            <CopyButton text={contentStr} label="Copy response" />
            {onRetry && (
              <ActionButton
                label="Regenerate response"
                icon={RotateCcw}
                disabled={!canBranch}
                onClick={() => onRetry(message)}
              />
            )}
            <time className="pl-1 text-micro text-fg-subtle">{time}</time>
            {message.answeredBy && (
              <span
                className="text-micro text-fg-subtle"
                title="Your chosen model was unavailable, so the next one in the fallback chain answered."
              >
                · answered by {message.answeredBy}
              </span>
            )}
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
  if (prev.message.answeredBy !== next.message.answeredBy) return false;
  // Tool cards update while the answer has no text yet, i.e. before the
  // bubble counts as streaming, so they need their own check.
  if (prev.message.tools !== next.message.tools) return false;
  return prev.message.content === next.message.content && prev.message.id === next.message.id;
});

MessageBubble.displayName = "MessageBubble";

export default MessageBubble;

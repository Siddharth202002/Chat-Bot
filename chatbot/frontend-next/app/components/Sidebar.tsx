"use client";

import { useFocusTrap } from "@/app/lib/useFocusTrap";
import { cn } from "@/app/lib/utils";
import {
  AlertCircle,
  FileText,
  MessageSquare,
  PanelLeft,
  PanelLeftClose,
  Plus,
  Search,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState, type DragEvent } from "react";
import Button, { IconButton } from "./ui/Button";
import { Badge, EmptyState, Skeleton } from "./ui/Primitives";

export interface ChatSummary {
  id: string;
  title: string;
}

export interface RagState {
  status: "idle" | "uploading" | "ready" | "error";
  fileName: string | null;
  pages: number;
  chunks: number;
  message: string;
}

interface SidebarProps {
  chatHistory: ChatSummary[];
  threadId: string;
  historyLoading: boolean;
  historyError: boolean;
  onRetryHistory: () => void;
  onNewChat: () => void;
  onLoadChat: (id: string) => void;
  onRequestDelete: (chat: ChatSummary) => void;
  onRequestDeleteAll: () => void;
  rag: RagState;
  isOpen: boolean;
  isDesktop: boolean;
  onClose: () => void;
  onToggle: () => void;
}

export default function Sidebar({
  chatHistory,
  threadId,
  historyLoading,
  historyError,
  onRetryHistory,
  onNewChat,
  onLoadChat,
  onRequestDelete,
  onRequestDeleteAll,
  rag,
  isOpen,
  isDesktop,
  onClose,
  onToggle,
}: SidebarProps) {
  const [query, setQuery] = useState("");
  const panelRef = useRef<HTMLElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  // Only trap focus when the drawer floats above the content (mobile).
  const isModal = isOpen && !isDesktop;
  useFocusTrap(panelRef, isModal, onClose);

  // Clear a stale filter when the panel is collapsed.
  useEffect(() => {
    if (!isOpen) setQuery("");
  }, [isOpen]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return chatHistory;
    return chatHistory.filter((c) => c.title.toLowerCase().includes(q));
  }, [chatHistory, query]);

  function focusSearch() {
    if (!isOpen) onToggle();
    requestAnimationFrame(() => searchRef.current?.focus());
  }

  /* ── Collapsed rail (desktop only) ──────────────────────────── */

  const rail = (
    <div className="flex w-14 flex-col items-center gap-1.5 py-3">
      <IconButton label="Expand sidebar (Ctrl+B)" onClick={onToggle}>
        <PanelLeft className="h-4.5 w-4.5" strokeWidth={1.75} />
      </IconButton>
      <IconButton label="New chat" onClick={onNewChat}>
        <Plus className="h-4.5 w-4.5" strokeWidth={1.75} />
      </IconButton>
      <IconButton label="Search chats" onClick={focusSearch}>
        <Search className="h-4.5 w-4.5" strokeWidth={1.75} />
      </IconButton>
    </div>
  );

  /* ── Chat list body ─────────────────────────────────────────── */

  let listBody: React.ReactNode;

  if (historyLoading) {
    listBody = (
      <div className="flex flex-col gap-1 px-2" aria-busy="true" aria-label="Loading chats">
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} className="h-9 rounded-md" />
        ))}
      </div>
    );
  } else if (historyError) {
    listBody = (
      <EmptyState
        icon={AlertCircle}
        title="Couldn't reach the server"
        description="Check that the API is running, then try again."
        action={
          <Button variant="secondary" size="sm" onClick={onRetryHistory}>
            Retry
          </Button>
        }
      />
    );
  } else if (chatHistory.length === 0) {
    listBody = (
      <EmptyState
        icon={MessageSquare}
        title="No conversations yet"
        description="Your chats will appear here."
      />
    );
  } else if (filtered.length === 0) {
    listBody = (
      <EmptyState icon={Search} title="No matches" description={`Nothing matches “${query}”.`} />
    );
  } else {
    listBody = (
      <ul className="flex flex-col gap-0.5 px-2">
        {filtered.map((chat) => {
          const isActive = chat.id === threadId;
          return (
            <li key={chat.id}>
              <div
                className={cn(
                  "group relative flex items-center rounded-md",
                  "transition-colors duration-150 ease-standard",
                  isActive
                    ? "border border-accent-muted/60 bg-accent-subtle"
                    : "border border-transparent hover:bg-hover"
                )}
              >
                <button
                  type="button"
                  onClick={() => onLoadChat(chat.id)}
                  aria-current={isActive ? "page" : undefined}
                  title={chat.title}
                  className={cn(
                    "flex min-w-0 flex-1 items-center gap-2 rounded-md py-2 pl-2.5 pr-1 text-left",
                    isActive ? "text-fg" : "text-fg-muted group-hover:text-fg"
                  )}
                >
                  <MessageSquare
                    className="h-3.5 w-3.5 shrink-0 opacity-60"
                    strokeWidth={1.75}
                    aria-hidden
                  />
                  <span className="truncate text-small">{chat.title}</span>
                </button>

                <button
                  type="button"
                  onClick={() => onRequestDelete(chat)}
                  aria-label={`Delete chat: ${chat.title}`}
                  title="Delete chat"
                  className={cn(
                    "mr-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-sm",
                    "text-fg-subtle transition-all duration-150",
                    "opacity-0 group-hover:opacity-100 focus-visible:opacity-100",
                    "hover:bg-danger-subtle hover:text-danger"
                  )}
                >
                  <Trash2 className="h-3.5 w-3.5" strokeWidth={1.75} />
                </button>
              </div>
            </li>
          );
        })}
      </ul>
    );
  }

  /* ── Indexed document status ────────────────────────────────
     Read-only on purpose. Uploading lives in the composer now, so the sidebar
     reports what is indexed without offering a second way to change it. */

  const ragCard =
    rag.status === "ready" ? (
      <div className="rounded-md border border-line bg-canvas/60 p-2.5">
        <div className="flex items-start gap-2">
          <FileText className="mt-0.5 h-4 w-4 shrink-0 text-accent-fg" strokeWidth={1.75} />
          <div className="min-w-0 flex-1">
            <p className="truncate text-small font-medium text-fg" title={rag.fileName ?? ""}>
              {rag.fileName ?? "Document"}
            </p>
            <p className="mt-0.5 text-micro text-fg-subtle">
              {rag.pages} pages · {rag.chunks} chunks
            </p>
          </div>
          <Badge tone="success">Indexed</Badge>
        </div>
      </div>
    ) : rag.status === "uploading" ? (
      <div className="rounded-md border border-line bg-canvas/60 p-2.5">
        <p className="truncate text-small text-fg-muted">{rag.message}</p>
        <div className="progress-indeterminate mt-2 h-1 rounded-full bg-hover" />
      </div>
    ) : rag.status === "error" ? (
      <div className="rounded-md border border-danger/25 bg-danger-subtle p-2.5">
        <p className="text-micro text-danger">{rag.message}</p>
      </div>
    ) : null;

  /* ── Expanded panel ─────────────────────────────────────────── */

  const panel = (
    <>
      {/* Brand + collapse */}
      <div className="flex h-14 shrink-0 items-center gap-2.5 border-b border-line-subtle px-3">
        <span
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-accent"
          aria-hidden
        >
          <Sparkles className="h-3.5 w-3.5 text-white" strokeWidth={2} />
        </span>
        <div className="min-w-0 flex-1">
          <p className="truncate text-small font-semibold text-fg">Zeno AI</p>
          <p className="truncate text-micro text-fg-subtle">Powered by LangGraph</p>
        </div>
        <IconButton label="Close sidebar (Ctrl+B)" size="sm" onClick={onClose}>
          <PanelLeftClose className="h-4 w-4 md:block hidden" strokeWidth={1.75} />
          <X className="h-4 w-4 md:hidden" strokeWidth={1.75} />
        </IconButton>
      </div>

      {/* Actions */}
      <div className="flex shrink-0 flex-col gap-2.5 p-3">
        <Button variant="primary" size="md" block onClick={onNewChat}>
          <Plus className="h-4 w-4" strokeWidth={2} />
          New chat
        </Button>
        {ragCard}
        {/* Only offered when there is actually something to clear. */}
        {chatHistory.length > 0 && (
          <button
            type="button"
            onClick={onRequestDeleteAll}
            className={cn(
              "flex items-center justify-center gap-1.5 rounded-md py-1.5",
              "text-micro font-medium text-fg-subtle",
              "transition-colors duration-150 ease-standard",
              "hover:bg-danger-subtle hover:text-danger"
            )}
          >
            <Trash2 className="h-3.5 w-3.5" strokeWidth={1.75} aria-hidden />
            Delete all chats
          </button>
        )}
      </div>

      {/* Search */}
      <div className="shrink-0 px-3 pb-2">
        <div className="flex items-center gap-2 rounded-md border border-line bg-canvas/60 px-2.5 transition-colors duration-150 focus-within:border-focus">
          <Search className="h-3.5 w-3.5 shrink-0 text-fg-subtle" strokeWidth={1.75} aria-hidden />
          <input
            ref={searchRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search chats"
            aria-label="Search chats"
            className="min-w-0 flex-1 bg-transparent py-1.5 text-small text-fg outline-none placeholder:text-fg-subtle"
          />
          {query && (
            <button
              type="button"
              onClick={() => setQuery("")}
              aria-label="Clear search"
              className="shrink-0 rounded-sm p-0.5 text-fg-subtle transition-colors hover:text-fg"
            >
              <X className="h-3 w-3" strokeWidth={2} />
            </button>
          )}
        </div>
      </div>

      {/* History */}
      <div className="min-h-0 flex-1 overflow-y-auto pb-2">
        {!historyLoading && !historyError && chatHistory.length > 0 && (
          <p className="px-4 pb-1.5 pt-1 text-micro font-semibold uppercase tracking-widest text-fg-faint">
            Recent
          </p>
        )}
        {listBody}
      </div>
    </>
  );

  /* ── Composition ────────────────────────────────────────────────
     One element for every breakpoint: below `md` it is a fixed drawer that
     slides in; at `md` and up it sits in the flow and animates its width
     between the rail and the full panel. Keeping it as a single node avoids
     a mount flash and lets the collapse actually animate. */

  return (
    <>
      {/* Backdrop — mobile only, purely CSS-gated so it never flashes */}
      {isOpen && (
        <div
          className="animate-fade-in fixed inset-0 z-60 bg-black/60 md:hidden"
          onClick={onClose}
          aria-hidden
        />
      )}

      <aside
        ref={panelRef}
        aria-label="Chats"
        aria-modal={isModal || undefined}
        role={isModal ? "dialog" : undefined}
        className={cn(
          "z-70 flex h-full flex-col overflow-hidden border-r border-line-subtle bg-raised",
          "fixed inset-y-0 left-0 w-68 shadow-e3 transition-transform duration-200 ease-standard",
          isOpen ? "translate-x-0" : "invisible -translate-x-full",
          // From `md` up it lives in the flow and animates its width instead.
          "md:visible md:relative md:z-auto md:shrink-0 md:translate-x-0 md:shadow-none",
          "md:transition-[width] md:duration-200 md:ease-standard",
          isOpen ? "md:w-68" : "md:w-14"
        )}
      >
        {isOpen ? panel : rail}
      </aside>
    </>
  );
}

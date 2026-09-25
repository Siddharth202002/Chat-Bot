"use client";

import type { Theme } from "@/app/lib/theme";
import { useFocusTrap } from "@/app/lib/useFocusTrap";
import { cn } from "@/app/lib/utils";
import {
  AlertCircle,
  FileText,
  MessageCircle,
  MessageSquare,
  PanelLeft,
  Search,
  SquarePen,
  Trash2,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import Button, { IconButton } from "../ui/Button";
import Logo from "../ui/Logo";
import { Badge, EmptyState, Skeleton } from "../ui/Primitives";
import ProfileMenu from "./ProfileMenu";

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
  userEmail: string;
  onLogout: () => void;
  theme: Theme;
  onThemeChange: (theme: Theme) => void;
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
  userEmail,
  onLogout,
  theme,
  onThemeChange,
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

  const railButton = "h-11 w-11 rounded-2xl text-fg-muted hover:bg-raised/70 hover:text-fg hover:shadow-e1";

  const rail = (
    <div className="flex h-full w-17 flex-col items-center gap-2 py-4">
      <Logo size={34} className="mb-2 rounded-[10px] shadow-e2" />
      <IconButton label="Expand sidebar (Ctrl+B)" onClick={onToggle} className={railButton}>
        <PanelLeft className="h-4.5 w-4.5" strokeWidth={1.9} />
      </IconButton>
      <IconButton label="New chat" onClick={onNewChat} className={railButton}>
        <SquarePen className="h-4.5 w-4.5" strokeWidth={1.9} />
      </IconButton>
      <IconButton label="Search chats" onClick={focusSearch} className={railButton}>
        <Search className="h-4.5 w-4.5" strokeWidth={1.9} />
      </IconButton>
      <div className="flex-1" />
      <ProfileMenu
        variant="compact"
        userEmail={userEmail}
        onLogout={onLogout}
        theme={theme}
        onThemeChange={onThemeChange}
      />
    </div>
  );

  /* ── Chat list body ─────────────────────────────────────────── */

  let listBody: React.ReactNode;

  if (historyLoading) {
    listBody = (
      <div className="flex flex-col gap-1.5 px-3" aria-busy="true" aria-label="Loading chats">
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} className="h-10 rounded-2xl" />
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
      <ul className="flex flex-col gap-0.5 px-3">
        {filtered.map((chat) => {
          const isActive = chat.id === threadId;
          return (
            <li key={chat.id}>
              <div
                className={cn(
                  "group relative flex items-center rounded-2xl",
                  "transition-[background-color,box-shadow] duration-150 ease-standard",
                  isActive ? "bg-raised shadow-e1" : "hover:bg-raised/60"
                )}
              >
                <button
                  type="button"
                  onClick={() => onLoadChat(chat.id)}
                  aria-current={isActive ? "page" : undefined}
                  title={chat.title}
                  className={cn(
                    "flex min-w-0 flex-1 items-center gap-2.5 rounded-2xl py-2.5 pl-3 pr-1 text-left",
                    isActive ? "text-fg" : "text-fg-muted group-hover:text-fg"
                  )}
                >
                  <MessageCircle
                    className={cn(
                      "h-4 w-4 shrink-0",
                      isActive ? "text-accent-fg" : "text-fg-subtle"
                    )}
                    strokeWidth={1.9}
                    aria-hidden
                  />
                  <span className={cn("truncate text-small", isActive && "font-medium")}>
                    {chat.title}
                  </span>
                </button>

                <button
                  type="button"
                  onClick={() => onRequestDelete(chat)}
                  aria-label={`Delete chat: ${chat.title}`}
                  title="Delete chat"
                  className={cn(
                    "mr-1.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full",
                    "text-fg-subtle transition-all duration-150",
                    "opacity-0 group-hover:opacity-100 focus-visible:opacity-100",
                    "hover:bg-danger-subtle hover:text-danger",
                    // Touch screens have no hover, so keep it reachable there.
                    "max-md:opacity-60"
                  )}
                >
                  <Trash2 className="h-3.5 w-3.5" strokeWidth={1.9} />
                </button>
              </div>
            </li>
          );
        })}
      </ul>
    );
  }

  /* ── Indexed document status ────────────────────────────────
     Read-only on purpose. Uploading lives in the composer, so the sidebar
     reports what is indexed without offering a second way to change it. */

  const ragCard =
    rag.status === "ready" ? (
      <div className="surface-card rounded-2xl p-3 shadow-e1">
        <div className="flex items-center gap-2.5">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-violet-500/12 text-violet-600 dark:text-violet-300">
            <FileText className="h-4.5 w-4.5" strokeWidth={1.9} aria-hidden />
          </span>
          <div className="min-w-0 flex-1">
            <p className="truncate text-small font-medium text-fg" title={rag.fileName ?? ""}>
              {rag.fileName ?? "Document"}
            </p>
            <p className="text-micro text-fg-subtle">
              {rag.pages} pages · {rag.chunks} chunks
            </p>
          </div>
          <Badge tone="success">Indexed</Badge>
        </div>
      </div>
    ) : rag.status === "uploading" ? (
      <div className="surface-card rounded-2xl p-3 shadow-e1">
        <p className="truncate text-small text-fg-muted">{rag.message}</p>
        <div className="progress-indeterminate mt-2 h-1 rounded-full bg-active" />
      </div>
    ) : rag.status === "error" ? (
      <div className="rounded-2xl border border-danger/20 bg-danger-subtle p-3">
        <p className="text-micro text-danger">{rag.message}</p>
      </div>
    ) : null;

  /* ── Expanded panel ─────────────────────────────────────────── */

  const panel = (
    <div className="flex h-full w-71 max-w-full flex-col">
      {/* Brand + collapse */}
      <div className="flex h-16 shrink-0 items-center gap-2.5 px-5 pt-1">
        <Logo size={32} className="rounded-[9px] shadow-e2" />
        <p className="min-w-0 flex-1 truncate text-[1.0625rem] font-bold tracking-tight text-fg">
          Zeno <span className="text-brand">AI</span>
        </p>
        <IconButton label="Close sidebar (Ctrl+B)" size="sm" onClick={onClose}>
          <PanelLeft className="hidden h-4.5 w-4.5 md:block" strokeWidth={1.9} />
          <X className="h-4.5 w-4.5 md:hidden" strokeWidth={1.9} />
        </IconButton>
      </div>

      {/* Search + primary action */}
      <div className="flex shrink-0 flex-col gap-2 px-4 pb-3 pt-2">
        <div
          className={cn(
            "flex h-11 items-center gap-2.5 rounded-full bg-accent-subtle/80 px-4",
            "ring-1 ring-transparent transition-[box-shadow,background-color] duration-150",
            "focus-within:bg-raised focus-within:shadow-focus"
          )}
        >
          <Search className="h-4 w-4 shrink-0 text-fg-subtle" strokeWidth={1.9} aria-hidden />
          <input
            ref={searchRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search"
            aria-label="Search chats"
            className="min-w-0 flex-1 bg-transparent text-small text-fg outline-none placeholder:text-fg-subtle"
          />
          {query && (
            <button
              type="button"
              onClick={() => setQuery("")}
              aria-label="Clear search"
              className="shrink-0 rounded-full p-1 text-fg-subtle transition-colors hover:bg-hover hover:text-fg"
            >
              <X className="h-3 w-3" strokeWidth={2.25} />
            </button>
          )}
        </div>

        <button
          type="button"
          onClick={onNewChat}
          className={cn(
            "group flex h-11 items-center gap-2.5 rounded-full bg-accent-subtle/80 pl-1.5 pr-4",
            "text-small font-medium text-fg",
            "transition-[background-color,box-shadow] duration-150 ease-standard",
            "hover:bg-raised hover:shadow-e1 active:scale-[0.99]"
          )}
        >
          <span className="bg-brand flex h-8 w-8 items-center justify-center rounded-full text-white shadow-glow transition-transform duration-150 group-hover:scale-105">
            <SquarePen className="h-4 w-4" strokeWidth={2} aria-hidden />
          </span>
          New chat
        </button>

        {ragCard}
      </div>

      {/* History */}
      <div className="flex shrink-0 items-center justify-between px-5 pb-1.5 pt-1">
        <p className="text-small font-semibold text-fg">Chat history</p>
        {/* Only offered when there is actually something to clear. */}
        {chatHistory.length > 0 && (
          <IconButton
            label="Delete all chats"
            size="sm"
            onClick={onRequestDeleteAll}
            className="h-7 w-7 text-fg-subtle hover:bg-danger-subtle hover:text-danger"
          >
            <Trash2 className="h-3.5 w-3.5" strokeWidth={1.9} />
          </IconButton>
        )}
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto pb-2">{listBody}</div>

      {/* Account */}
      <div className="shrink-0 p-3 pt-2">
        <ProfileMenu
          variant="full"
          userEmail={userEmail}
          onLogout={onLogout}
          theme={theme}
          onThemeChange={onThemeChange}
        />
      </div>
    </div>
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
          className="animate-fade-in fixed inset-0 z-60 bg-fg/20 backdrop-blur-[3px] md:hidden"
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
          "z-70 flex h-full flex-col overflow-hidden",
          // Phone: a floating glass drawer.
          "glass-strong fixed inset-y-2 left-2 h-auto w-[min(18rem,calc(100vw-3rem))] rounded-3xl shadow-e3",
          "transition-transform duration-250 ease-out-soft",
          isOpen ? "translate-x-0" : "invisible -translate-x-[110%]",
          // From `md` up it lives in the flow and animates its width instead.
          "md:visible md:relative md:inset-auto md:z-auto md:h-full md:shrink-0 md:translate-x-0",
          "md:rounded-none md:border-0 md:border-r md:border-line-subtle md:bg-[var(--sidebar-bg)] md:shadow-none md:backdrop-blur-none",
          "md:transition-[width] md:duration-200 md:ease-standard",
          isOpen ? "md:w-71" : "md:w-17"
        )}
      >
        {isOpen ? panel : rail}
      </aside>
    </>
  );
}

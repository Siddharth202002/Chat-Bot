"use client";

import { cn } from "@/app/lib/utils";
import { AlertCircle, ArrowDown, RotateCcw } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import MessageBubble, { type Message } from "./MessageBubble";
import TypingIndicator from "./TypingIndicator";
import WelcomeHero from "./WelcomeHero";
import Button from "../ui/Button";
import { Skeleton } from "../ui/Primitives";

interface ChatAreaProps {
  messages: Message[];
  isLoading: boolean;
  isStreaming: boolean;
  isLoadingHistory: boolean;
  /** Text of a send that failed, so the turn can be retried in place. */
  failedMessage: string | null;
  onRetry: () => void;
  /** Client id of the user message currently open in the inline editor. */
  editingId: string | null;
  onStartEdit: (message: Message) => void;
  onCancelEdit: () => void;
  onSubmitEdit: (message: Message, text: string) => void;
  onRegenerate: (message: Message) => void;
}

/** Placeholder turns shown while an existing thread is being fetched. */
function HistorySkeleton() {
  return (
    <div className="flex flex-col gap-8" aria-busy="true" aria-label="Loading conversation">
      {[0, 1, 2].map((i) => (
        <div key={i} className="flex flex-col gap-8">
          <div className="flex justify-end">
            <Skeleton className="h-11 w-[45%] rounded-3xl" />
          </div>
          <div className="flex gap-3">
            <Skeleton className="h-7.5 w-7.5 shrink-0 rounded-lg" />
            <div className="flex w-full flex-col gap-2.5 pt-1.5">
              <Skeleton className="h-3.5 w-full rounded-full" />
              <Skeleton className="h-3.5 w-[92%] rounded-full" />
              <Skeleton className="h-3.5 w-[60%] rounded-full" />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function ChatArea({
  messages,
  isLoading,
  isStreaming,
  isLoadingHistory,
  failedMessage,
  onRetry,
  editingId,
  onStartEdit,
  onCancelEdit,
  onSubmitEdit,
  onRegenerate,
}: ChatAreaProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [userScrolledUp, setUserScrolledUp] = useState(false);

  const isNearBottom = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return true;
    return el.scrollHeight - el.scrollTop - el.clientHeight < 100;
  }, []);

  const scrollToBottom = useCallback((instant = false) => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: instant ? "instant" : "smooth" });
  }, []);

  const handleScroll = useCallback(() => {
    setUserScrolledUp(!isNearBottom());
  }, [isNearBottom]);

  // Follow the conversation unless the reader has deliberately scrolled away.
  useEffect(() => {
    // Nothing to follow on the welcome screen — scrolling there would push the
    // hero up under the header on short viewports.
    if (messages.length === 0) return;
    if (userScrolledUp) return;
    if (isStreaming) {
      requestAnimationFrame(() => scrollToBottom(true));
    } else {
      scrollToBottom(false);
    }
  }, [messages, isLoading, isStreaming, userScrolledUp, scrollToBottom]);

  // Sending always snaps back to the bottom.
  const prevCount = useRef(messages.length);
  useEffect(() => {
    if (messages.length > prevCount.current) {
      if (messages[messages.length - 1]?.role === "user") {
        setUserScrolledUp(false);
        scrollToBottom(true);
      }
    }
    prevCount.current = messages.length;
  }, [messages, scrollToBottom]);

  const isEmpty = messages.length === 0 && !isLoading && !isLoadingHistory;

  // While a tool is running its card is the progress indicator, so the
  // generic "thinking" bubble would only repeat it.
  const last = messages[messages.length - 1];
  const toolRunning =
    last?.role === "assistant" && (last.tools ?? []).some((t) => t.status === "running");

  return (
    // The welcome screen never scrolls: it keeps its natural height (the page
    // spacer below the chips gives way first) and has no scroll container.
    <div className={cn("relative flex flex-col", isEmpty ? "flex-[1_0_auto]" : "min-h-0 flex-1")}>
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className={cn(
          "flex flex-1 flex-col",
          isEmpty
            ? "overflow-hidden"
            : // Apple-style scroll edge: messages fade into the top bar
              // instead of being cut off under it.
              "scroll-edge-fade min-h-0 overflow-y-auto"
        )}
      >
        {isEmpty ? (
          /* ══ Welcome hero ══ */
          /* On md+ the hero sits at the bottom of this region and the page adds
             a matching spacer under the composer, so hero + composer + chips
             read as one centred group. On phones it stays centred and the
             composer docks at the bottom. */
          <div className="flex flex-1 items-center justify-center px-4 py-6 sm:px-6 md:items-end md:pb-7 [@media(max-height:760px)]:py-3">
            <WelcomeHero />
          </div>
        ) : (
          /* ══ Message feed ══ */
          <div className="flex flex-1 flex-col px-3 py-6 sm:px-6 lg:px-10">
            <div className="mx-auto mt-auto flex w-full max-w-3xl flex-col gap-7 pb-2">
              {isLoadingHistory ? (
                <HistorySkeleton />
              ) : (
                <>
                  {messages
                    .filter((msg, i) => {
                      // Hide the empty assistant placeholder while the dots are
                      // showing -- unless it already has tool cards to show.
                      if (
                        isLoading &&
                        !isStreaming &&
                        msg.role === "assistant" &&
                        !msg.content &&
                        !msg.tools?.length &&
                        i === messages.length - 1
                      ) {
                        return false;
                      }
                      return true;
                    })
                    .map((msg, i) => (
                      <MessageBubble
                        key={msg.id}
                        message={msg}
                        index={i}
                        isStreaming={
                          isStreaming &&
                          msg.role === "assistant" &&
                          msg.id === messages[messages.length - 1]?.id
                        }
                        // Rewriting history needs the stored id, and needs the
                        // thread to be idle: a fork while a turn is in flight
                        // is the one thing the server refuses outright.
                        canBranch={Boolean(msg.serverId) && !isLoading}
                        isEditing={editingId === msg.id}
                        onStartEdit={onStartEdit}
                        onCancelEdit={onCancelEdit}
                        onSubmitEdit={onSubmitEdit}
                        onRetry={onRegenerate}
                      />
                    ))}

                  {isLoading && !isStreaming && !toolRunning && <TypingIndicator />}

                  {failedMessage && (
                    <div
                      role="alert"
                      className="animate-rise flex flex-col gap-3 rounded-3xl border border-danger/20 bg-danger-subtle p-4 sm:flex-row sm:items-center sm:justify-between"
                    >
                      <div className="flex items-start gap-2.5">
                        <AlertCircle
                          className="mt-0.5 h-4 w-4 shrink-0 text-danger"
                          strokeWidth={1.75}
                          aria-hidden
                        />
                        <div className="min-w-0">
                          <p className="text-small font-medium text-fg">Message failed to send</p>
                          <p className="mt-0.5 text-micro text-fg-muted">
                            The server didn&apos;t respond. Check that the API is running.
                          </p>
                        </div>
                      </div>
                      <Button variant="secondary" size="sm" onClick={onRetry} className="sm:ml-3">
                        <RotateCcw className="h-3.5 w-3.5" strokeWidth={1.75} />
                        Retry
                      </Button>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Anchored to the viewport of the scroller, not the scrolled content */}
      {userScrolledUp && messages.length > 0 && (
        <button
          type="button"
          onClick={() => {
            setUserScrolledUp(false);
            scrollToBottom(false);
          }}
          aria-label="Scroll to latest message"
          title="Scroll to latest message"
          className={cn(
            "animate-pop-in absolute bottom-3 left-1/2 z-20 flex h-9 w-9 -translate-x-1/2 items-center justify-center",
            "glass-strong rounded-full text-fg-muted shadow-e2",
            "transition-colors duration-150 hover:text-accent-fg"
          )}
        >
          <ArrowDown className="h-4 w-4" strokeWidth={1.75} />
        </button>
      )}
    </div>
  );
}

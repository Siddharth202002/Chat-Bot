"use client";

import { cn } from "@/app/lib/utils";
import { AlertCircle, ArrowDown, RotateCcw, Sparkles } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import MessageBubble, { type Message } from "./MessageBubble";
import TypingIndicator from "./TypingIndicator";
import Button from "./ui/Button";
import { Skeleton } from "./ui/Primitives";

interface ChatAreaProps {
  messages: Message[];
  isLoading: boolean;
  isStreaming: boolean;
  isLoadingHistory: boolean;
  /** Text of a send that failed, so the turn can be retried in place. */
  failedMessage: string | null;
  onRetry: () => void;
}

/** Placeholder turns shown while an existing thread is being fetched. */
function HistorySkeleton() {
  return (
    <div className="flex flex-col gap-8" aria-busy="true" aria-label="Loading conversation">
      {[0, 1, 2].map((i) => (
        <div key={i} className="flex flex-col gap-8">
          <div className="flex justify-end">
            <Skeleton className="h-9 w-[45%] rounded-lg" />
          </div>
          <div className="flex gap-3">
            <Skeleton className="h-7 w-7 shrink-0 rounded-full" />
            <div className="flex w-full flex-col gap-2">
              <Skeleton className="h-3.5 w-full rounded-sm" />
              <Skeleton className="h-3.5 w-[92%] rounded-sm" />
              <Skeleton className="h-3.5 w-[60%] rounded-sm" />
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

  return (
    <div className="relative flex min-h-0 flex-1 flex-col">
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex min-h-0 flex-1 flex-col overflow-y-auto"
      >
        {isEmpty ? (
          /* ══ Welcome hero ══ */
          <div className="flex flex-1 items-center justify-center px-4 py-8 sm:px-6">
            <div className="flex w-full max-w-xl flex-col items-center text-center">
              <span
                className="animate-rise mb-5 flex h-12 w-12 items-center justify-center rounded-xl bg-accent shadow-e2"
                aria-hidden
              >
                <Sparkles className="h-5.5 w-5.5 text-white" strokeWidth={1.75} />
              </span>
              <h1
                className="animate-rise text-[1.75rem] leading-tight font-bold tracking-tight text-fg sm:text-display"
                style={{ animationDelay: "40ms" }}
              >
                What can I help you with?
              </h1>
              <p
                className="animate-rise mt-3 max-w-md text-body-lg text-fg-muted"
                style={{ animationDelay: "80ms" }}
              >
                Ask a question, paste some code, or attach a PDF and I&apos;ll answer from it.
              </p>
            </div>
          </div>
        ) : (
          /* ══ Message feed ══ */
          <div className="flex flex-1 flex-col px-4 py-6 sm:px-6 lg:px-8">
            <div className="mx-auto mt-auto flex w-full max-w-3xl flex-col gap-8 pb-2">
              {isLoadingHistory ? (
                <HistorySkeleton />
              ) : (
                <>
                  {messages
                    .filter((msg, i) => {
                      // Hide the empty assistant placeholder while the dots are showing.
                      if (
                        isLoading &&
                        !isStreaming &&
                        msg.role === "assistant" &&
                        !msg.content &&
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
                      />
                    ))}

                  {isLoading && !isStreaming && <TypingIndicator />}

                  {failedMessage && (
                    <div
                      role="alert"
                      className="animate-rise flex flex-col gap-3 rounded-lg border border-danger/25 bg-danger-subtle p-3.5 sm:flex-row sm:items-center sm:justify-between"
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
            "animate-fade-in absolute bottom-4 right-4 z-20 flex h-9 w-9 items-center justify-center",
            "rounded-full border border-line-strong bg-overlay text-fg-muted shadow-e2",
            "transition-colors duration-150 hover:bg-hover hover:text-fg"
          )}
        >
          <ArrowDown className="h-4 w-4" strokeWidth={1.75} />
        </button>
      )}
    </div>
  );
}

"use client";

import { cn } from "@/app/lib/utils";
import { useRef, useEffect, useCallback, useState, ChangeEvent, KeyboardEvent } from "react";
import { AnimatePresence } from "framer-motion";
import { Sparkles, Code, Lightbulb, Pen, Search, BookOpen } from "lucide-react";
import MessageBubble, { type Message } from "./MessageBubble";
import TypingIndicator from "./TypingIndicator";

const SUGGESTIONS = [
  { icon: Code, text: "Explain Python decorators", color: "text-blue-400" },
  { icon: Lightbulb, text: "Give me a project idea", color: "text-amber-400" },
  { icon: Pen, text: "Write something creative", color: "text-pink-400" },
  { icon: Search, text: "Teach me something new", color: "text-emerald-400" },
  { icon: Code, text: "Debug my code", color: "text-purple-400" },
  { icon: BookOpen, text: "Summarise a concept", color: "text-cyan-400" },
];

interface ChatAreaProps {
  messages: Message[];
  input: string;
  isLoading: boolean;
  isStreaming: boolean;
  onInputChange: (value: string) => void;
  onSendMessage: (text: string) => void;
  className?: string;
  style?: React.CSSProperties;
}

export default function ChatArea({
  messages,
  input,
  isLoading,
  isStreaming,
  onInputChange,
  onSendMessage,
  className,
  style,
}: ChatAreaProps) {
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [userScrolledUp, setUserScrolledUp] = useState(false);

  // ── Smart auto-scroll ──
  // Check if user is near bottom (within 100px)
  const isNearBottom = useCallback(() => {
    const el = scrollContainerRef.current;
    if (!el) return true;
    return el.scrollHeight - el.scrollTop - el.clientHeight < 100;
  }, []);

  // Scroll to bottom smoothly
  const scrollToBottom = useCallback((instant = false) => {
    const el = scrollContainerRef.current;
    if (!el) return;
    el.scrollTo({
      top: el.scrollHeight,
      behavior: instant ? "instant" : "smooth",
    });
  }, []);

  // Track user scroll
  const handleScroll = useCallback(() => {
    if (isNearBottom()) {
      setUserScrolledUp(false);
    } else {
      setUserScrolledUp(true);
    }
  }, [isNearBottom]);

  // Auto-scroll when new messages arrive or during streaming
  useEffect(() => {
    if (!userScrolledUp) {
      // Use instant scroll during streaming to avoid jank
      if (isStreaming) {
        requestAnimationFrame(() => scrollToBottom(true));
      } else {
        scrollToBottom(false);
      }
    }
  }, [messages, isLoading, isStreaming, userScrolledUp, scrollToBottom]);

  // When user sends a message, always scroll to bottom
  const prevMessageCount = useRef(messages.length);
  useEffect(() => {
    if (messages.length > prevMessageCount.current) {
      const lastMsg = messages[messages.length - 1];
      if (lastMsg?.role === "user") {
        setUserScrolledUp(false);
        scrollToBottom(true);
      }
    }
    prevMessageCount.current = messages.length;
  }, [messages, scrollToBottom]);

  const showGreeting = messages.length === 0 && !isLoading;

  function handleWelcomeKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") {
      e.preventDefault();
      onSendMessage(input);
    }
  }

  return (
    <div
      ref={scrollContainerRef}
      onScroll={handleScroll}
      className={cn("flex-1 overflow-y-auto relative z-10 flex flex-col min-h-0", className)}
      style={style}
    >
      {showGreeting ? (
        /* ══════ WELCOME SCREEN ══════ */
        <div className="flex flex-1 items-center justify-center px-6 py-8 md:py-10">
          <div className="flex w-full max-w-[680px] flex-col items-center text-center">
            {/* ── Animated Icon ── */}
            <div className="relative mb-4 animate-float">
              <div className="flex h-[88px] w-[88px] items-center justify-center rounded-[24px] bg-gradient-to-br from-[var(--gradient-start)] via-[var(--gradient-mid)] to-[var(--gradient-end)] shadow-[0_0_40px_rgba(168,85,247,0.28)]">
                <Sparkles className="h-10 w-10 text-white" />
              </div>
              <div className="animate-pulse-glow absolute inset-0 rounded-[24px] bg-gradient-to-br from-[var(--gradient-start)] to-[var(--gradient-end)] opacity-25 blur-2xl" />
            </div>

            {/* ── Welcome Text ── */}
            <div className="mb-2 flex items-center gap-2 text-[14px] font-medium text-[#a18cae] animate-fade-slide-up">
              <Sparkles className="h-3.5 w-3.5 text-[#c084fc]" />
              <span>Welcome back</span>
            </div>

            {/* ── Hero Heading ── */}
            <h1
              className="mb-3 text-[32px] font-bold leading-[1.1] tracking-[-0.03em] text-white sm:text-[38px] lg:text-[42px] animate-fade-slide-up"
              style={{ animationDelay: "0.05s" }}
            >
              What can I help you{" "}
              <span className="text-transparent bg-gradient-to-r from-[#e8c8ff] to-[#b85cff] bg-clip-text">
                with?
              </span>
            </h1>

            {/* ── Subtitle ── */}
            <p
              className="mb-8 max-w-[480px] text-[15px] leading-[1.6] text-[#9e9ab0] animate-fade-slide-up"
              style={{ animationDelay: "0.1s", marginBottom: "25px" }}
            >
              I can help you code, brainstorm ideas, explain concepts,<br />
              write content, and much more.
            </p>

            {/* ── Search Input ── */}
            <div
              className="flex w-full max-w-[620px] items-center rounded-full border border-[#342945] bg-[#1a1528]/90 shadow-[inset_0_1px_0_rgba(255,255,255,0.02)] animate-fade-slide-up glow-on-focus"
              style={{ animationDelay: "0.16s", marginBottom: "25px", padding: "12px 20px" }}
            >
              {!input && <Search className="mr-3 h-[18px] w-[18px] shrink-0 text-[#b3adc5]" />}
              <input
                value={input}
                onChange={(e: ChangeEvent<HTMLInputElement>) => onInputChange(e.target.value)}
                onKeyDown={handleWelcomeKeyDown}
                disabled={isLoading}
                placeholder="Ask me anything..."
                className="min-w-0 flex-1 bg-transparent text-[15px] text-[#f3eefe] outline-none placeholder:text-[#8e889c] disabled:opacity-60"
              />
            </div>

            {/* ── Suggestion Grid ── */}
            <div
              className="mt-1 grid w-full max-w-[620px] grid-cols-1 gap-3.5 sm:grid-cols-2 animate-fade-slide-up"
              style={{ animationDelay: "0.2s" }}
            >
              {SUGGESTIONS.map((s, i) => {
                const Icon = s.icon;
                return (
                  <button
                    key={i}
                    onClick={() => onSendMessage(s.text)}
                    className={cn(
                      "group flex w-full items-center gap-3 rounded-xl border",
                      "bg-[#171427]/80 border-[#2d2640]",
                      "backdrop-blur-sm",
                      "text-left text-[14px] text-[#c6bfd5]",
                      "hover:text-white",
                      "hover:border-purple-400/35 hover:bg-[#1e1a30]",
                      "hover:shadow-[0_8px_24px_rgba(0,0,0,0.2)]",
                      "active:scale-[0.98]",
                      "transition-all duration-200 ease-out"
                    )}
                    style={{ padding: "14px 20px" }}
                  >
                    <Icon className={cn("h-[18px] w-[18px] shrink-0", s.color)} />
                    <span className="truncate">{s.text}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      ) : (
        /* ══════ MESSAGE FEED ══════ */
        <div className="flex flex-1 flex-col px-6 sm:px-8 md:px-12 lg:px-16 py-8">
          <div className="w-full max-w-3xl mx-auto mt-auto space-y-8 pb-4">
            {messages
              .filter((msg, i) => {
                // Hide empty assistant placeholder while typing indicator is showing
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
            <AnimatePresence>
              {isLoading && !isStreaming && <TypingIndicator />}
            </AnimatePresence>
            <div ref={messagesEndRef} />
          </div>
        </div>
      )}

      {/* Scroll to bottom button — shows when user scrolled up */}
      {userScrolledUp && messages.length > 0 && (
        <button
          onClick={() => {
            setUserScrolledUp(false);
            scrollToBottom(false);
          }}
          className={cn(
            "fixed bottom-28 right-8 z-50",
            "w-10 h-10 rounded-full",
            "bg-[#1e1a30] border border-[#3d2e5a]",
            "flex items-center justify-center",
            "text-purple-400 hover:text-white hover:bg-[#2a2240]",
            "shadow-lg shadow-black/30",
            "transition-all duration-200"
          )}
          aria-label="Scroll to bottom"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M8 3v10M4 9l4 4 4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
      )}
    </div>
  );
}

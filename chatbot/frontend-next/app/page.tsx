"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import Sidebar from "./components/Sidebar";
import Navbar from "./components/Navbar";
import ChatArea from "./components/ChatArea";
import InputBar from "./components/InputBar";
import { type Message } from "./components/MessageBubble";

interface ChatHistory {
  id: string;
  title: string;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function generateId() {
  return Math.random().toString(36).substring(2, 10);
}

// Safely extract string content from API responses
function extractContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (content === null || content === undefined) return "";
  if (Array.isArray(content)) {
    return content
      .map((item) => {
        if (typeof item === "string") return item;
        if (item && typeof item === "object" && "text" in item) return String(item.text);
        return "";
      })
      .join("");
  }
  if (typeof content === "object" && content !== null) {
    if ("text" in content) return String((content as { text: unknown }).text);
    if ("content" in content) return extractContent((content as { content: unknown }).content);
  }
  return "";
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [threadId, setThreadId] = useState(() => generateId());
  const [chatHistory, setChatHistory] = useState<ChatHistory[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // AbortController ref for cancelling streams
  const abortControllerRef = useRef<AbortController | null>(null);
  // Ref to track accumulated content without causing re-renders
  const accumulatedRef = useRef("");
  // RAF scheduling ref
  const rafRef = useRef<number | null>(null);

  // Fetch chat history on mount
  useEffect(() => {
    async function fetchHistory() {
      try {
        const res = await fetch(`${API_URL}/api/chats`);
        const data = await res.json();
        if (data.chats) {
          setChatHistory(data.chats);
        }
      } catch (err) {
        console.error("Failed to fetch chat history:", err);
      }
    }
    fetchHistory();
  }, []);

  // ── Stop generation ──
  const stopGenerating = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    setIsLoading(false);
    setIsStreaming(false);
  }, []);

  // ── Send message with smooth SSE streaming ──
  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || isLoading) return;

      // Cancel any existing stream
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      const controller = new AbortController();
      abortControllerRef.current = controller;

      const userMsg: Message = {
        id: generateId(),
        role: "user",
        content: text.trim(),
        timestamp: new Date(),
      };

      const assistantId = generateId();

      setMessages((prev) => [...prev, userMsg]);
      setInput("");
      setIsLoading(true);
      setIsStreaming(false);
      accumulatedRef.current = "";

      try {
        const res = await fetch(`${API_URL}/api/chat/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text.trim(), thread_id: threadId }),
          signal: controller.signal,
        });

        if (!res.ok) throw new Error(`Streaming request failed with status ${res.status}`);
        if (!res.body) throw new Error("No response body");

        // Add empty assistant message placeholder
        setMessages((prev) => [
          ...prev,
          { id: assistantId, role: "assistant", content: "", timestamp: new Date() },
        ]);

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let pendingUpdate = false;

        // Batch DOM updates with requestAnimationFrame
        function scheduleUpdate() {
          if (pendingUpdate) return;
          pendingUpdate = true;
          rafRef.current = requestAnimationFrame(() => {
            pendingUpdate = false;
            const content = accumulatedRef.current;
            setMessages((prev) => {
              const updated = [...prev];
              const lastIdx = updated.length - 1;
              if (updated[lastIdx]?.id === assistantId) {
                updated[lastIdx] = {
                  ...updated[lastIdx],
                  content,
                };
              }
              return updated;
            });
          });
        }

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data:")) continue;
            const dataStr = line.slice(5).trim();
            if (!dataStr) continue;

            try {
              let parsed = JSON.parse(dataStr);
              if (typeof parsed === "string") {
                try { parsed = JSON.parse(parsed); } catch { /* use as-is */ }
              }

              if (parsed.done) {
                // Stream complete
                continue;
              }

              if (parsed.token !== undefined) {
                const token = typeof parsed.token === "string"
                  ? parsed.token
                  : extractContent(parsed.token);
                accumulatedRef.current += token;
                if (!isStreaming) setIsStreaming(true);
                scheduleUpdate();
              } else if (parsed.error) {
                accumulatedRef.current += "\n\n" + parsed.error;
                scheduleUpdate();
              }
            } catch {
              // Raw string token
              accumulatedRef.current += dataStr;
              scheduleUpdate();
            }
          }
        }

        // Handle remaining buffer
        if (buffer.startsWith("data:")) {
          const remaining = buffer.slice(5).trim();
          if (remaining) {
            try {
              const parsed = JSON.parse(remaining);
              if (parsed.token) {
                accumulatedRef.current += typeof parsed.token === "string"
                  ? parsed.token
                  : extractContent(parsed.token);
              }
            } catch {
              accumulatedRef.current += remaining;
            }
          }
        }

        if (rafRef.current) {
          cancelAnimationFrame(rafRef.current);
          rafRef.current = null;
        }

        // Final flush
        setMessages((prev) => {
          const updated = [...prev];
          const lastIdx = updated.length - 1;
          if (updated[lastIdx]?.id === assistantId) {
            updated[lastIdx] = {
              ...updated[lastIdx],
              content: accumulatedRef.current,
            };
          }
          return updated;
        });
      } catch (err) {
        if ((err as Error).name === "AbortError") {
          // Stream was cancelled by user — keep what we have
          return;
        }
        const errorMsg: Message = {
          id: generateId(),
          role: "assistant",
          content: `⚠️ Could not reach the server. Make sure the API is running on ${API_URL}`,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, errorMsg]);
      } finally {
        abortControllerRef.current = null;
        setIsLoading(false);
        setIsStreaming(false);
      }
    },
    [isLoading, isStreaming, threadId]
  );

  // ── New chat ──
  function handleNewChat() {
    stopGenerating();
    if (messages.length > 0) {
      const firstMsg = messages[0].content;
      const title = typeof firstMsg === "string" && firstMsg.length > 36
        ? firstMsg.substring(0, 36) + "…"
        : String(firstMsg || "New Chat");

      setChatHistory((prev) => {
        if (prev.some((chat) => chat.id === threadId)) return prev;
        return [{ id: threadId, title }, ...prev];
      });
    }
    setMessages([]);
    setThreadId(generateId());
  }

  // ── Load existing chat ──
  async function loadChat(id: string) {
    if (id === threadId) return;
    stopGenerating();

    if (messages.length > 0) {
      const firstMsg = messages[0].content;
      const title = typeof firstMsg === "string" && firstMsg.length > 36
        ? firstMsg.substring(0, 36) + "…"
        : String(firstMsg || "New Chat");
      setChatHistory((prev) => {
        if (prev.some((chat) => chat.id === threadId)) return prev;
        return [{ id: threadId, title }, ...prev];
      });
    }

    setIsLoading(true);
    try {
      const res = await fetch(`${API_URL}/api/chat/${id}`);
      const data = await res.json();

      if (data.history) {
        const loadedMessages: Message[] = data.history.map(
          (msg: { role: "user" | "assistant"; content: string }) => ({
            id: generateId(),
            role: msg.role,
            content: extractContent(msg.content),
            timestamp: new Date(),
          })
        );
        setMessages(loadedMessages);
        setThreadId(id);
      }
    } catch (err) {
      console.error("Error loading chat:", err);
    } finally {
      setIsLoading(false);
    }
  }

  // ── Delete chat ──
  async function handleDeleteChat(e: React.MouseEvent, id: string) {
    e.stopPropagation();

    try {
      const res = await fetch(`${API_URL}/api/chat/${id}`, { method: "DELETE" });
      const data = await res.json();

      if (data.status === "ok") {
        setChatHistory((prev) => prev.filter((chat) => chat.id !== id));
        if (id === threadId) {
          setMessages([]);
          setThreadId(generateId());
        }
      }
    } catch (err) {
      console.error("Error deleting chat:", err);
    }
  }

  const showWelcomeFooter = messages.length === 0 && !isLoading;

  return (
    <div className="relative flex h-screen w-screen overflow-hidden bg-[var(--background)]">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(168,85,247,0.12),transparent_30%),radial-gradient(circle_at_center,rgba(124,58,237,0.06),transparent_45%)]" />
      {/* Sidebar drawer — always rendered, slides in/out */}
      <Sidebar
        chatHistory={chatHistory}
        threadId={threadId}
        onNewChat={handleNewChat}
        onLoadChat={loadChat}
        onDeleteChat={handleDeleteChat}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onToggle={() => setSidebarOpen((prev) => !prev)}
      />

      {/* Main area */}
      <main
        className="flex-1  flex flex-col min-w-0 overflow-hidden relative z-10"
        style={{
          background: "var(--background)",
          // marginLeft: "100px",
        }}
      >
        <Navbar
          isStreaming={isStreaming}
          style={{
            marginLeft: "10px",

          }}
        />

        <ChatArea
          messages={messages}
          input={input}
          isLoading={isLoading}
          isStreaming={isStreaming}
          onInputChange={setInput}
          onSendMessage={sendMessage}
          style={{
            marginLeft: "150px",
          }}
        />

        {showWelcomeFooter && (
          <div className="w-full border-t border-[#2a2436]/85 px-6 py-2.5 text-center text-[11px] text-[#756d86]">
            Gemini can make mistakes. Verify important information.
          </div>
        )}

        {(messages.length > 0 || isLoading) && (
          <InputBar
            input={input}
            isLoading={isLoading}
            isStreaming={isStreaming}
            onInputChange={setInput}
            onSend={sendMessage}
            onStopGenerating={stopGenerating}
            style={{
              marginLeft: "150px",
            }}
          />
        )}
      </main>
    </div>
  );
}

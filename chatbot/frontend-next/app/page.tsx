"use client";

import { useCallback, useEffect, useRef, useState, type ChangeEvent } from "react";
import ChatArea from "./components/ChatArea";
import Composer from "./components/Composer";
import { type Message } from "./components/MessageBubble";
import Navbar from "./components/Navbar";
import Sidebar, { type ChatSummary, type RagState } from "./components/Sidebar";
import SuggestionGrid from "./components/SuggestionGrid";
import ConfirmDialog from "./components/ui/ConfirmDialog";
import { useToast } from "./components/ui/Toast";
import { useIsDesktop } from "./lib/useMediaQuery";

interface RagStatusResponse {
  status?: string;
  message?: string;
  file_name?: string | null;
  chunks?: number;
  pages?: number;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const SIDEBAR_PREF_KEY = "zeno-chat:sidebar-open";

const IDLE_RAG: RagState = {
  status: "idle",
  fileName: null,
  pages: 0,
  chunks: 0,
  message: "No PDF indexed yet.",
};

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

function titleFor(messages: Message[]): string {
  const first = messages[0]?.content;
  if (typeof first !== "string" || !first) return "New Chat";
  return first.length > 36 ? `${first.substring(0, 36)}…` : first;
}

export default function Home() {
  const toast = useToast();
  const isDesktop = useIsDesktop();

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [threadId, setThreadId] = useState(() => generateId());

  const [chatHistory, setChatHistory] = useState<ChatSummary[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState(false);

  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<ChatSummary | null>(null);
  const [failedMessage, setFailedMessage] = useState<string | null>(null);
  const [rag, setRag] = useState<RagState>(IDLE_RAG);

  const pdfInputRef = useRef<HTMLInputElement | null>(null);
  const composerRef = useRef<HTMLTextAreaElement | null>(null);

  // AbortController ref for cancelling streams
  const abortControllerRef = useRef<AbortController | null>(null);
  // Ref to track accumulated content without causing re-renders
  const accumulatedRef = useRef("");
  // RAF scheduling ref
  const rafRef = useRef<number | null>(null);
  /** Set when the SSE stream reports an error mid-flight. */
  const streamErrorRef = useRef<string | null>(null);

  /* ── Chat history ─────────────────────────────────────────────── */

  const fetchHistory = useCallback(async () => {
    setHistoryLoading(true);
    setHistoryError(false);
    try {
      const res = await fetch(`${API_URL}/api/chats`);
      const data = await res.json();
      if (data.chats) {
        setChatHistory(data.chats);
      } else {
        throw new Error(data.error || "Malformed response");
      }
    } catch (err) {
      console.error("Failed to fetch chat history:", err);
      setHistoryError(true);
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  const fetchRagStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/rag/status`);
      if (!res.ok) return;
      const data: RagStatusResponse = await res.json();
      if (data.status === "ready") {
        setRag({
          status: "ready",
          fileName: data.file_name ?? "Document",
          pages: data.pages ?? 0,
          chunks: data.chunks ?? 0,
          message: "",
        });
      } else {
        setRag({ ...IDLE_RAG, message: data.message || IDLE_RAG.message });
      }
    } catch (err) {
      console.error("Failed to fetch RAG status:", err);
    }
  }, []);

  useEffect(() => {
    fetchHistory();
    fetchRagStatus();
  }, [fetchHistory, fetchRagStatus]);

  /* ── Sidebar preference ───────────────────────────────────────── */

  useEffect(() => {
    // The stored preference is a desktop rail-vs-panel choice. On small screens
    // the sidebar is a modal drawer, so it always starts closed — restoring
    // "open" there would cover the conversation on first paint.
    if (!isDesktop) {
      setSidebarOpen(false);
      return;
    }
    const stored = window.localStorage.getItem(SIDEBAR_PREF_KEY);
    setSidebarOpen(stored !== null ? stored === "true" : true);
    // Runs once the breakpoint is known; later toggles persist below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isDesktop]);

  const toggleSidebar = useCallback(() => {
    setSidebarOpen((prev) => {
      if (isDesktop) window.localStorage.setItem(SIDEBAR_PREF_KEY, String(!prev));
      return !prev;
    });
  }, [isDesktop]);

  const closeSidebar = useCallback(() => {
    setSidebarOpen(false);
    if (isDesktop) window.localStorage.setItem(SIDEBAR_PREF_KEY, "false");
  }, [isDesktop]);

  /* ── PDF / RAG ────────────────────────────────────────────────── */

  const uploadPdf = useCallback(
    async (file: File) => {
      if (!file) return;
      const isPdf =
        file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
      if (!isPdf) {
        toast("error", "Only PDF files can be indexed.");
        return;
      }

      setRag((prev) => ({
        ...prev,
        status: "uploading",
        message: `Uploading ${file.name}…`,
      }));

      try {
        const formData = new FormData();
        formData.append("file", file);

        const res = await fetch(`${API_URL}/api/rag/upload-pdf`, {
          method: "POST",
          body: formData,
        });
        const data = await res.json();

        if (!res.ok) {
          const detail = typeof data?.detail === "string" ? data.detail : "Upload failed.";
          setRag({ ...IDLE_RAG, status: "error", message: detail });
          toast("error", detail);
          return;
        }

        const fileName = data.file_name || file.name;
        const pages = Number(data.pages || 0);
        const chunks = Number(data.chunks || 0);
        setRag({ status: "ready", fileName, pages, chunks, message: "" });
        toast("success", `${fileName} indexed — ${pages} pages, ${chunks} chunks`);
      } catch (err) {
        console.error("PDF upload failed:", err);
        setRag({
          ...IDLE_RAG,
          status: "error",
          message: "Could not upload PDF. Check API connectivity.",
        });
        toast("error", "Could not upload PDF. Check API connectivity.");
      }
    },
    [toast]
  );

  const handlePdfInputChange = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      event.target.value = "";
      if (!file) return;
      await uploadPdf(file);
    },
    [uploadPdf]
  );

  const openPdfPicker = useCallback(() => {
    pdfInputRef.current?.click();
  }, []);

  /* ── Stop generation ──────────────────────────────────────────── */

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

  const handleStopGenerating = useCallback(() => {
    stopGenerating();
    toast("info", "Generation stopped");
  }, [stopGenerating, toast]);

  /* ── Send message with smooth SSE streaming ───────────────────── */

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || isLoading) return;
      if (rag.status === "uploading") {
        toast("info", "Wait for the PDF to finish indexing, then send your question.");
        return;
      }

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

      setFailedMessage(null);
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

        streamErrorRef.current = null;

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
                // Keep the text in the transcript for context, but flag it so
                // it isn't mistaken for part of the answer.
                accumulatedRef.current += "\n\n" + parsed.error;
                streamErrorRef.current = String(parsed.error);
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
        // Drop the empty placeholder and surface a retryable error card instead
        // of a fake assistant turn that reads like model output.
        setMessages((prev) =>
          prev.filter((m) => !(m.id === assistantId && !m.content))
        );
        setFailedMessage(text.trim());
        toast("error", "Message failed to send. Check your connection.");
      } finally {
        abortControllerRef.current = null;
        setIsLoading(false);
        setIsStreaming(false);
        if (streamErrorRef.current) {
          toast("error", "The assistant hit an error while replying.");
          streamErrorRef.current = null;
        }
      }
    },
    [isLoading, isStreaming, rag.status, threadId, toast]
  );

  const retryFailed = useCallback(() => {
    if (!failedMessage) return;
    const text = failedMessage;
    setFailedMessage(null);
    // Drop the user turn that failed; sendMessage re-adds it.
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      return last?.role === "user" && last.content === text ? prev.slice(0, -1) : prev;
    });
    sendMessage(text);
  }, [failedMessage, sendMessage]);

  /* ── Thread management ────────────────────────────────────────── */

  const rememberCurrentThread = useCallback(() => {
    setChatHistory((prev) => {
      if (prev.some((chat) => chat.id === threadId)) return prev;
      return [{ id: threadId, title: titleFor(messages) }, ...prev];
    });
  }, [messages, threadId]);

  const handleNewChat = useCallback(() => {
    stopGenerating();
    if (messages.length > 0) rememberCurrentThread();
    setMessages([]);
    setFailedMessage(null);
    setThreadId(generateId());
    if (!isDesktop) closeSidebar();
    requestAnimationFrame(() => composerRef.current?.focus());
  }, [closeSidebar, isDesktop, messages.length, rememberCurrentThread, stopGenerating]);

  const loadChat = useCallback(
    async (id: string) => {
      if (id === threadId) {
        if (!isDesktop) closeSidebar();
        return;
      }
      stopGenerating();
      if (messages.length > 0) rememberCurrentThread();

      setFailedMessage(null);
      setIsLoadingHistory(true);
      setMessages([]);
      setThreadId(id);
      if (!isDesktop) closeSidebar();

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
        }
      } catch (err) {
        console.error("Error loading chat:", err);
        toast("error", "Couldn't load that conversation.");
      } finally {
        setIsLoadingHistory(false);
      }
    },
    [closeSidebar, isDesktop, messages.length, rememberCurrentThread, stopGenerating, threadId, toast]
  );

  const confirmDelete = useCallback(async () => {
    const chat = pendingDelete;
    if (!chat) return;
    setPendingDelete(null);

    // Optimistic removal, with enough state captured to roll back on failure.
    const snapshot = { history: chatHistory, messages, threadId };
    const wasActive = chat.id === threadId;

    setChatHistory((prev) => prev.filter((c) => c.id !== chat.id));
    if (wasActive) {
      stopGenerating();
      setMessages([]);
      setThreadId(generateId());
    }

    try {
      const res = await fetch(`${API_URL}/api/chat/${chat.id}`, { method: "DELETE" });
      const data = await res.json();
      if (data.status !== "ok") throw new Error(data.error || "Delete failed");
      toast("success", "Chat deleted");
    } catch (err) {
      console.error("Error deleting chat:", err);
      setChatHistory(snapshot.history);
      if (wasActive) {
        setMessages(snapshot.messages);
        setThreadId(snapshot.threadId);
      }
      toast("error", "Couldn't delete chat. Try again.");
    }
  }, [chatHistory, messages, pendingDelete, stopGenerating, threadId, toast]);

  /* ── Keyboard shortcuts ───────────────────────────────────────── */

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      const mod = e.metaKey || e.ctrlKey;
      if (!mod) return;
      if (e.key === "k") {
        e.preventDefault();
        composerRef.current?.focus();
      } else if (e.key === "b") {
        e.preventDefault();
        toggleSidebar();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [toggleSidebar]);

  const isWelcome = messages.length === 0 && !isLoading && !isLoadingHistory;

  return (
    <div className="flex h-dvh w-full overflow-hidden bg-canvas">
      <input
        ref={pdfInputRef}
        type="file"
        accept=".pdf,application/pdf"
        className="hidden"
        onChange={handlePdfInputChange}
      />

      <Sidebar
        chatHistory={chatHistory}
        threadId={threadId}
        historyLoading={historyLoading}
        historyError={historyError}
        onRetryHistory={fetchHistory}
        onNewChat={handleNewChat}
        onUploadPdf={openPdfPicker}
        onDropPdf={uploadPdf}
        onLoadChat={loadChat}
        onRequestDelete={setPendingDelete}
        rag={rag}
        isOpen={sidebarOpen}
        isDesktop={isDesktop}
        onClose={closeSidebar}
        onToggle={toggleSidebar}
      />

      <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <Navbar
          isGenerating={isLoading}
          onToggleSidebar={toggleSidebar}
        />

        <ChatArea
          messages={messages}
          isLoading={isLoading}
          isStreaming={isStreaming}
          isLoadingHistory={isLoadingHistory}
          failedMessage={failedMessage}
          onRetry={retryFailed}
        />

        <Composer
          input={input}
          isLoading={isLoading}
          isUploadingPdf={rag.status === "uploading"}
          variant={isWelcome ? "welcome" : "docked"}
          onInputChange={setInput}
          onSend={sendMessage}
          onStopGenerating={handleStopGenerating}
          onAttach={openPdfPicker}
          onDropPdf={uploadPdf}
          textareaRef={composerRef}
        />

        {isWelcome && (
          <SuggestionGrid
            onSelect={sendMessage}
            disabled={isLoading || rag.status === "uploading"}
          />
        )}
      </main>

      <ConfirmDialog
        open={pendingDelete !== null}
        title="Delete chat?"
        description={
          pendingDelete
            ? `“${pendingDelete.title}” and its full history will be permanently deleted. This cannot be undone.`
            : ""
        }
        confirmLabel="Delete"
        cancelLabel="Cancel"
        destructive
        onConfirm={confirmDelete}
        onCancel={() => setPendingDelete(null)}
      />
    </div>
  );
}

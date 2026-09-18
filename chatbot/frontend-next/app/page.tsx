"use client";

import { useCallback, useEffect, useRef, useState, type ChangeEvent } from "react";
import ChatArea from "./components/ChatArea";
import Composer, { type ModelOption } from "./components/Composer";
import LocationStatus from "./components/LocationStatus";
import { type Message } from "./components/MessageBubble";
import Navbar from "./components/Navbar";
import Sidebar, { type ChatSummary, type RagState } from "./components/Sidebar";
import SuggestionGrid from "./components/SuggestionGrid";
import Button from "./components/ui/Button";
import ConfirmDialog from "./components/ui/ConfirmDialog";
import Logo from "./components/ui/Logo";
import { useToast } from "./components/ui/Toast";
import { useUserLocation } from "./hooks/useUserLocation";
import { needsLocation } from "./lib/location";
import { closeDanglingMarkup } from "./lib/streamingMarkdown";
import { useIsDesktop } from "./lib/useMediaQuery";

interface RagStatusResponse {
  status?: string;
  message?: string;
  file_name?: string | null;
  chunks?: number;
  pages?: number;
}

interface AuthUser {
  id: string;
  email: string;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const SIDEBAR_PREF_KEY = "zeno-chat:sidebar-open";
const MODEL_PREF_KEY = "zeno-chat:model";

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

/**
 * One turn of the conversation, in one of three flavours.
 *
 * `target` on a branch is always the *user* message being re-answered: it is
 * both what the server forks around and where the on-screen list is cut.
 */
type TurnRequest =
  | { kind: "send"; text: string }
  | { kind: "edit"; target: Message; text: string }
  | { kind: "retry"; target: Message };

/** A request the server refused before streaming; its reason is user-facing. */
class StreamStartError extends Error {}

function endpointFor(req: TurnRequest, threadId: string): string {
  return req.kind === "send"
    ? `${API_URL}/api/chat/stream`
    : `${API_URL}/api/chat/${encodeURIComponent(threadId)}/fork`;
}

function bodyFor(
  req: TurnRequest,
  threadId: string,
  text: string,
  model: string | null
): unknown {
  if (req.kind === "send") return { message: text, thread_id: threadId, model };
  if (req.kind === "edit")
    return { message_id: req.target.serverId, mode: "edit", message: text, model };
  return { message_id: req.target.serverId, mode: "retry", model };
}

async function readErrorDetail(res: Response): Promise<string | null> {
  try {
    const data = await res.json();
    if (typeof data?.detail === "string") return data.detail;
  } catch {
    // A non-JSON body (a proxy's error page) tells the user nothing useful.
  }
  return null;
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
  const [pendingDeleteAll, setPendingDeleteAll] = useState(false);
  const [isDeletingAll, setIsDeletingAll] = useState(false);
  const [failedMessage, setFailedMessage] = useState<string | null>(null);
  /** Client id of the user message open in the inline editor, if any. */
  const [editingId, setEditingId] = useState<string | null>(null);
  const [models, setModels] = useState<ModelOption[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [rag, setRag] = useState<RagState>(IDLE_RAG);
  const [currentUser, setCurrentUser] = useState<AuthUser | null>(null);
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [authEmail, setAuthEmail] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authLoading, setAuthLoading] = useState(false);
  const [isCheckingAuth, setIsCheckingAuth] = useState(true);

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
  const interruptedRef = useRef(false);
  // The message list as of this render. Edit and Retry need to read it and to
  // snapshot it for rollback; depending on `messages` instead would rebuild
  // the turn runner (and every memoised bubble's callbacks) on every token.
  const messagesRef = useRef<Message[]>(messages);
  messagesRef.current = messages;
  const selectedModelRef = useRef<string | null>(selectedModel);
  selectedModelRef.current = selectedModel;
  const modelsRef = useRef<ModelOption[]>(models);
  modelsRef.current = models;

  const resetChatState = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    setMessages([]);
    setInput("");
    setIsLoading(false);
    setIsStreaming(false);
    setChatHistory([]);
    setHistoryError(false);
    setHistoryLoading(false);
    setFailedMessage(null);
    setEditingId(null);
    setPendingDelete(null);
    setRag(IDLE_RAG);
    setThreadId(generateId());
  }, []);

  /* ── User location (weather / "near me" requests) ─────────────── */

  const handleLocationUnauthorized = useCallback(() => {
    setCurrentUser(null);
    resetChatState();
  }, [resetChatState]);

  const {
    status: locationStatus,
    location: userLocation,
    message: locationMessage,
    ensureLocation,
    submitManualCity,
    clearLocation,
    dismissed: locationDismissed,
    dismiss: dismissLocation,
  } = useUserLocation({
    enabled: Boolean(currentUser),
    onUnauthorized: handleLocationUnauthorized,
    onToast: toast,
  });

  const checkAuth = useCallback(async () => {
    setIsCheckingAuth(true);
    try {
      const res = await fetch(`${API_URL}/api/auth/me`, {
        credentials: "include",
      });
      if (!res.ok) {
        setCurrentUser(null);
        resetChatState();
        return;
      }
      const data = await res.json();
      setCurrentUser(data.user ?? null);
    } catch (err) {
      console.error("Failed to check auth:", err);
      setCurrentUser(null);
      resetChatState();
    } finally {
      setIsCheckingAuth(false);
    }
  }, [resetChatState]);

  const submitAuth = useCallback(async () => {
    if (!authEmail.trim() || !authPassword) return;
    setAuthLoading(true);
    try {
      const res = await fetch(`${API_URL}/api/auth/${authMode}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ email: authEmail.trim(), password: authPassword }),
      });
      const data = await res.json();
      if (!res.ok) {
        const detail = typeof data?.detail === "string" ? data.detail : "Authentication failed.";
        toast("error", detail);
        return;
      }
      setCurrentUser(data.user ?? null);
      setAuthPassword("");
      setThreadId(generateId());
      toast("success", authMode === "login" ? "Signed in" : "Account created");
    } catch (err) {
      console.error("Authentication failed:", err);
      toast("error", "Authentication failed. Check API connectivity.");
    } finally {
      setAuthLoading(false);
    }
  }, [authEmail, authMode, authPassword, toast]);

  const logout = useCallback(async () => {
    try {
      await fetch(`${API_URL}/api/auth/logout`, {
        method: "POST",
        credentials: "include",
      });
    } catch (err) {
      console.error("Logout failed:", err);
    } finally {
      setCurrentUser(null);
      resetChatState();
      toast("info", "Signed out");
    }
  }, [resetChatState, toast]);

  /* ── Chat history ─────────────────────────────────────────────── */

  const fetchHistory = useCallback(async () => {
    if (!currentUser) return;
    setHistoryLoading(true);
    setHistoryError(false);
    try {
      const res = await fetch(`${API_URL}/api/chats`, {
        credentials: "include",
      });
      if (res.status === 401) {
        setCurrentUser(null);
        resetChatState();
        return;
      }
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
  }, [currentUser, resetChatState]);

  const fetchModels = useCallback(async () => {
    if (!currentUser) return;
    try {
      const res = await fetch(`${API_URL}/api/models`, { credentials: "include" });
      if (!res.ok) return;
      const data = await res.json();
      const list: ModelOption[] = Array.isArray(data.models) ? data.models : [];
      setModels(list);

      // A remembered id can outlive the model itself — an API key removed from
      // the deployment drops it from the chain — so it is only honoured if the
      // server still offers it. Otherwise fall back to the chain head.
      let stored: string | null = null;
      try {
        stored = window.localStorage.getItem(MODEL_PREF_KEY);
      } catch {
        // Private mode / blocked storage: the default is a fine answer.
      }
      setSelectedModel(
        stored && list.some((m) => m.id === stored) ? stored : data.default ?? null
      );
    } catch (err) {
      console.error("Failed to fetch models:", err);
    }
  }, [currentUser]);

  const selectModel = useCallback((id: string) => {
    setSelectedModel(id);
    try {
      window.localStorage.setItem(MODEL_PREF_KEY, id);
    } catch {
      // The choice still applies to this session; it just will not persist.
    }
  }, []);

  const fetchRagStatus = useCallback(async () => {
    if (!currentUser) return;
    try {
      const res = await fetch(
        `${API_URL}/api/rag/status?thread_id=${encodeURIComponent(threadId)}`,
        { credentials: "include" }
      );
      if (res.status === 401) {
        setCurrentUser(null);
        resetChatState();
        return;
      }
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
  }, [currentUser, resetChatState, threadId]);

  useEffect(() => {
    checkAuth();
  }, [checkAuth]);

  useEffect(() => {
    if (!currentUser) return;
    fetchHistory();
    fetchRagStatus();
    fetchModels();
  }, [currentUser, fetchHistory, fetchModels, fetchRagStatus]);

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
        formData.append("thread_id", threadId);

        const res = await fetch(`${API_URL}/api/rag/upload-pdf`, {
          method: "POST",
          body: formData,
          credentials: "include",
        });
        const data = await res.json();

        if (res.status === 401) {
          setCurrentUser(null);
          resetChatState();
          toast("error", "Please sign in again.");
          return;
        }

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
    [resetChatState, threadId, toast]
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

  /* ── Running a turn: send, edit, retry ────────────────────────── */

  /**
   * Send a new message, or re-run an existing turn on a new branch.
   *
   * All three go through one request/stream path because the server treats
   * them the same way: an edit and a retry are a normal turn executed against
   * the state that preceded an earlier one. Only the endpoint, the body and
   * what happens to the messages already on screen differ.
   */
  const runTurn = useCallback(
    async (req: TurnRequest) => {
      if (isLoading) return;
      if (rag.status === "uploading") {
        toast("info", "Wait for the PDF to finish indexing, then send your question.");
        return;
      }

      const isBranch = req.kind !== "send";
      // The list as it stands, so a failed branch can be put back exactly.
      const snapshot = messagesRef.current;

      let promptText: string;
      let priorMessages: Message[];

      if (req.kind === "send") {
        promptText = req.text.trim();
        if (!promptText) return;
        priorMessages = snapshot;
      } else {
        if (!req.target.serverId) {
          toast("error", "That message can't be edited yet. Try again in a moment.");
          return;
        }
        const targetIndex = snapshot.findIndex((m) => m.id === req.target.id);
        if (targetIndex < 0) {
          toast("error", "That message is no longer in this conversation.");
          return;
        }
        promptText = (req.kind === "edit" ? req.text : req.target.content).trim();
        if (!promptText) return;
        // Everything from the edited turn onwards belongs to the branch being
        // replaced, so it leaves the screen the moment the new one starts.
        priorMessages = snapshot.slice(0, targetIndex);
      }

      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      const controller = new AbortController();
      abortControllerRef.current = controller;

      const userMsg: Message = {
        id: generateId(),
        role: "user",
        content: promptText,
        timestamp: new Date(),
      };
      const assistantId = generateId();

      setEditingId(null);
      setFailedMessage(null);
      setMessages([...priorMessages, userMsg]);
      if (req.kind === "send") setInput("");
      setIsLoading(true);
      setIsStreaming(false);
      accumulatedRef.current = "";
      // Set only once the turn is committed, so its absence at the end is a
      // reliable "nothing was written".
      let committedIds: { user?: string | null; assistant?: string | null } | null = null;
      let answeredBy: string | null = null;

      try {
        // Own-location questions ("what's the weather?", "anything near me?")
        // need coordinates the backend cannot infer, so resolve them before the
        // request goes out. The user's turn is already on screen and the
        // LocationStatus strip above the composer explains the pause.
        //
        // A null result is deliberately NOT fatal: we still send the message and
        // let the backend/LLM handle the missing location gracefully — it asks
        // the user for a city. Aborting here would just swallow their message.
        if (needsLocation(promptText)) {
          await ensureLocation();
        }

        const res = await fetch(endpointFor(req, threadId), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: JSON.stringify(
            bodyFor(req, threadId, promptText, selectedModelRef.current)
          ),
          signal: controller.signal,
        });

        if (res.status === 401) {
          setCurrentUser(null);
          resetChatState();
          toast("error", "Please sign in again.");
          return;
        }
        if (!res.ok) {
          // A branch is rejected before any of it runs — a stale message id, a
          // turn already in flight, someone else's thread — so the server's
          // reason is worth showing verbatim.
          const detail = isBranch ? await readErrorDetail(res) : null;
          throw new StreamStartError(
            detail || `Streaming request failed with status ${res.status}`
          );
        }
        if (!res.body) throw new Error("No response body");

        streamErrorRef.current = null;
        interruptedRef.current = false;

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

              if (parsed.message_ids) {
                // The turn is now in the checkpointer. These are the handles
                // Edit and Retry will need for it.
                committedIds = parsed.message_ids;
                answeredBy =
                  typeof parsed.answered_by === "string" ? parsed.answered_by : null;
                continue;
              }

              if (parsed.reset) {
                // The model emitted text on its way to deciding to call a
                // tool. That text is not the answer, so drop what has been
                // shown so far and let the real answer stream in clean.
                accumulatedRef.current = "";
                scheduleUpdate();
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
                streamErrorRef.current = String(parsed.error);
                if (parsed.interrupted) {
                  // A provider died part-way through. What already arrived is
                  // genuine model output, so it stays as-is and the user is
                  // offered a retry instead of a fabricated tail.
                  interruptedRef.current = true;
                } else {
                  // Keep the text in the transcript for context, but flag it so
                  // it isn't mistaken for part of the answer.
                  accumulatedRef.current += "\n\n" + parsed.error;
                }
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

        if (isBranch && !committedIds) {
          // The server abandons a branch it could not finish, leaving the old
          // one active. Showing the half-built branch would be showing a
          // conversation that no longer exists anywhere.
          setMessages(snapshot);
          toast("error", "Couldn't regenerate that response. Nothing was changed.");
          return;
        }

        // Final flush. A reply cut off at the token ceiling, or one ended by an
        // interruption, can stop mid-`**bold**`; the backend closes the copy it
        // stores, and this closes the copy on screen so the two agree.
        const finalContent = closeDanglingMarkup(accumulatedRef.current);
        setMessages((prev) => {
          const updated = [...prev];
          const lastIdx = updated.length - 1;
          if (updated[lastIdx]?.id === assistantId) {
            updated[lastIdx] = {
              ...updated[lastIdx],
              content: finalContent,
            };
          }
          if (!committedIds) return updated;
          // The chain falls through when the picked model is rate-limited, so
          // name the model that really answered rather than let the picker
          // imply one that never ran.
          const asked = selectedModelRef.current;
          const substituted =
            answeredBy && asked && answeredBy !== asked
              ? modelsRef.current.find((m) => m.id === answeredBy)?.label ?? answeredBy
              : null;
          // Both ids change on a branch: the re-run writes a new user message
          // as well as a new answer.
          return updated.map((m) => {
            if (m.id === userMsg.id) return { ...m, serverId: committedIds?.user ?? null };
            if (m.id === assistantId)
              return {
                ...m,
                serverId: committedIds?.assistant ?? null,
                answeredBy: substituted,
              };
            return m;
          });
        });

        if (interruptedRef.current && !isBranch) {
          // Half an answer is on screen and it is not going to finish, so give
          // the user the one-click retry rather than leaving them to retype.
          setFailedMessage(promptText);
        }
      } catch (err) {
        if ((err as Error).name === "AbortError") {
          // Stream was cancelled by user — keep what we have.
          if (isBranch && !committedIds) {
            // Except on a branch, where "what we have" is a conversation the
            // server never committed to. Stopping means the old one stands.
            setMessages(snapshot);
          }
          return;
        }
        if (isBranch) {
          // Same reasoning as the no-commit case above: the stored
          // conversation never moved, so neither does the one on screen.
          setMessages(snapshot);
          toast(
            "error",
            err instanceof StreamStartError
              ? err.message
              : "Couldn't regenerate that response. Nothing was changed."
          );
          return;
        }
        // Drop the empty placeholder and surface a retryable error card instead
        // of a fake assistant turn that reads like model output.
        setMessages((prev) =>
          prev.filter((m) => !(m.id === assistantId && !m.content))
        );
        setFailedMessage(promptText);
        toast("error", "Message failed to send. Check your connection.");
      } finally {
        abortControllerRef.current = null;
        setIsLoading(false);
        setIsStreaming(false);
        if (streamErrorRef.current) {
          if (!isBranch) {
            toast(
              "error",
              interruptedRef.current
                ? "The reply was cut short. Tap retry to ask again."
                : "The assistant hit an error while replying."
            );
          }
          streamErrorRef.current = null;
          interruptedRef.current = false;
        }
      }
    },
    [ensureLocation, isLoading, isStreaming, rag.status, resetChatState, threadId, toast]
  );

  const sendMessage = useCallback(
    (text: string) => {
      void runTurn({ kind: "send", text });
    },
    [runTurn]
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

  /* ── Edit / regenerate ────────────────────────────────────────── */

  const startEdit = useCallback(
    (message: Message) => {
      if (isLoading) return;
      setEditingId(message.id);
    },
    [isLoading]
  );

  const cancelEdit = useCallback(() => setEditingId(null), []);

  const submitEdit = useCallback(
    (message: Message, text: string) => {
      if (text.trim() === message.content.trim()) {
        // Nothing changed, so there is no new branch to create.
        setEditingId(null);
        return;
      }
      void runTurn({ kind: "edit", target: message, text });
    },
    [runTurn]
  );

  const regenerate = useCallback(
    (message: Message) => {
      // Retry is addressed by the request being re-answered, which is also the
      // point the on-screen list is truncated to.
      const list = messagesRef.current;
      const index = list.findIndex((m) => m.id === message.id);
      if (index < 0) return;
      const target =
        message.role === "user"
          ? message
          : [...list.slice(0, index)].reverse().find((m) => m.role === "user");
      if (!target) {
        toast("error", "There's no message to regenerate from.");
        return;
      }
      void runTurn({ kind: "retry", target });
    },
    [runTurn, toast]
  );

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
    setEditingId(null);
    setRag(IDLE_RAG);
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
      setEditingId(null);
      setIsLoadingHistory(true);
      setMessages([]);
      setRag(IDLE_RAG);
      setThreadId(id);
      if (!isDesktop) closeSidebar();

      try {
        const res = await fetch(`${API_URL}/api/chat/${id}`, {
          credentials: "include",
        });
        if (res.status === 401) {
          setCurrentUser(null);
          resetChatState();
          toast("error", "Please sign in again.");
          return;
        }
        if (!res.ok) throw new Error(`Failed to load chat: ${res.status}`);
        const data = await res.json();

        if (data.history) {
          const loadedMessages: Message[] = data.history.map(
            (msg: { id?: string; role: "user" | "assistant"; content: string }) => ({
              id: generateId(),
              // What Edit and Retry address the turn by. A reloaded
              // conversation is the case where it always exists.
              serverId: msg.id ?? null,
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
    [closeSidebar, isDesktop, messages.length, rememberCurrentThread, resetChatState, stopGenerating, threadId, toast]
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
      const res = await fetch(`${API_URL}/api/chat/${chat.id}`, {
        method: "DELETE",
        credentials: "include",
      });
      if (res.status === 401) {
        setCurrentUser(null);
        resetChatState();
        toast("error", "Please sign in again.");
        return;
      }
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
  }, [chatHistory, messages, pendingDelete, resetChatState, stopGenerating, threadId, toast]);

  const confirmDeleteAll = useCallback(async () => {
    setPendingDeleteAll(false);
    setIsDeletingAll(true);

    // Enough state to put everything back if the request fails, so a network
    // blip cannot look like "all my chats vanished".
    const snapshot = { history: chatHistory, messages, threadId };

    try {
      const res = await fetch(`${API_URL}/api/chats`, {
        method: "DELETE",
        credentials: "include",
      });
      if (res.status === 401) {
        setCurrentUser(null);
        resetChatState();
        toast("error", "Please sign in again.");
        return;
      }
      const data = await res.json();
      if (!res.ok || data.status !== "ok") {
        throw new Error(data.detail || "Delete failed");
      }

      // Only clear locally once the server has confirmed. The open
      // conversation is one of the deleted ones, so it goes too.
      stopGenerating();
      setChatHistory([]);
      setMessages([]);
      setFailedMessage(null);
      setRag(IDLE_RAG);
      setThreadId(generateId());
      const count = Number(data.deleted ?? 0);
      toast("success", count === 1 ? "1 chat deleted" : `${count} chats deleted`);
    } catch (err) {
      console.error("Error deleting all chats:", err);
      setChatHistory(snapshot.history);
      setMessages(snapshot.messages);
      setThreadId(snapshot.threadId);
      toast("error", "Couldn't delete your chats. Try again.");
    } finally {
      setIsDeletingAll(false);
    }
  }, [chatHistory, messages, resetChatState, stopGenerating, threadId, toast]);

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

  if (isCheckingAuth) {
    return (
      <div className="flex h-dvh w-full items-center justify-center bg-canvas text-small text-fg-muted">
        Checking session...
      </div>
    );
  }

  if (!currentUser) {
    return (
      <main className="flex min-h-dvh w-full items-center justify-center bg-canvas px-4">
        <form
          className="w-full max-w-sm rounded-lg border border-line bg-raised p-5 shadow-e2"
          onSubmit={(event) => {
            event.preventDefault();
            submitAuth();
          }}
        >
          <div className="mb-5">
            <Logo size={40} className="mb-3 rounded-[10px] shadow-e2" />
            <h1 className="text-title-sm font-semibold text-fg">Zeno AI</h1>
            <p className="mt-1 text-small text-fg-muted">
              {authMode === "login" ? "Sign in to your chats." : "Create your account."}
            </p>
          </div>

          <label className="mb-3 block">
            <span className="mb-1 block text-small font-medium text-fg">Email</span>
            <input
              type="email"
              value={authEmail}
              onChange={(event) => setAuthEmail(event.target.value)}
              autoComplete="email"
              className="h-10 w-full rounded-md border border-line bg-canvas px-3 text-small text-fg outline-none focus:border-focus"
              required
            />
          </label>

          <label className="mb-4 block">
            <span className="mb-1 block text-small font-medium text-fg">Password</span>
            <input
              type="password"
              value={authPassword}
              onChange={(event) => setAuthPassword(event.target.value)}
              autoComplete={authMode === "login" ? "current-password" : "new-password"}
              minLength={8}
              className="h-10 w-full rounded-md border border-line bg-canvas px-3 text-small text-fg outline-none focus:border-focus"
              required
            />
          </label>

          <Button type="submit" variant="primary" block disabled={authLoading}>
            {authLoading
              ? "Please wait..."
              : authMode === "login"
                ? "Sign in"
                : "Create account"}
          </Button>

          <button
            type="button"
            className="mt-4 w-full text-center text-small text-accent-fg hover:underline"
            onClick={() => {
              setAuthMode((prev) => (prev === "login" ? "register" : "login"));
              setAuthPassword("");
            }}
          >
            {authMode === "login"
              ? "Need an account? Register"
              : "Already have an account? Sign in"}
          </button>
        </form>
      </main>
    );
  }

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
        onLoadChat={loadChat}
        onRequestDelete={setPendingDelete}
        onRequestDeleteAll={() => setPendingDeleteAll(true)}
        rag={rag}
        isOpen={sidebarOpen}
        isDesktop={isDesktop}
        onClose={closeSidebar}
        onToggle={toggleSidebar}
      />

      <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <Navbar
          userEmail={currentUser.email}
          onToggleSidebar={toggleSidebar}
          onLogout={logout}
        />

        <ChatArea
          messages={messages}
          isLoading={isLoading}
          isStreaming={isStreaming}
          isLoadingHistory={isLoadingHistory}
          failedMessage={failedMessage}
          onRetry={retryFailed}
          editingId={editingId}
          onStartEdit={startEdit}
          onCancelEdit={cancelEdit}
          onSubmitEdit={submitEdit}
          onRegenerate={regenerate}
        />

        {!locationDismissed && (
          <LocationStatus
            status={locationStatus}
            message={locationMessage}
            onSubmitCity={submitManualCity}
            onDismiss={dismissLocation}
          />
        )}

        <Composer
          input={input}
          isLoading={isLoading}
          isUploadingPdf={rag.status === "uploading"}
          models={models}
          selectedModel={selectedModel}
          onSelectModel={selectModel}
          variant={isWelcome ? "welcome" : "docked"}
          onInputChange={setInput}
          onSend={sendMessage}
          onStopGenerating={handleStopGenerating}
          onAttach={openPdfPicker}
          onDropPdf={uploadPdf}
          onRequestLocation={ensureLocation}
          onClearLocation={clearLocation}
          locationLabel={userLocation?.label ?? null}
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

      <ConfirmDialog
        open={pendingDeleteAll}
        title="Delete all chats?"
        description={
          `All ${chatHistory.length} conversation${chatHistory.length === 1 ? "" : "s"} ` +
          "and their full history will be permanently deleted. This cannot be " +
          "undone. What the assistant remembers about you is kept."
        }
        confirmLabel={isDeletingAll ? "Deleting…" : "Delete all"}
        cancelLabel="Cancel"
        destructive
        onConfirm={confirmDeleteAll}
        onCancel={() => setPendingDeleteAll(false)}
      />
    </div>
  );
}

"use client";

import { useState, useRef, useEffect, KeyboardEvent, ChangeEvent } from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { atomDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import styles from "./ChatInterface.module.css";

interface Message {
    role: "user" | "assistant";
    content: string;
}

interface ChatHistory {
    id: string;
    title: string;
}

const SUGGESTIONS = [
    "🐍 Explain Python decorators",
    "🚀 Give me a project idea",
    "✍️ Write something creative",
    "🧠 Teach me something new",
    "🔍 Debug my code",
    "📖 Summarise a concept",
];

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function generateId() {
    return Math.random().toString(36).substring(2, 10);
}

export default function ChatInterface() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [threadId, setThreadId] = useState(() => generateId());
    const [chatHistory, setChatHistory] = useState<ChatHistory[]>([]);
    const [theme, setTheme] = useState<"dark" | "light">("dark");

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    // Load saved theme and chat history on mount
    useEffect(() => {
        const savedTheme = localStorage.getItem("chatbot-theme") as "dark" | "light" | null;
        if (savedTheme) {
            setTheme(savedTheme);
            document.documentElement.setAttribute("data-theme", savedTheme);
        }

        const savedHistory = localStorage.getItem("chatbot-history");
        if (savedHistory) {
            try {
                setChatHistory(JSON.parse(savedHistory));
            } catch (e) {
                console.error("Failed to parse chat history:", e);
            }
        }
    }, []);

    // Apply theme changes
    useEffect(() => {
        document.documentElement.setAttribute("data-theme", theme);
        localStorage.setItem("chatbot-theme", theme);
    }, [theme]);

    // Save chat history to local storage when it changes
    useEffect(() => {
        localStorage.setItem("chatbot-history", JSON.stringify(chatHistory));
    }, [chatHistory]);

    function toggleTheme() {
        setTheme((prev) => (prev === "dark" ? "light" : "dark"));
    }

    // Auto-scroll to bottom when messages change
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, isLoading]);

    // Auto-resize textarea
    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = "auto";
            textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
        }
    }, [input]);

    async function sendMessage(text: string) {
        if (!text.trim() || isLoading) return;

        const userMsg: Message = { role: "user", content: text.trim() };
        setMessages((prev) => [...prev, userMsg]);
        setInput("");
        setIsLoading(true);

        try {
            const res = await fetch(`${API_URL}/api/chat/stream`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: text.trim(), thread_id: threadId }),
            });

            if (!res.body) throw new Error("No response body");

            // Add an empty assistant message that we'll fill progressively
            const assistantIdx = messages.length + 1; // +1 for the user msg we just added
            setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";
            let accumulated = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });

                // Process complete SSE lines
                const lines = buffer.split("\n");
                buffer = lines.pop() || ""; // keep incomplete line in buffer

                for (const line of lines) {
                    if (line.startsWith("data:")) {
                        const dataStr = line.slice(5).trim();
                        if (!dataStr) continue;

                        try {
                            let parsed = JSON.parse(dataStr);
                            // Handle double-encoded JSON (sse-starlette wraps in extra quotes)
                            if (typeof parsed === "string") {
                                parsed = JSON.parse(parsed);
                            }
                            if (parsed.token !== undefined) {
                                accumulated += parsed.token;
                            } else if (parsed.error) {
                                accumulated += "\n\n" + parsed.error;
                            }
                        } catch (e) {
                            // fallback for non-JSON stream chunks
                            accumulated += dataStr;
                        }

                        setMessages((prev) => {
                            const updated = [...prev];
                            updated[assistantIdx] = {
                                role: "assistant",
                                content: accumulated,
                            };
                            return updated;
                        });
                    }
                    // "event: done" or "event: error" are handled by the stream ending
                }
            }

            // Handle any remaining buffer
            if (buffer.startsWith("data:")) {
                accumulated += buffer.slice(5);
                setMessages((prev) => {
                    const updated = [...prev];
                    updated[assistantIdx] = {
                        role: "assistant",
                        content: accumulated,
                    };
                    return updated;
                });
            }
        } catch (err) {
            const errorMsg: Message = {
                role: "assistant",
                content: `⚠️ Error: Could not reach the server. Make sure the API is running on ${API_URL}`,
            };
            setMessages((prev) => [...prev, errorMsg]);
        } finally {
            setIsLoading(false);
        }
    }

    function handleNewChat() {
        if (messages.length > 0) {
            const firstMsg = messages[0].content;
            const title =
                firstMsg.length > 36 ? firstMsg.substring(0, 36) + "…" : firstMsg;

            // Only add if not already in history
            setChatHistory((prev) => {
                if (prev.some(chat => chat.id === threadId)) {
                    return prev;
                }
                return [{ id: threadId, title }, ...prev];
            });
        }
        setMessages([]);
        setThreadId(generateId());
    }

    async function loadChat(id: string) {
        if (id === threadId) return;

        // Save current chat if needed before switching
        if (messages.length > 0) {
            const firstMsg = messages[0].content;
            const title = firstMsg.length > 36 ? firstMsg.substring(0, 36) + "…" : firstMsg;
            setChatHistory((prev) => {
                if (prev.some(chat => chat.id === threadId)) {
                    return prev;
                }
                return [{ id: threadId, title }, ...prev];
            });
        }

        setIsLoading(true);
        try {
            const res = await fetch(`${API_URL}/api/chat/${id}`);
            const data = await res.json();

            if (data.history) {
                setMessages(data.history);
                setThreadId(id);
            } else {
                console.error("Failed to fetch history:", data.error);
            }
        } catch (err) {
            console.error("Error loading chat:", err);
        } finally {
            setIsLoading(false);
        }
    }

    function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage(input);
        }
    }

    function handleChipClick(suggestion: string) {
        // Remove emoji prefix
        const text = suggestion.replace(/^[^\w]*\s/, "");
        sendMessage(text);
    }

    const showGreeting = messages.length === 0 && !isLoading;

    return (
        <div className={styles.container}>
            {/* ── SIDEBAR ── */}
            <aside className={styles.sidebar}>
                <div className={styles.sidebarHeader}>
                    <div className={styles.sidebarLogo}>
                        <svg viewBox="0 0 28 28" fill="none">
                            <path
                                d="M14 2 L16 12 L26 14 L16 16 L14 26 L12 16 L2 14 L12 12 Z"
                                fill="url(#sideGrad)"
                            />
                            <defs>
                                <linearGradient id="sideGrad" x1="0" y1="0" x2="1" y2="1">
                                    <stop offset="0%" stopColor="#a78bfa" />
                                    <stop offset="50%" stopColor="#667eea" />
                                    <stop offset="100%" stopColor="#f472b6" />
                                </linearGradient>
                            </defs>
                        </svg>
                        <span>Gemini Chat</span>
                    </div>
                    <button className={styles.themeToggle} onClick={toggleTheme} title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}>
                        {theme === "dark" ? (
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <circle cx="12" cy="12" r="5" />
                                <line x1="12" y1="1" x2="12" y2="3" />
                                <line x1="12" y1="21" x2="12" y2="23" />
                                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
                                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                                <line x1="1" y1="12" x2="3" y2="12" />
                                <line x1="21" y1="12" x2="23" y2="12" />
                                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
                                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
                            </svg>
                        ) : (
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
                            </svg>
                        )}
                    </button>
                </div>

                <button className={styles.newChatBtn} onClick={handleNewChat}>
                    <svg
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                    >
                        <line x1="12" y1="5" x2="12" y2="19" />
                        <line x1="5" y1="12" x2="19" y2="12" />
                    </svg>
                    New chat
                </button>

                {chatHistory.length > 0 && (
                    <>
                        <div className={styles.chatHistoryLabel}>Recent</div>
                        <div className={styles.chatHistoryList}>
                            {chatHistory.slice(0, 20).map((chat) => (
                                <div
                                    key={chat.id}
                                    className={`${styles.chatHistoryItem} ${chat.id === threadId ? styles.activeChat : ''}`}
                                    onClick={() => loadChat(chat.id)}
                                >
                                    💬 {chat.title}
                                </div>
                            ))}
                        </div>
                    </>
                )}

                <div className={styles.sidebarFooter}>
                    <p>
                        Gemini Chatbot
                        <br />
                        LangGraph · Gemini 2.5 Flash
                    </p>
                </div>
            </aside>

            {/* ── MAIN ── */}
            <main className={styles.main}>
                <div className={styles.messagesArea}>
                    {showGreeting && (
                        <>
                            {/* Greeting */}
                            <div className={styles.greeting}>
                                <div className={styles.greetingSub}>
                                    <svg width="22" height="22" viewBox="0 0 28 28" fill="none">
                                        <path
                                            d="M14 2 L16 12 L26 14 L16 16 L14 26 L12 16 L2 14 L12 12 Z"
                                            fill="url(#mainGrad)"
                                        />
                                        <defs>
                                            <linearGradient
                                                id="mainGrad"
                                                x1="0"
                                                y1="0"
                                                x2="1"
                                                y2="1"
                                            >
                                                <stop offset="0%" stopColor="#a78bfa" />
                                                <stop offset="50%" stopColor="#667eea" />
                                                <stop offset="100%" stopColor="#f472b6" />
                                            </linearGradient>
                                        </defs>
                                    </svg>
                                    <span>Hi there</span>
                                </div>
                                <h1 className={styles.greetingMain}>
                                    What can I help you with?
                                </h1>
                            </div>

                            {/* Suggestion chips */}
                            <div className={styles.chips}>
                                {SUGGESTIONS.map((s) => (
                                    <button
                                        key={s}
                                        className={styles.chip}
                                        onClick={() => handleChipClick(s)}
                                    >
                                        {s}
                                    </button>
                                ))}
                            </div>
                        </>
                    )}

                    {/* Messages */}
                    {messages.length > 0 && (
                        <div className={styles.messageList}>
                            {messages.map((msg, i) => (
                                <div key={i} className={styles.message}>
                                    <div
                                        className={`${styles.avatar} ${msg.role === "user"
                                            ? styles.userAvatar
                                            : styles.assistantAvatar
                                            }`}
                                    >
                                        {msg.role === "user" ? "🧑‍💻" : "✨"}
                                    </div>
                                    <div className={styles.messageContent}>
                                        <ReactMarkdown
                                            components={{
                                                code({ className, children, ...props }) {
                                                    const match = /language-(\w+)/.exec(className || "");
                                                    const codeString = String(children).replace(/\n$/, "");
                                                    if (match) {
                                                        return (
                                                            <div className={styles.codeBlock}>
                                                                <div className={styles.codeHeader}>
                                                                    <div className={styles.codeLanguage}>
                                                                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                                                            <polyline points="16 18 22 12 16 6"></polyline>
                                                                            <polyline points="8 6 2 12 8 18"></polyline>
                                                                        </svg>
                                                                        <span>{match[1].charAt(0).toUpperCase() + match[1].slice(1)}</span>
                                                                    </div>
                                                                    <button
                                                                        className={styles.copyBtn}
                                                                        onClick={() => navigator.clipboard.writeText(codeString)}
                                                                        title="Copy code"
                                                                    >
                                                                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                                                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                                                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                                                                        </svg>
                                                                    </button>
                                                                </div>
                                                                <SyntaxHighlighter
                                                                    style={atomDark}
                                                                    language={match[1]}
                                                                    PreTag="div"
                                                                    customStyle={{
                                                                        margin: 0,
                                                                        borderRadius: "0 0 12px 12px",
                                                                        fontSize: "0.85rem",
                                                                    }}
                                                                >
                                                                    {codeString}
                                                                </SyntaxHighlighter>
                                                            </div>
                                                        );
                                                    }
                                                    return (
                                                        <code className={styles.inlineCode} {...props}>
                                                            {children}
                                                        </code>
                                                    );
                                                },
                                            }}
                                        >
                                            {msg.content}
                                        </ReactMarkdown>
                                    </div>
                                </div>
                            ))}

                            {/* Typing indicator */}
                            {isLoading && (
                                <div className={styles.typingIndicator}>
                                    <div className={styles.typingDots}>
                                        <span />
                                        <span />
                                        <span />
                                    </div>
                                </div>
                            )}

                            <div ref={messagesEndRef} />
                        </div>
                    )}
                </div>

                {/* ── INPUT BAR ── */}
                <div className={styles.inputArea}>
                    <div className={styles.inputCard}>
                        <textarea
                            ref={textareaRef}
                            value={input}
                            onChange={(e: ChangeEvent<HTMLTextAreaElement>) =>
                                setInput(e.target.value)
                            }
                            onKeyDown={handleKeyDown}
                            placeholder="Ask me anything…"
                            rows={1}
                            disabled={isLoading}
                        />
                        <button
                            className={`${styles.sendBtn} ${input.trim() ? styles.sendBtnActive : ""
                                }`}
                            onClick={() => sendMessage(input)}
                            disabled={!input.trim() || isLoading}
                        >
                            <svg
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                            >
                                <line x1="12" y1="19" x2="12" y2="5" />
                                <polyline points="5 12 12 5 19 12" />
                            </svg>
                        </button>
                    </div>
                </div>
            </main>
        </div>
    );
}

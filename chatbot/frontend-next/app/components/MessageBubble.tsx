"use client";

import { cn } from "@/app/lib/utils";
import { motion } from "framer-motion";
import { Bot, Copy, Check, Code } from "lucide-react";
import { useState, memo } from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus, oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

interface MessageBubbleProps {
  message: Message;
  index: number;
  isStreaming?: boolean;
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <button
      onClick={handleCopy}
      className={cn(
        "p-1.5 rounded-md",
        "text-[var(--text-muted)] hover:text-[var(--text-primary)]",
        "hover:bg-white/10 transition-all duration-200"
      )}
      title="Copy code"
    >
      {copied ? (
        <Check className="w-3.5 h-3.5 text-emerald-400" />
      ) : (
        <Copy className="w-3.5 h-3.5" />
      )}
    </button>
  );
}

function MessageBubbleInner({ message, index, isStreaming = false }: MessageBubbleProps) {
  const isUser = message.role === "user";
  const contentStr = String(message.content || "");

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: 0.3,
        delay: index * 0.04,
        ease: [0.4, 0, 0.2, 1],
      }}
      className={cn(
        "flex w-full",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      <div
        className={cn(
          "flex flex-col",
          isUser ? "items-end" : "items-start",
          "w-full max-w-2xl"
        )}
      >
        {/* Avatar + Bubble row */}
        <div
          className={cn(
            "flex items-start gap-2.5",
            isUser ? "flex-row-reverse" : "flex-row"
          )}
        >
          {/* Avatar — assistant only */}
          {!isUser && (
            <div className="shrink-0 mt-0.5">
              <div
                className={cn(
                  "w-8 h-8 rounded-full flex items-center justify-center",
                  "bg-gradient-to-br from-[var(--gradient-start)] to-[var(--gradient-end)]",
                  "shadow-lg shadow-purple-500/20"
                )}
              >
                <Bot className="w-4 h-4 text-white" />
              </div>
            </div>
          )}

          {/* Bubble */}
          <div
            className={cn(
              "relative rounded-2xl message-bubble",
              isUser
                ? [
                    "user-gradient text-white",
                    "rounded-br-md",
                    "shadow-lg shadow-purple-500/15",
                  ]
                : [
                    "glass-card",
                    "rounded-bl-md",
                    "text-[var(--text-primary)]",
                  ],
              isStreaming && "is-streaming"
            )}
            style={{ padding: "14px 20px" }}
          >
            <div
              className={cn(
                "text-[14px] leading-relaxed",
                "prose prose-invert prose-sm max-w-none",
                "[&_p]:mb-2 [&_p:last-child]:mb-0",
                "[&_ul]:ml-4 [&_ol]:ml-4",
                "[&_li]:mb-1",
                "[&_strong]:font-semibold",
                "[&_a]:text-purple-400 [&_a]:underline [&_a:hover]:text-purple-300",
                isUser && "[&_*]:text-white",
                // Blinking cursor at end while streaming
                isStreaming && !isUser && "streaming-cursor"
              )}
            >
              {contentStr ? (
                <ReactMarkdown
                  components={{
                    code({ className, children, ...props }) {
                      const match = /language-(\w+)/.exec(className || "");
                      const codeString = String(children).replace(/\n$/, "");

                      if (match) {
                        const lang = match[1];
                        const langName = lang === "javascript" ? "JavaScript" : lang === "typescript" ? "TypeScript" : lang.charAt(0).toUpperCase() + lang.slice(1);

                        return (
                          <div className="my-4 rounded-[12px] overflow-hidden bg-[#1e1e1e] border border-white/5 shadow-md">
                            <div className="flex items-center justify-between px-4 py-2 bg-[#2d2d2d]/80 border-b border-white/5">
                              <div className="flex items-center gap-2 text-gray-300">
                                <Code className="w-4 h-4" />
                                <span className="text-[13px] font-medium font-sans">
                                  {langName}
                                </span>
                              </div>
                              <CopyButton text={codeString} />
                            </div>
                            <SyntaxHighlighter
                              style={vscDarkPlus || oneDark}
                              language={match[1]}
                              PreTag="div"
                              customStyle={{
                                margin: 0,
                                borderRadius: 0,
                                fontSize: "14px",
                                padding: "16px",
                                background: "#1e1e1e",
                              }}
                            >
                              {codeString}
                            </SyntaxHighlighter>
                          </div>
                        );
                      }
                      return (
                        <code
                          className="px-1.5 py-0.5 rounded-md bg-white/10 text-purple-300 text-[13px] font-mono"
                          {...props}
                        >
                          {children}
                        </code>
                      );
                    },
                  }}
                >
                  {contentStr}
                </ReactMarkdown>
              ) : (
                <span className="text-[var(--text-muted)]">...</span>
              )}
            </div>
          </div>
        </div>

        {/* Timestamp — below the bubble, hidden while streaming */}
        {!isStreaming && (
          <p
            className={cn(
              "text-[10px] mt-1.5",
              isUser ? "text-[#706a7e] mr-1" : "text-[#706a7e] ml-11"
            )}
          >
            {message.timestamp.toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </p>
        )}
      </div>
    </motion.div>
  );
}

// Memoize to prevent re-rendering non-streaming messages on each token
const MessageBubble = memo(MessageBubbleInner, (prev, next) => {
  // Always re-render if streaming state changes
  if (prev.isStreaming !== next.isStreaming) return false;
  // Always re-render the streaming message
  if (next.isStreaming) return false;
  // Don't re-render completed messages
  return prev.message.content === next.message.content && prev.message.id === next.message.id;
});

MessageBubble.displayName = "MessageBubble";

export default MessageBubble;

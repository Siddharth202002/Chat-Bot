"use client";

import { cn } from "@/app/lib/utils";
import { useEffect, useCallback } from "react";
import {
  Sparkles,
  Plus,
  MessageSquare,
  Trash2,
  User,
  PanelLeftClose,
  PanelLeft,
  Search,
  FileText,
} from "lucide-react";

interface ChatHistory {
  id: string;
  title: string;
}

interface SidebarProps {
  chatHistory: ChatHistory[];
  threadId: string;
  onNewChat: () => void;
  onUploadPdf: () => void;
  onLoadChat: (id: string) => void;
  onDeleteChat: (e: React.MouseEvent, id: string) => void;
  ragStatusText: string;
  isUploadingPdf: boolean;
  isOpen: boolean;
  onClose: () => void;
  onToggle: () => void;
}

export default function Sidebar({
  chatHistory,
  threadId,
  onNewChat,
  onUploadPdf,
  onLoadChat,
  onDeleteChat,
  ragStatusText,
  isUploadingPdf,
  isOpen,
  onClose,
  onToggle,
}: SidebarProps) {
  const handleEscape = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) onClose();
    },
    [isOpen, onClose]
  );

  useEffect(() => {
    document.addEventListener("keydown", handleEscape);
    return () => document.removeEventListener("keydown", handleEscape);
  }, [handleEscape]);

  return (
    <div className="h-full flex shrink-0 relative z-20 mr-3">
      {!isOpen && (
        <div
          className={cn(
            "h-full flex flex-col items-center py-4 gap-5",
            "bg-[linear-gradient(180deg,#161129_0%,#141026_26%,#131022_100%)]",
            "border-r border-[#2a2435]"
          )}
          style={{ width: "52px" }}
        >
          <button
            onClick={onToggle}
            className={cn(
              "w-9 h-9 rounded-lg flex items-center justify-center",
              "text-[#9f94b4] hover:text-white",
              "hover:bg-white/[0.08]",
              "transition-all duration-200"
            )}
            title="Open sidebar"
          >
            <PanelLeft className="w-5 h-5" />
          </button>

          <button
            onClick={onNewChat}
            className={cn(
              "w-9 h-9 rounded-lg flex items-center justify-center",
              "text-[#9f94b4] hover:text-white",
              "hover:bg-white/[0.08]",
              "transition-all duration-200"
            )}
            title="New chat"
          >
            <Plus className="w-5 h-5" />
          </button>

          <button
            className={cn(
              "w-9 h-9 rounded-lg flex items-center justify-center",
              "text-[#9f94b4] hover:text-white",
              "hover:bg-white/[0.08]",
              "transition-all duration-200"
            )}
            title="Search"
          >
            <Search className="w-5 h-5" />
          </button>

          <button
            onClick={onUploadPdf}
            className={cn(
              "w-9 h-9 rounded-lg flex items-center justify-center",
              "text-[#9f94b4] hover:text-white",
              "hover:bg-white/[0.08]",
              "transition-all duration-200",
              
              isUploadingPdf && "opacity-60 pointer-events-none"
            )}
            title={isUploadingPdf ? "Uploading PDF..." : "Add PDF"}
            disabled={isUploadingPdf}
          >
            <FileText className="w-5 h-5" />
          </button>

          <div className="mt-auto">
            <div
              className={cn(
                "w-8 h-8 rounded-full flex items-center justify-center",
                "border border-[#6b4ba0] bg-[#2b1746]"
              )}
            >
              <User className="h-3.5 w-3.5 text-[#c273ff]" />
            </div>
          </div>
        </div>
      )}

      <aside
        className={cn(
          "group/sidebar h-full flex flex-col overflow-hidden",
          "bg-[linear-gradient(180deg,#161129_0%,#141026_26%,#131022_100%)]",
          "border-r border-[#2a2435]",
          "transition-all duration-300 ease-[cubic-bezier(0.4,0,0.2,1)]"
        )}
        style={{ width: isOpen ? "260px" : "0px" }}
      >
        <div style={{ width: "260px", minWidth: "260px" }} className="h-full flex flex-col relative">
          <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(168,85,247,0.12),transparent_34%)]" />

          <div className="relative z-10 flex items-center gap-3 px-4 pt-5 pb-2">
            <div className="relative">
              <div className="flex h-9 w-9 items-center justify-center rounded-[12px] border border-[#7a52e7]/40 bg-gradient-to-br from-[#6f34e4] via-[#8d43f2] to-[#d245bb] shadow-[0_8px_26px_rgba(124,58,237,0.35)]">
                <Sparkles className="h-4 w-4 text-white" />
              </div>
            </div>
            <div className="flex-1">
              <h1 className="text-[15px] font-semibold tracking-tight text-[#e9d7ff]">
                Gemini AI
              </h1>
              <p className="text-[11px] text-[#9f94b4]">
                Powered by LangGraph
              </p>
            </div>
            <button
              onClick={onToggle}
              className={cn(
                "w-8 h-8 rounded-lg flex items-center justify-center",
                "text-[#716a7f] hover:text-[#c4b5d9]",
                "hover:bg-white/[0.06]",
                "transition-all duration-200",
                "opacity-0 group-hover/sidebar:opacity-100"
              )}
              title="Close sidebar"
            >
              <PanelLeftClose className="w-[18px] h-[18px]" />
            </button>
          </div>

          <div
            className="relative z-10"
            style={{ borderBottom: "1px solid #2a2435", margin: "4px 16px 16px 16px" }}
          />

          <div className="relative z-10" style={{ padding: "0 14px 16px 14px" }}>
            <div className="flex flex-col gap-3">
              <button
                onClick={() => {
                  onNewChat();
                }}
                className={cn(
                  "flex w-full items-center justify-center gap-2 rounded-[10px]",
                  "bg-gradient-to-r from-[#6528db] via-[#8a36ef] to-[#bb42f8]",
                  "text-[13px] font-semibold text-white",
                  "hover:brightness-110 hover:shadow-[0_8px_26px_rgba(124,58,237,0.38)]",
                  "active:scale-[0.98]",
                  "transition-all duration-200 ease-out"
                )}
                style={{ padding: "10px 14px" }}
              >
                <Plus className="w-4 h-4" />
                New Chat
              </button>
              <button
                onClick={onUploadPdf}
                disabled={isUploadingPdf}
                className={cn(
                  "flex w-full items-center justify-center gap-2 rounded-[10px]",
                  "bg-gradient-to-r from-[#6528db] via-[#8a36ef] to-[#bb42f8]",
                  "text-[13px] font-semibold text-white",
                  "hover:brightness-110 hover:shadow-[0_8px_26px_rgba(124,58,237,0.38)]",
                  "active:scale-[0.98]",
                  "transition-all duration-200 ease-out",
                  isUploadingPdf && "opacity-60 pointer-events-none"
                )}
                style={{ padding: "10px 14px" }}
              >
                <FileText className="w-4 h-4" />
                {isUploadingPdf ? "Uploading PDF..." : "Add PDF"}
              </button>
            </div>
            <p className="mt-2 px-1 text-[11px] leading-relaxed text-[#8d84a4]">
              {ragStatusText}
            </p>
          </div>

          <div className="relative z-10 flex-1 overflow-y-auto" style={{ padding: "0 10px" }}>
            {chatHistory.length > 0 && (
              <>
                <p
                  className="text-[10px] font-semibold uppercase tracking-[0.2em] text-[#716a7f]"
                  style={{ padding: "8px 10px 6px" }}
                >
                  Recent Chats
                </p>
                {chatHistory.slice(0, 20).map((chat) => (
                  <button
                    key={chat.id}
                    onClick={() => {
                      onLoadChat(chat.id);
                    }}
                    className={cn(
                      "group flex w-full items-center gap-2 rounded-lg",
                      "text-left transition-all duration-200",
                      chat.id === threadId
                        ? "border border-purple-500/20 bg-white/[0.08] text-white"
                        : "text-[#a19ab2] hover:bg-white/5 hover:text-gray-200"
                    )}
                    style={{ padding: "9px 12px", marginBottom: "2px" }}
                  >
                    <MessageSquare size={13} className="shrink-0 opacity-50" />
                    <span className="text-[12px] truncate flex-1">
                      {chat.title}
                    </span>
                    <Trash2
                      size={13}
                      className="shrink-0 opacity-0 group-hover:opacity-50 hover:!opacity-100 hover:text-red-400 transition-all duration-200"
                      onClick={(e) => onDeleteChat(e, chat.id)}
                    />
                  </button>
                ))}
              </>
            )}
          </div>

          <div className="relative z-10 mt-auto">
            <div className="border-t border-[#2b2338] bg-[#171327]/80" style={{ padding: "10px 12px" }}>
              <div className="flex items-center gap-2.5">
                <div className="flex h-7 w-7 items-center justify-center rounded-full border border-[#6b4ba0] bg-[#2b1746]">
                  <User className="h-3.5 w-3.5 text-[#c273ff]" />
                </div>
                <p className="truncate text-[13px] font-medium text-[#aea6be]">
                  Gemini 2.5 Flash
                </p>
              </div>
            </div>
          </div>
        </div>
      </aside>
    </div>
  );
}

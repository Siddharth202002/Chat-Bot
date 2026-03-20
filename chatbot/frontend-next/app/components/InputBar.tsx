"use client";

import { cn } from "@/app/lib/utils";
import { useRef, useEffect, ChangeEvent, KeyboardEvent, useState } from "react";
import { Send, Plus, Square, Paperclip, X } from "lucide-react";

interface InputBarProps {
  input: string;
  isLoading: boolean;
  isStreaming: boolean;
  onInputChange: (value: string) => void;
  onSend: (text: string) => void;
  onStopGenerating: () => void;
  style?: React.CSSProperties;
}

export default function InputBar({
  input,
  isLoading,
  isStreaming,
  onInputChange,
  onSend,
  onStopGenerating,
  style,
}: InputBarProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [stagedFiles, setStagedFiles] = useState<{name: string; data: string; type: string; size: number}[]>([]);

  function handleFileChange(e: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;

    files.forEach(file => {
      const reader = new FileReader();
      reader.onload = (event) => {
        const data = event.target?.result as string;
        setStagedFiles(prev => [...prev, { name: file.name, data, type: file.type, size: file.size }]);
      };
      reader.readAsDataURL(file);
    });

    e.target.value = '';
  }

  function handleSend() {
    if (!canSend) return;
    onSend(input);
    setStagedFiles([]);
  }

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        120
      )}px`;
    }
  }, [input]);

  const canSend = (input.trim().length > 0 || stagedFiles.length > 0) && !isLoading;

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="relative z-10 w-full px-6 sm:px-8 md:px-12 lg:px-16 pb-6 pt-2" style={style}>
      <div className="w-full max-w-3xl mx-auto">

        {/* ── Stop Generating Button ── */}
        {isStreaming && (
          <div className="flex justify-center" style={{ marginBottom: "10px" }}>
            <button
              onClick={onStopGenerating}
              className={cn(
                "flex items-center gap-2 rounded-full",
                "border border-[#3d2e5a] bg-[#1a1528]/95",
                "text-[13px] text-[#c4b5d9]",
                "hover:text-white hover:bg-[#231e36] hover:border-purple-500/40",
                "active:scale-[0.97]",
                "transition-all duration-200",
                "stop-generating-btn"
              )}
              style={{ padding: "7px 18px" }}
            >
              <Square className="w-3 h-3 fill-current" />
              Stop generating
            </button>
          </div>
        )}

        {/* ── Staged Files Preview ── */}
        {stagedFiles.length > 0 && (
          <div className="flex flex-wrap gap-3 mb-4 px-2">
            {stagedFiles.map((f, i) => (
              <div key={i} className="relative group flex items-center gap-3 bg-[#1a1528]/95 backdrop-blur-md border border-[#32283f] rounded-[22px] px-3 py-2 shadow-md transition-all hover:border-purple-500/30">
                {/* Icon or Image */}
                {f.type.startsWith('image/') ? (
                   <img src={f.data} alt={f.name} className="w-10 h-10 object-cover rounded-[14px] shadow-sm border border-white/5" />
                ) : (
                   <div className="flex items-center justify-center w-10 h-10 rounded-[14px] bg-[#231e36] border border-white/5 text-[#a18cae]">
                     <Paperclip className="w-4 h-4" />
                   </div>
                )}
                
                {/* File Details */}
                <div className="flex flex-col justify-center min-w-[120px] mr-2">
                  <span className="text-white text-[14px] font-medium max-w-[160px] truncate leading-tight mb-0.5">{f.name}</span>
                  <span className="text-[#878197] text-[12px] leading-tight">{(f.size / 1024).toFixed(1)} KB</span>
                </div>

                {/* remove button */}
                <button 
                  onClick={() => setStagedFiles(prev => prev.filter((_, idx) => idx !== i))}
                  className="absolute -top-1.5 -right-1.5 bg-[#2d2640] hover:bg-red-500 text-[#a18cae] hover:text-white border border-[#483d65] w-[22px] h-[22px] flex items-center justify-center rounded-full opacity-0 group-hover:opacity-100 transition-all shadow-md z-20"
                  title="Remove attachment"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* ── Pill Input Container ── */}
        <div
          className={cn(
            "flex w-full items-end gap-2",
            "border border-[#32283f] bg-[#1a1528]/90",
            "backdrop-blur-sm",
            "transition-all duration-300",
            "focus-within:border-purple-500/40",
            "focus-within:shadow-[0_0_0_1px_rgba(168,85,247,0.2),0_8px_24px_rgba(0,0,0,0.2)]",
            isLoading && "opacity-70"
          )}
          style={{ borderRadius: "9999px", padding: "8px 8px 8px 16px" }}
        >
          {/* Hidden file input */}
          <input
            type="file"
            ref={fileInputRef}
            style={{ display: "none" }}
            accept="image/*,.pdf,.txt,.doc,.docx"
            multiple
            onChange={handleFileChange}
          />

          {/* Plus button */}
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className={cn(
              "shrink-0 rounded-full flex items-center justify-center",
              "text-[#878197] hover:text-white",
              "hover:bg-white/8",
              "transition-all duration-200",
              isLoading && "pointer-events-none opacity-50"
            )}
            style={{ width: "32px", height: "32px", marginBottom: "2px" }}
            title="Attach"
            disabled={isLoading}
          >
            <Plus className="w-5 h-5" />
          </button>

          {/* Textarea */}
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e: ChangeEvent<HTMLTextAreaElement>) =>
              onInputChange(e.target.value)
            }
            onKeyDown={handleKeyDown}
            placeholder={isStreaming ? "Wait for response..." : "Ask anything..."}
            rows={1}
            disabled={isLoading}
            className={cn(
              "flex-1 resize-none bg-transparent",
              "text-[15px] text-white",
              "placeholder:text-[#6b6580]",
              "outline-none border-none",
              "disabled:opacity-50 disabled:cursor-not-allowed"
            )}
            style={{ maxHeight: "120px", paddingTop: "6px", paddingBottom: "6px" }}
          />

          {/* Send button */}
          <button
            onClick={handleSend}
            disabled={!canSend}
            className={cn(
              "shrink-0 rounded-full flex items-center justify-center",
              "transition-all duration-300 ease-out",
              canSend
                ? [
                    "bg-gradient-to-r from-purple-600 to-fuchsia-500",
                    "text-white shadow-lg shadow-purple-500/25",
                    "hover:shadow-purple-500/40 hover:scale-105",
                    "active:scale-95",
                  ]
                : [
                    "bg-white/5",
                    "text-[#4a4460]",
                    "cursor-not-allowed",
                  ]
            )}
            style={{ width: "36px", height: "36px" }}
          >
            <Send className="w-4 h-4" />
          </button>
        </div>

        {/* Disclaimer */}
        <p
          className="text-center text-[#5e586e]"
          style={{ fontSize: "11px", marginTop: "8px", paddingBottom: "2px" }}
        >
          Gemini can make mistakes. Verify important information.
        </p>
      </div>
    </div>
  );
}

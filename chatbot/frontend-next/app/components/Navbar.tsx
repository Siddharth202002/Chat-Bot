"use client";

import { cn } from "@/app/lib/utils";
import { Sparkles, Circle } from "lucide-react";

interface NavbarProps {
  isStreaming?: boolean;
  style?: React.CSSProperties;
}

export default function Navbar({ isStreaming = false, style }: NavbarProps) {
  return (
    <nav
      className={cn(
        "flex h-[56px] items-center justify-between px-4",
        "border-b border-[#2a2436]/85",
        "bg-[#120f1d]/62 backdrop-blur-xl",
        "relative z-10"
      )}
      style={style}
    >
      <div className="flex items-center gap-2.5">
        <Sparkles className="h-4 w-4 text-[#d35cff]" />
        <div className="flex items-center gap-2">
          <span className="text-[14px] font-semibold text-white">
            Gemini 2.5 Flash
          </span>
          <div
            className="flex items-center gap-1.5 rounded-full border border-emerald-500/24 bg-emerald-500/12"
            style={{ padding: "2px 8px" }}
          >
            <Circle className="h-2 w-2 fill-emerald-400 text-emerald-400" />
            <span className="text-[10px] font-medium text-emerald-400">
              Online
            </span>
          </div>
        </div>
      </div>

      {/* Streaming indicator */}
      {isStreaming && (
        <div
          className="flex items-center gap-2 rounded-full border border-purple-500/25 bg-purple-500/10"
          style={{ padding: "4px 12px" }}
        >
          <div className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-purple-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-purple-500" />
          </div>
          <span className="text-[11px] font-medium text-purple-300">
            Generating...
          </span>
        </div>
      )}
    </nav>
  );
}

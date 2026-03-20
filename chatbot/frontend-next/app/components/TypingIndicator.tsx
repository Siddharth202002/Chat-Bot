"use client";

import { cn } from "@/app/lib/utils";
import { motion } from "framer-motion";
import { Bot } from "lucide-react";

export default function TypingIndicator() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.3 }}
      className="flex gap-3 w-full max-w-3xl mx-auto"
    >
      {/* Avatar */}
      <div className="shrink-0 mt-1">
        <div
          className={cn(
            "w-8 h-8 rounded-xl flex items-center justify-center",
            "bg-gradient-to-br from-[var(--gradient-start)] to-[var(--gradient-end)]",
            "shadow-lg shadow-purple-500/20"
          )}
        >
          <Bot className="w-4 h-4 text-white" />
        </div>
      </div>

      {/* Typing dots */}
      <div className="glass-card rounded-2xl rounded-bl-md px-5 py-4">
        <div className="flex items-end gap-1.5 h-[18px]">
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              className={cn(
                "block w-2.5 h-2.5 rounded-full",
                "bg-gradient-to-br from-purple-400 to-pink-400",
                "animate-typing-bounce",
                i === 0 && "typing-dot-1",
                i === 1 && "typing-dot-2",
                i === 2 && "typing-dot-3"
              )}
            />
          ))}
        </div>
      </div>
    </motion.div>
  );
}

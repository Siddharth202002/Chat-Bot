"use client";

import { Menu, SquarePen } from "lucide-react";
import ModelPicker, { type ModelOption } from "../chat/ModelPicker";
import { IconButton } from "../ui/Button";

interface HeaderProps {
  onToggleSidebar: () => void;
  onNewChat: () => void;
  models: ModelOption[];
  selectedModel: string | null;
  onSelectModel: (id: string) => void;
}

/**
 * Top bar of the chat panel. Account actions (sign out, theme) live in the
 * sidebar's profile menu, which is reachable on every breakpoint.
 */
export default function Header({
  onToggleSidebar,
  onNewChat,
  models,
  selectedModel,
  onSelectModel,
}: HeaderProps) {
  return (
    <header className="relative z-20 flex h-15 shrink-0 items-center justify-between gap-2 px-3 sm:h-16 sm:px-5">
      <div className="flex min-w-0 items-center gap-2">
        {/* Mobile-only drawer trigger; on desktop the sidebar owns its toggle. */}
        <IconButton
          label="Open sidebar"
          onClick={onToggleSidebar}
          className="glass-panel shadow-e1 md:hidden"
        >
          <Menu className="h-4.5 w-4.5" strokeWidth={1.9} />
        </IconButton>

        <ModelPicker
          models={models}
          selectedModel={selectedModel}
          onSelectModel={onSelectModel}
        />
      </div>

      {/* Phones lose the always-visible sidebar, so New chat surfaces here. */}
      <div className="flex shrink-0 items-center md:hidden">
        <IconButton label="New chat" onClick={onNewChat} className="glass-panel shadow-e1">
          <SquarePen className="h-4 w-4" strokeWidth={1.9} />
        </IconButton>
      </div>
    </header>
  );
}

"use client";

import { LogOut, PanelLeft } from "lucide-react";
import { Badge } from "./ui/Primitives";
import { IconButton } from "./ui/Button";

interface NavbarProps {
  /** True for the whole in-flight window, not just once tokens arrive. */
  isGenerating: boolean;
  userEmail: string;
  onToggleSidebar: () => void;
  onLogout: () => void;
}

export default function Navbar({
  isGenerating,
  userEmail,
  onToggleSidebar,
  onLogout,
}: NavbarProps) {
  return (
    <header className="relative z-20 flex h-14 shrink-0 items-center justify-between gap-3 border-b border-line-subtle bg-canvas/80 px-3 backdrop-blur-md sm:px-4">
      <div className="flex min-w-0 items-center gap-2">
        {/* Mobile-only drawer trigger; on desktop the sidebar owns its toggle. */}
        <IconButton
          label="Open sidebar"
          size="sm"
          onClick={onToggleSidebar}
          className="md:hidden"
        >
          <PanelLeft className="h-4.5 w-4.5" strokeWidth={1.75} />
        </IconButton>

        <span className="truncate text-small font-semibold text-fg">Zeno AI</span>
      </div>

      <div className="flex min-w-0 items-center gap-2">
        {isGenerating && (
          <Badge tone="accent" dot pulse className="shrink-0">
            Generating
          </Badge>
        )}
        <span className="hidden max-w-[180px] truncate text-small text-fg-muted sm:inline">
          {userEmail}
        </span>
        <IconButton label="Sign out" size="sm" onClick={onLogout}>
          <LogOut className="h-4 w-4" strokeWidth={1.75} />
        </IconButton>
      </div>
    </header>
  );
}

"use client";

import { cn } from "@/app/lib/utils";
import { AlertTriangle, CheckCircle2, Info, X } from "lucide-react";
import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

export type ToastVariant = "success" | "error" | "info";

interface Toast {
  id: number;
  variant: ToastVariant;
  message: string;
}

interface ToastContextValue {
  toast: (variant: ToastVariant, message: string) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

const MAX_VISIBLE = 3;
const DURATION: Record<ToastVariant, number> = {
  success: 3000,
  info: 3000,
  error: 5000,
};

const STYLES: Record<ToastVariant, { bar: string; icon: string; Icon: typeof Info }> = {
  success: { bar: "bg-success", icon: "text-success", Icon: CheckCircle2 },
  error: { bar: "bg-danger", icon: "text-danger", Icon: AlertTriangle },
  info: { bar: "bg-accent-fg", icon: "text-accent-fg", Icon: Info },
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const nextId = useRef(0);
  const timers = useRef(new Map<number, ReturnType<typeof setTimeout>>());

  const dismiss = useCallback((id: number) => {
    const timer = timers.current.get(id);
    if (timer) {
      clearTimeout(timer);
      timers.current.delete(id);
    }
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const toast = useCallback(
    (variant: ToastVariant, message: string) => {
      const id = nextId.current++;
      setToasts((prev) => [...prev, { id, variant, message }].slice(-MAX_VISIBLE));
      timers.current.set(
        id,
        setTimeout(() => dismiss(id), DURATION[variant])
      );
    },
    [dismiss]
  );

  const value = useMemo(() => ({ toast }), [toast]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div
        className="pointer-events-none fixed bottom-4 right-4 z-100 flex w-[min(22rem,calc(100vw-2rem))] flex-col gap-2 sm:bottom-6 sm:right-6"
        role="region"
        aria-label="Notifications"
      >
        {toasts.map((t) => {
          const { bar, icon, Icon } = STYLES[t.variant];
          return (
            <div
              key={t.id}
              role="status"
              aria-live={t.variant === "error" ? "assertive" : "polite"}
              className={cn(
                "animate-toast-in pointer-events-auto relative flex items-start gap-2.5",
                "overflow-hidden rounded-md border border-line bg-overlay py-3 pl-4 pr-2",
                "shadow-e2"
              )}
            >
              <span className={cn("absolute inset-y-0 left-0 w-[3px]", bar)} aria-hidden />
              <Icon className={cn("mt-px h-4 w-4 shrink-0", icon)} strokeWidth={1.75} aria-hidden />
              <p className="flex-1 text-small text-fg">{t.message}</p>
              <button
                type="button"
                onClick={() => dismiss(t.id)}
                aria-label="Dismiss notification"
                className="-mt-0.5 shrink-0 rounded-sm p-1 text-fg-subtle transition-colors hover:bg-hover hover:text-fg"
              >
                <X className="h-3.5 w-3.5" strokeWidth={2} />
              </button>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used inside a ToastProvider");
  return ctx.toast;
}

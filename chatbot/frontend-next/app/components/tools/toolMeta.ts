import type { ToolActivity } from "@/app/lib/tools";
import {
  Calculator,
  CloudSun,
  FileSearch,
  Globe,
  LocateFixed,
  MapPin,
  TrendingUp,
  Wrench,
  type LucideIcon,
} from "lucide-react";

export type ToolTone = "sky" | "violet" | "pink" | "emerald" | "amber" | "indigo";

interface ToolMeta {
  label: string;
  /** Present-tense line shown while the call is in flight. */
  running: string;
  icon: LucideIcon;
  tone: ToolTone;
  /** A short description of *what* the call is about, from its real args/result. */
  subject: (tool: ToolActivity) => string | null;
}

function str(value: unknown): string | null {
  if (typeof value === "string" && value.trim()) return value.trim();
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  return null;
}

const OPERATORS: Record<string, string> = {
  add: "+",
  subtract: "−",
  multiply: "×",
  divide: "÷",
};

const KNOWN: Record<string, ToolMeta> = {
  duckduckgo_search: {
    label: "Web search",
    running: "Searching the web",
    icon: Globe,
    tone: "sky",
    subject: (t) => str(t.args.query),
  },
  duckduckgo_search_unavailable: {
    label: "Web search",
    running: "Searching the web",
    icon: Globe,
    tone: "sky",
    subject: (t) => str(t.args.query),
  },
  rag_search: {
    label: "Document search",
    running: "Reading your PDF",
    icon: FileSearch,
    tone: "violet",
    subject: (t) => str(t.args.query),
  },
  Mathematical_calculations: {
    label: "Calculator",
    running: "Calculating",
    icon: Calculator,
    tone: "amber",
    subject: (t) => {
      const a = str(t.args.num1);
      const b = str(t.args.num2);
      const op = str(t.args.operation)?.toLowerCase();
      if (!a || !b || !op) return null;
      return `${a} ${OPERATORS[op] ?? op} ${b}`;
    },
  },
  get_stock_price: {
    label: "Stock price",
    running: "Fetching the latest price",
    icon: TrendingUp,
    tone: "emerald",
    subject: (t) => str(t.args.symbol)?.toUpperCase() ?? null,
  },
  get_current_location: {
    label: "Your location",
    running: "Finding where you are",
    icon: LocateFixed,
    tone: "pink",
    subject: (t) => str(t.data?.label) ?? str(t.data?.city),
  },
  geocode_location: {
    label: "Place lookup",
    running: "Looking up the place",
    icon: MapPin,
    tone: "pink",
    subject: (t) => str(t.data?.label) ?? str(t.args.place),
  },
  get_weather: {
    label: "Weather",
    running: "Checking live weather",
    icon: CloudSun,
    tone: "sky",
    subject: (t) => str(t.data?.location) ?? str(t.args.location_name),
  },
};

/** "spending_by_category" / "getTopMerchants" → "Spending by category". */
function humanize(name: string): string {
  const words = name
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/[_-]+/g, " ")
    .trim()
    .toLowerCase();
  return words ? words.charAt(0).toUpperCase() + words.slice(1) : name;
}

/** Tools not listed above (e.g. MCP servers) get a generic, still-honest card. */
export function toolMeta(name: string): ToolMeta {
  const known = KNOWN[name];
  if (known) return known;
  const label = humanize(name);
  return {
    label,
    running: `Running ${label.toLowerCase()}`,
    icon: Wrench,
    tone: "indigo",
    subject: (t) => {
      for (const value of Object.values(t.args)) {
        const s = str(value);
        if (s) return s;
      }
      return null;
    },
  };
}

/** A tool's own error message, when its JSON result carried one. */
export function toolErrorMessage(tool: ToolActivity): string | null {
  const err = tool.data?.error;
  if (err && typeof err === "object" && "message" in err) {
    return str((err as { message: unknown }).message);
  }
  return null;
}

export const TONE_CLASSES: Record<ToolTone, string> = {
  sky: "bg-sky-500/12 text-sky-600 dark:text-sky-300",
  violet: "bg-violet-500/12 text-violet-600 dark:text-violet-300",
  pink: "bg-pink-500/12 text-pink-600 dark:text-pink-300",
  emerald: "bg-emerald-500/12 text-emerald-600 dark:text-emerald-300",
  amber: "bg-amber-500/14 text-amber-600 dark:text-amber-300",
  indigo: "bg-indigo-500/12 text-indigo-600 dark:text-indigo-300",
};

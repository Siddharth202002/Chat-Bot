import type { Metadata, Viewport } from "next";
import { Plus_Jakarta_Sans } from "next/font/google";
import { ToastProvider } from "./components/ui/Toast";
import { THEME_PREF_KEY } from "./lib/theme";
import "./globals.css";

const jakarta = Plus_Jakarta_Sans({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Zeno AI",
  description:
    "Zeno AI — an assistant with real-time streaming answers, web search, live weather and PDF-grounded search.",
  // Declared explicitly so the brand mark wins over app/favicon.ico, which
  // Next would otherwise serve as the tab icon.
  icons: {
    icon: [{ url: "/brand/zeno-mark-spark.svg", type: "image/svg+xml" }],
    apple: [{ url: "/brand/zeno-mark-spark.svg" }],
  },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#e3e8fb" },
    { media: "(prefers-color-scheme: dark)", color: "#0d0e1c" },
  ],
  width: "device-width",
  initialScale: 1,
  // Keep the composer usable on iOS without zoom-on-focus surprises.
  maximumScale: 5,
};

// Runs before first paint so a stored dark preference never flashes light.
const THEME_BOOTSTRAP = `try{if(localStorage.getItem(${JSON.stringify(
  THEME_PREF_KEY
)})==="dark")document.documentElement.dataset.theme="dark"}catch(e){}`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    // The bootstrap script may set data-theme before React hydrates.
    <html lang="en" className={jakarta.variable} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOTSTRAP }} />
      </head>
      <body className="font-[family-name:var(--font-sans)] antialiased">
        <ToastProvider>{children}</ToastProvider>
      </body>
    </html>
  );
}

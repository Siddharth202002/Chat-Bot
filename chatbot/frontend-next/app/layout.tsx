import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import { ToastProvider } from "./components/ui/Toast";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Zeno AI",
  description:
    "Zeno AI — an assistant with real-time streaming answers and PDF-grounded search.",
  // Declared explicitly so the brand mark wins over app/favicon.ico, which
  // Next would otherwise serve as the tab icon.
  icons: {
    icon: [{ url: "/brand/zeno-mark-spark.svg", type: "image/svg+xml" }],
    apple: [{ url: "/brand/zeno-mark-spark.svg" }],
  },
};

export const viewport: Viewport = {
  themeColor: "#0b0b12",
  width: "device-width",
  initialScale: 1,
  // Keep the composer usable on iOS without zoom-on-focus surprises.
  maximumScale: 5,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="font-[family-name:var(--font-inter)] antialiased">
        <ToastProvider>{children}</ToastProvider>
      </body>
    </html>
  );
}

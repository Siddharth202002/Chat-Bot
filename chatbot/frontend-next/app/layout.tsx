import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "Gemini AI Chat — Intelligent Conversations",
  description: "AI Chatbot powered by LangGraph and Gemini 2.5 Flash with real-time streaming",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="font-[family-name:var(--font-inter)] antialiased">
        {children}
      </body>
    </html>
  );
}

"use client";

import Logo from "../ui/Logo";

/** The empty-conversation greeting. The mark is deliberately static. */
export default function WelcomeHero() {
  return (
    <div className="flex w-full max-w-xl flex-col items-center text-center">
      <div className="mb-7 sm:mb-8 [@media(max-height:760px)]:mb-4">
        <div className="flex h-22 w-22 items-center justify-center rounded-[1.75rem] glass-strong shadow-e3 sm:h-24 sm:w-24 [@media(max-height:760px)]:h-16 [@media(max-height:760px)]:w-16 [@media(max-height:760px)]:rounded-2xl">
          <Logo size={52} className="rounded-2xl shadow-e2 [@media(max-height:760px)]:h-10! [@media(max-height:760px)]:w-10! [@media(max-height:760px)]:rounded-xl" />
        </div>
      </div>

      <h1
        className="animate-rise text-[1.75rem] font-semibold leading-tight tracking-tight text-fg sm:text-display"
        style={{ animationDelay: "40ms" }}
      >
        How can I help you <span className="text-brand">today?</span>
      </h1>
      <p
        className="animate-rise mt-3 max-w-md text-body-lg text-fg-muted [@media(max-height:620px)]:hidden"
        style={{ animationDelay: "80ms" }}
      >
        Ask a question, search the web, check live weather, write or debug code, or
        attach a PDF and ask about it.
      </p>
    </div>
  );
}

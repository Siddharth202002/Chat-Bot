/**
 * Cosmetic repair of half-written Markdown, for streaming only.
 *
 * Tokens arrive a couple of characters at a time, so the accumulated buffer
 * spends most of its life mid-token: `Here are five solid **la`. react-markdown
 * re-parses that buffer on every frame and renders the unmatched `**` literally,
 * so the user watches asterisks flicker through the whole answer (measured at
 * 21-37% of frames across the Groq models).
 *
 * This closes the dangling markup for display. It is never stored -- the message
 * keeps the model's real text, and once the closing marker actually arrives the
 * repair becomes a no-op on its own.
 */

// Up to three leading spaces still opens a fence in CommonMark.
const FENCE_RE = /^ {0,3}(`{3,}|~{3,})/;

function isDelimiterRow(line: string): boolean {
  return /^[\s|:-]+$/.test(line) && line.includes("-");
}

/**
 * Which lines sit inside a fenced code block, and which fence is still open.
 *
 * Markers inside a fence are literal text, so nothing in here may be "closed".
 */
function scanFences(lines: string[]): { isCode: boolean[]; openFence: string | null } {
  const isCode: boolean[] = [];
  let fence: string | null = null;

  for (const line of lines) {
    const match = FENCE_RE.exec(line);
    if (fence) {
      isCode.push(true);
      // A closing fence is the same character, at least as long, and alone.
      if (match && match[1][0] === fence[0] && match[1].length >= fence.length) {
        fence = null;
      }
    } else if (match) {
      isCode.push(true);
      fence = match[1];
    } else {
      isCode.push(false);
    }
  }

  return { isCode, openFence: fence };
}

/**
 * Trailing table lines that cannot render as a table yet.
 *
 * A row arriving cell by cell, and a header row whose delimiter row has not
 * landed, both render as a paragraph of raw pipes. Holding them back for a few
 * frames is far less jarring than showing the pipes and then replacing them.
 */
function dropUnrenderableTableTail(lines: string[], isCode: boolean[]): string[] {
  const isPipeRow = (index: number) => !isCode[index] && /^\s*\|/.test(lines[index]);

  let end = lines.length;
  // A trailing "" is just the newline the last real line ends with.
  if (end > 0 && lines[end - 1] === "") end--;

  if (end > 0 && isPipeRow(end - 1) && !/\|\s*$/.test(lines[end - 1])) {
    end--;
  }

  let start = end;
  while (start > 0 && isPipeRow(start - 1)) start--;

  if (start < end && !lines.slice(start, end).some(isDelimiterRow)) {
    end = start;
  }

  return lines.slice(0, end);
}

/** Emphasis markers we repair. `_` is left alone: snake_case would trip it. */
const PAIRED = ["~~", "**", "*"] as const;

/**
 * The closers needed to balance the inline markup in `text`.
 *
 * Openers and closers are distinguished by CommonMark's flanking rule -- an
 * opener is not followed by a space, a closer is not preceded by one -- so
 * arithmetic like `3 * 4` and bullets like `* item` are left as literal text.
 */
function danglingClosers(text: string): string {
  const stack: string[] = [];
  let index = 0;
  let atLineStart = true;

  while (index < text.length) {
    const char = text[index];

    if (char === "\\") {
      index += 2;
      continue;
    }

    if (char === "\n") {
      atLineStart = true;
      index += 1;
      continue;
    }

    if (char === "`") {
      let run = 0;
      while (text[index + run] === "`") run++;
      const closer = new RegExp(`(?<!\`)\`{${run}}(?!\`)`).exec(text.slice(index + run));
      if (!closer) {
        // An unclosed span swallows the rest of the text, so nothing after it
        // can be an unbalanced marker. The span closes first, then whatever was
        // already open around it -- `*` before a code span still needs its `*`.
        return "`".repeat(run) + stack.reverse().join("");
      }
      index += run + closer.index + run;
      atLineStart = false;
      continue;
    }

    if (char === "*" || char === "~") {
      // "* item" and "- item" bullets, and "***" rules, are not emphasis.
      if (atLineStart && char === "*" && /^\*+\s/.test(text.slice(index))) {
        index += 1;
        atLineStart = false;
        continue;
      }

      const marker = PAIRED.find((candidate) => text.startsWith(candidate, index));
      if (marker) {
        const next = text[index + marker.length] ?? "";
        const previous = index > 0 ? text[index - 1] : "";
        const canClose = previous !== "" && !/\s/.test(previous);
        const canOpen = next !== "" && !/\s/.test(next);

        if (canClose && stack[stack.length - 1] === marker) {
          stack.pop();
        } else if (canOpen) {
          stack.push(marker);
        }
        index += marker.length;
        atLineStart = false;
        continue;
      }
    }

    if (!/\s/.test(char)) atLineStart = false;
    index += 1;
  }

  return stack.reverse().join("");
}

/**
 * `text` with any half-written markup closed off, for rendering mid-stream.
 */
export function closeDanglingMarkup(text: string): string {
  if (!text) return text;

  const { isCode, openFence } = scanFences(text.split("\n"));

  // Inside a fence everything is literal, so the only repair is the fence
  // itself -- otherwise the user sees ``` and unhighlighted source.
  if (openFence) {
    return text.endsWith("\n") ? `${text}${openFence}` : `${text}\n${openFence}`;
  }

  const kept = dropUnrenderableTableTail(text.split("\n"), isCode).join("\n");

  // Two things have to come off the tail first. A half-arrived marker run has
  // nothing to wrap yet, and an empty `**` renders literally anyway; and a
  // closer is only a closer when the character before it is not a space, so
  // `**Determine ` + `**` would still show all four asterisks. Trailing
  // whitespace comes back with the next token, so dropping it costs nothing.
  const body = kept.replace(/[\s*~`]*$/, "");

  return body + danglingClosers(body);
}

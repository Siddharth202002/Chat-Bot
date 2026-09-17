/**
 * Streaming replay: renders every intermediate frame of a real model response
 * through the app's markdown pipeline and asserts no literal "*" survives.
 *
 * The fixture holds real captures from every provider in the chain, replayed at
 * the chunk size that provider actually streamed at. This is the check that
 * uses the real remark-gfm parser; the Python suite asserts the structural
 * property underneath it, since there is no Markdown parser on that side.
 *
 * Run: node app/lib/streamingMarkdown.test.mjs [capture.json]
 */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert";
import { fileURLToPath } from "node:url";
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import { closeDanglingMarkup } from "./streamingMarkdown.ts";

const DEFAULT_FIXTURE = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../../tests/fixtures/provider_captures.json"
);

const proc = unified().use(remarkParse).use(remarkGfm);
const textOf = (n, a) => {
  if (n.type === "text") a.push(n.value);
  (n.children || []).forEach((c) => textOf(c, a));
  return a;
};
// inlineCode/code are literal by definition; stars inside them are not a bug.
const renderedText = (md) => {
  const tree = proc.parse(md);
  proc.runSync(tree);
  return textOf(tree, []).join("");
};

const fixture = JSON.parse(
  fs.readFileSync(process.argv[2] || DEFAULT_FIXTURE, "utf8")
);
// Replay the pre-fix captures: they are the ones with markup to get wrong.
const cap = fixture.before_fix ?? fixture;
let failures = 0;
let checked = 0;

for (const [prov, runs] of Object.entries(cap)) {
  for (const [pk, r] of Object.entries(runs)) {
    if (!r.raw) continue;
    const t = r.raw;
    const step = Math.max(1, Math.round(t.length / (r.n_chunks || 1)));
    let before = 0;
    let after = 0;
    let frames = 0;
    let worst = null;

    for (let i = step; i <= t.length; i += step) {
      const partial = t.slice(0, i);
      frames++;
      if (/\*/.test(renderedText(partial))) before++;
      const repaired = renderedText(closeDanglingMarkup(partial));
      if (/\*/.test(repaired)) {
        after++;
        if (!worst) worst = partial.slice(-70);
      }
    }
    checked++;
    const ok = after === 0;
    if (!ok) failures++;
    console.log(
      `${ok ? "PASS" : "FAIL"} ${prov.padEnd(12)}/${pk.padEnd(7)} frames=${String(frames).padEnd(5)} starFrames ${String(before).padEnd(4)} -> ${after}`
    );
    if (worst) console.log("     first bad frame tail:", JSON.stringify(worst));
  }
}

// Repair must be a no-op once the text is complete and well formed. Degenerate
// samples are excluded: gemini/list5 is the 15k-whitespace collapse, which the
// backend now rejects outright, so "preserve it verbatim" is not a property
// worth holding.
const alnumRatio = (s) =>
  (s.match(/[\p{L}\p{N}]/gu) || []).length / Math.max(1, s.length);
let noops = 0;
for (const [prov, runs] of Object.entries(cap)) {
  for (const [pk, r] of Object.entries(runs)) {
    if (!r.raw) continue;
    if (alnumRatio(r.raw) < 0.12) {
      console.log(`SKIP ${prov}/${pk} no-op check: degenerate (alnum ${alnumRatio(r.raw).toFixed(3)})`);
      continue;
    }
    assert.strictEqual(
      renderedText(closeDanglingMarkup(r.raw)),
      renderedText(r.raw),
      `${prov}/${pk}: repair altered already-complete text`
    );
    noops++;
  }
}
console.log(`\nno-op on complete text: PASS (${noops} samples)`);
console.log(failures ? `\n${failures} FAILED` : `\nALL ${checked} PASSED`);
process.exit(failures ? 1 : 0);

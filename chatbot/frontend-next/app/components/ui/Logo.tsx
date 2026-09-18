import { cn } from "@/app/lib/utils";

/**
 * The Zeno AI mark.
 *
 * The single place the brand asset is named, so swapping the file or its path
 * is a one-line change rather than a hunt through four components.
 *
 * The mark carries its own gradient tile and 25% corner radius, so it needs
 * no background or rounding of its own — pass `rounded-*` only where a shadow
 * has to follow the corners (a box-shadow otherwise squares them off).
 *
 * Decorative by default: every placement sits beside the visible words "Zeno
 * AI" or a heading, so announcing the mark as well would just repeat it.
 */
export default function Logo({
  size = 28,
  className,
}: {
  /** Rendered edge length in px. Also set as width/height, so no layout shift. */
  size?: number;
  className?: string;
}) {
  return (
    // eslint-disable-next-line @next/next/no-img-element -- a fixed-size local
    // SVG has nothing for next/image to optimise; it would only add a wrapper.
    <img
      src="/brand/zeno-mark-spark.svg"
      alt=""
      aria-hidden
      width={size}
      height={size}
      draggable={false}
      className={cn("shrink-0 select-none", className)}
      style={{ width: size, height: size }}
    />
  );
}

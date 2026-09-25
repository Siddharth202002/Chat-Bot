import { memo } from "react";

/**
 * Slow, silk-like waves behind the whole app.
 *
 * How it stays cheap and seamless:
 * - Each wave is a static SVG path, drawn twice as wide as the screen and
 *   exactly periodic over half its width. Sliding it by -50% therefore lands
 *   on an identical frame, so the loop never visibly resets.
 * - Only `transform` is animated (see `.wave-*` in globals.css), so the
 *   browser moves already-painted layers instead of repainting each frame.
 * - The paths are computed once at module load and the component takes no
 *   props, so the chat's per-token re-renders never touch it.
 *
 * Colours, opacity and speed are CSS variables (`--wave-*`), so the light
 * and dark themes swap them without any JavaScript.
 */

/** One horizontal period of every wave, in SVG units. */
const PERIOD = 1440;
const HEIGHT = 320;

interface WaveSpec {
  /** Resting height of the wave's crest line, from the top of its box. */
  base: number;
  /** [amplitude, whole cycles per period, phase] for each summed sine. */
  harmonics: [number, number, number][];
}

// Whole-number cycle counts keep every wave periodic over PERIOD, which is
// what makes the -50% slide seamless.
const WAVES: WaveSpec[] = [
  { base: 150, harmonics: [[34, 1, 0.0], [14, 3, 1.3]] },
  { base: 175, harmonics: [[28, 2, 2.1], [12, 1, 0.4]] },
  { base: 200, harmonics: [[24, 1, 3.6], [10, 4, 0.9]] },
  { base: 225, harmonics: [[18, 3, 5.0], [9, 2, 2.7]] },
];

/** The crest as an open curve, plus the closed shape filled beneath it. */
function wavePaths({ base, harmonics }: WaveSpec): { crest: string; fill: string } {
  const width = PERIOD * 2;
  const step = 24;
  const points: [number, number][] = [];
  for (let x = 0; x <= width; x += step) {
    let y = base;
    for (const [amp, cycles, phase] of harmonics) {
      y += amp * Math.sin((2 * Math.PI * cycles * x) / PERIOD + phase);
    }
    points.push([x, y]);
  }
  // Quadratic smoothing through segment midpoints: a soft curve with no kinks.
  let d = `M${points[0][0]},${points[0][1].toFixed(1)}`;
  for (let i = 1; i < points.length - 1; i++) {
    const [x, y] = points[i];
    const [nx, ny] = points[i + 1];
    d += ` Q${x},${y.toFixed(1)} ${((x + nx) / 2).toFixed(1)},${((y + ny) / 2).toFixed(1)}`;
  }
  const [lx, ly] = points[points.length - 1];
  d += ` L${lx},${ly.toFixed(1)}`;
  return { crest: d, fill: `M0,${HEIGHT} L${d.slice(1)} L${width},${HEIGHT} Z` };
}

const PATHS = WAVES.map(wavePaths);

/**
 * `index` picks the shape and colour (`--wave-N`); `slot` names this instance,
 * since the same shape appears on both edges and SVG ids must be unique.
 */
function Wave({ index, slot }: { index: number; slot: string }) {
  const id = `zeno-wave-${slot}`;
  return (
    <div className={`wave-layer wave-${index + 1} wave-slot-${slot}`}>
      <div className="wave-swell">
        <svg
          viewBox={`0 0 ${PERIOD * 2} ${HEIGHT}`}
          preserveAspectRatio="none"
          className="wave-svg"
          focusable="false"
        >
          <defs>
            {/* Solid towards the bottom, fading out at the crest: depth
                without a hard edge. */}
            <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" style={{ stopColor: `var(--wave-${index + 1})`, stopOpacity: 0 }} />
              <stop offset="45%" style={{ stopColor: `var(--wave-${index + 1})`, stopOpacity: 0.85 }} />
              <stop offset="100%" style={{ stopColor: `var(--wave-${index + 1})`, stopOpacity: 1 }} />
            </linearGradient>
          </defs>
          <path d={PATHS[index].fill} fill={`url(#${id})`} />
          {/* A thin, soft highlight on the crest gives the silk sheen. */}
          <path
            d={PATHS[index].crest}
            className="wave-crest"
            fill="none"
            vectorEffect="non-scaling-stroke"
          />
        </svg>
      </div>
    </div>
  );
}

function AnimatedWaveBackground() {
  return (
    <div className="wave-bg" aria-hidden>
      {/* A faint mirrored pair along the top frames the upper corners. */}
      <div className="wave-edge wave-edge-top">
        <Wave index={3} slot="t1" />
        <Wave index={1} slot="t2" />
      </div>
      {/* Back to front: the tallest (the accent wave) first, so each shorter
          wave in front leaves the one behind it visible above its crest. */}
      <div className="wave-edge wave-edge-bottom">
        <Wave index={0} slot="b1" />
        <Wave index={1} slot="b2" />
        <Wave index={2} slot="b3" />
        <Wave index={3} slot="b4" />
      </div>
    </div>
  );
}

export default memo(AnimatedWaveBackground);

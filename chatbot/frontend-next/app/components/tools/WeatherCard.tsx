"use client";

import { cn } from "@/app/lib/utils";
import {
  Cloud,
  CloudDrizzle,
  CloudFog,
  CloudLightning,
  CloudRain,
  CloudSnow,
  CloudSun,
  Droplets,
  Sun,
  Umbrella,
  Wind,
  type LucideIcon,
} from "lucide-react";

function num(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function text(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function iconFor(condition: string | null): LucideIcon {
  const c = (condition ?? "").toLowerCase();
  if (/thunder|storm/.test(c)) return CloudLightning;
  if (/snow|sleet|ice/.test(c)) return CloudSnow;
  if (/drizzle/.test(c)) return CloudDrizzle;
  if (/rain|shower/.test(c)) return CloudRain;
  if (/fog|mist|haze|smoke|dust/.test(c)) return CloudFog;
  if (/partly|few|scattered/.test(c)) return CloudSun;
  if (/cloud|overcast/.test(c)) return Cloud;
  if (/clear|sun/.test(c)) return Sun;
  return CloudSun;
}

/** Whether a weather result has the one reading the card is built around. */
export function hasWeatherReading(
  data: Record<string, unknown> | null
): data is Record<string, unknown> {
  return data !== null && num(record(data.current).temperature) !== null;
}

/**
 * The finished `get_weather` call, rendered from the tool's own JSON result.
 * Every field is optional because the provider omits what it did not return;
 * missing values are left out rather than filled in.
 */
export default function WeatherCard({ data }: { data: Record<string, unknown> }) {
  const current = record(data.current);
  const outlook = record(data.outlook);
  const units = record(current.units);

  const location = text(data.location);
  const temperature = num(current.temperature);
  if (temperature === null) return null;

  const tempUnit = text(units.temperature) ?? "°";
  const windUnit = text(units.wind_speed) ?? "";
  const condition = text(current.condition);
  const feelsLike = num(current.feels_like);
  const humidity = num(current.humidity);
  const wind = num(current.wind_speed);
  const high = num(outlook.high);
  const low = num(outlook.low);
  const rain = num(outlook.precipitation_probability);
  const Icon = iconFor(condition);

  const stats: { icon: LucideIcon; label: string; value: string }[] = [];
  if (humidity !== null) stats.push({ icon: Droplets, label: "Humidity", value: `${humidity}%` });
  if (wind !== null) stats.push({ icon: Wind, label: "Wind", value: `${Math.round(wind * 10) / 10} ${windUnit}` });
  if (rain !== null) stats.push({ icon: Umbrella, label: "Rain", value: `${rain}%` });

  return (
    <div
      className={cn(
        "animate-pop-in relative w-full max-w-sm overflow-hidden rounded-3xl p-5",
        "glass-strong shadow-e2"
      )}
    >
      {/* Soft sky wash behind the reading */}
      <div
        className="pointer-events-none absolute -right-10 -top-12 h-40 w-40 rounded-full bg-sky-300/40 blur-3xl dark:bg-sky-500/20"
        aria-hidden
      />
      <div
        className="pointer-events-none absolute -bottom-16 -left-10 h-40 w-40 rounded-full bg-violet-300/35 blur-3xl dark:bg-violet-500/20"
        aria-hidden
      />

      <div className="relative flex items-start justify-between gap-4">
        <div className="min-w-0">
          <p className="text-micro font-semibold uppercase tracking-wider text-fg-subtle">
            Latest reading
          </p>
          {location && (
            <p className="mt-0.5 truncate text-small font-medium text-fg" title={location}>
              {location}
            </p>
          )}
          <p className="mt-2 flex items-start text-fg">
            <span className="text-[2.75rem] font-semibold leading-none tracking-tight">
              {Math.round(temperature)}
            </span>
            <span className="ml-0.5 mt-1 text-body-lg font-medium text-fg-muted">{tempUnit}</span>
          </p>
          {(condition || feelsLike !== null) && (
            <p className="mt-1.5 text-small text-fg-muted">
              {condition && <span className="capitalize">{condition}</span>}
              {condition && feelsLike !== null && " · "}
              {feelsLike !== null && <>Feels like {Math.round(feelsLike)}{tempUnit}</>}
            </p>
          )}
        </div>

        <div className="flex flex-col items-end gap-2">
          <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-brand text-white shadow-glow">
            <Icon className="h-7 w-7" strokeWidth={1.75} aria-hidden />
          </span>
          {(high !== null || low !== null) && (
            <p className="whitespace-nowrap text-micro font-medium text-fg-muted">
              {high !== null && <>H {Math.round(high)}°</>}
              {high !== null && low !== null && "  ·  "}
              {low !== null && <>L {Math.round(low)}°</>}
            </p>
          )}
        </div>
      </div>

      {stats.length > 0 && (
        <dl className="relative mt-4 grid grid-cols-3 gap-2">
          {stats.map(({ icon: StatIcon, label, value }) => (
            <div key={label} className="rounded-2xl bg-raised/70 px-3 py-2 dark:bg-white/5">
              <dt className="flex items-center gap-1 text-micro text-fg-subtle">
                <StatIcon className="h-3 w-3" strokeWidth={2} aria-hidden />
                {label}
              </dt>
              <dd className="mt-0.5 text-small font-semibold text-fg">{value}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}

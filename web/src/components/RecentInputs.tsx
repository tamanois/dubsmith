import React from "react";
import { api } from "../api";
import type { Favorite, FileInfo } from "../types";
import { FilePickerDialog } from "./FilePickerDialog";

type Item = { key: string; label: string; path: string; name: string };

//: How many quick-access chips to show before offering "show all". Keeps the
//: band from growing without bound as favorites accumulate.
const CHIP_LIMIT = 8;

function basename(path: string): string {
  return path.split(/[\\/]/).pop() ?? path;
}

function suffixAllows(name: string, accept?: string): boolean {
  if (!accept) return true;
  const allowed = accept
    .split(",")
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean);
  if (!allowed.length) return true;
  const lower = name.toLowerCase();
  return allowed.some((ext) => lower.endsWith(ext));
}

/**
 * Recently uploaded files and explicit favorites, for picking an input without
 * opening the Files page. Also lets the user pin the current input and browse
 * every upload through a paginated picker.
 */
export function RecentInputs({
  onPick,
  current,
  accept,
}: {
  onPick: (path: string, name: string) => void;
  current?: { path: string; name: string } | null;
  accept?: string;
}) {
  const [files, setFiles] = React.useState<FileInfo[]>([]);
  const [favorites, setFavorites] = React.useState<Favorite[]>([]);
  const [error, setError] = React.useState<string | null>(null);
  const [expanded, setExpanded] = React.useState(false);
  const [browsing, setBrowsing] = React.useState(false);

  const load = React.useCallback(() => {
    api
      .files({ pageSize: 8 })
      .then((data) => setFiles(data.files))
      .catch(() => undefined);
    api
      .favorites()
      .then((data) => setFavorites(data.favorites))
      .catch(() => undefined);
  }, []);

  React.useEffect(load, [load]);

  async function pin() {
    if (!current) return;
    setError(null);
    try {
      await api.addFavorite({ path: current.path, label: current.name, kind: "input" });
      load();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Could not save the favorite");
    }
  }

  const favoritePaths = new Set(favorites.map((item) => item.path));
  const items: Item[] = [
    ...favorites.map((item) => ({
      key: `favorite-${item.path}`,
      label: item.label ?? basename(item.path),
      path: item.path,
      name: item.label ?? basename(item.path),
    })),
    ...files
      .filter((item) => !favoritePaths.has(item.path))
      .filter((item) => suffixAllows(item.original_name, accept))
      .map((item) => ({
        key: `upload-${item.id}`,
        label: item.original_name,
        path: item.path,
        name: item.original_name,
      })),
  ];
  const visible = expanded ? items : items.slice(0, CHIP_LIMIT);
  const hidden = items.length - visible.length;

  return (
    <div className="space-y-1">
      <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
        <span>Recent / favorite inputs:</span>
        {items.length === 0 ? (
          <span className="text-slate-600">none yet</span>
        ) : (
          visible.map((item) => (
            <button
              key={item.key}
              type="button"
              onClick={() => onPick(item.path, item.name)}
              className="max-w-[14rem] truncate rounded-full border border-white/10 px-2 py-0.5 text-slate-300 transition hover:border-white/20 hover:text-white"
            >
              {item.label}
            </button>
          ))
        )}
        {hidden > 0 ? (
          <button
            type="button"
            onClick={() => setExpanded(true)}
            className="rounded-full border border-white/10 px-2 py-0.5 text-slate-400 hover:text-slate-200"
          >
            +{hidden} more
          </button>
        ) : null}
        <button
          type="button"
          onClick={() => setBrowsing(true)}
          className="rounded-full border border-white/10 px-2 py-0.5 text-slate-300 hover:border-white/20 hover:text-white"
        >
          Browse all…
        </button>
        {current ? (
          <button
            type="button"
            onClick={pin}
            className="rounded-full border border-accent-500/30 px-2 py-0.5 text-accent-300 hover:bg-accent-500/10"
          >
            ★ Save current input
          </button>
        ) : null}
      </div>
      {error ? <div className="text-xs text-rose-300">{error}</div> : null}

      <FilePickerDialog
        open={browsing}
        title="Choose an uploaded file"
        accept={accept}
        onPick={onPick}
        onClose={() => setBrowsing(false)}
      />
    </div>
  );
}

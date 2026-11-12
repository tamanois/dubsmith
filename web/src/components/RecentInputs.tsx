import React from "react";
import { api } from "../api";
import type { Favorite, FileInfo } from "../types";

type Item = { key: string; label: string; path: string; name: string };

function basename(path: string): string {
  return path.split(/[\\/]/).pop() ?? path;
}

/**
 * Recently uploaded files and explicit favorites, for picking an input without
 * opening the Files page. Also lets the user pin the current input.
 */
export function RecentInputs({
  onPick,
  current,
}: {
  onPick: (path: string, name: string) => void;
  current?: { path: string; name: string } | null;
}) {
  const [files, setFiles] = React.useState<FileInfo[]>([]);
  const [favorites, setFavorites] = React.useState<Favorite[]>([]);
  const [error, setError] = React.useState<string | null>(null);

  const load = React.useCallback(() => {
    api
      .files({ pageSize: 6 })
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
      .map((item) => ({
        key: `upload-${item.id}`,
        label: item.original_name,
        path: item.path,
        name: item.original_name,
      })),
  ];

  return (
    <div className="space-y-1">
      <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
        <span>Recent / favorite inputs:</span>
        {items.length === 0 ? (
          <span className="text-slate-600">none yet</span>
        ) : (
          items.map((item) => (
            <button
              key={item.key}
              type="button"
              onClick={() => onPick(item.path, item.name)}
              className="rounded-full border border-white/10 px-2 py-0.5 text-slate-300 transition hover:border-white/20 hover:text-white"
            >
              {item.label}
            </button>
          ))
        )}
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
    </div>
  );
}

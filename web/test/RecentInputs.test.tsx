import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { RecentInputs } from "../src/components/RecentInputs";
import { makeFile } from "./factories";

const files = vi.fn();
const favorites = vi.fn();
const addFavorite = vi.fn();

vi.mock("../src/api", () => ({
  api: {
    files: (...args: unknown[]) => files(...args),
    favorites: (...args: unknown[]) => favorites(...args),
    addFavorite: (...args: unknown[]) => addFavorite(...args),
    fileInlineUrl: (id: string) => `/api/files/${id}/download?inline=1`,
  },
}));

beforeEach(() => {
  files.mockReset();
  favorites.mockReset();
  addFavorite.mockReset();
  files.mockResolvedValue({
    files: [makeFile({ id: "f1", original_name: "talk.mp4", path: "data/uploads/f1__talk.mp4" })],
    total: 1,
    page: 1,
    page_size: 20,
  });
  favorites.mockResolvedValue({
    favorites: [{ path: "input/a.mp4", label: "a.mp4", kind: "input", created: 1 }],
  });
  addFavorite.mockResolvedValue({
    favorite: { path: "x", kind: "input", created: 1 },
  });
});

describe("RecentInputs", () => {
  it("lists favorites and recent uploads and picks one", async () => {
    const onPick = vi.fn();
    render(<RecentInputs onPick={onPick} />);

    fireEvent.click(await screen.findByRole("button", { name: "a.mp4" }));
    expect(onPick).toHaveBeenCalledWith("input/a.mp4", "a.mp4");

    fireEvent.click(screen.getByRole("button", { name: "talk.mp4" }));
    expect(onPick).toHaveBeenCalledWith("data/uploads/f1__talk.mp4", "talk.mp4");
  });

  it("pins the current input as a favorite", async () => {
    render(
      <RecentInputs
        onPick={vi.fn()}
        current={{ path: "data/uploads/f1__talk.mp4", name: "talk.mp4" }}
      />,
    );
    fireEvent.click(await screen.findByRole("button", { name: "★ Save current input" }));
    await waitFor(() =>
      expect(addFavorite).toHaveBeenCalledWith({
        path: "data/uploads/f1__talk.mp4",
        label: "talk.mp4",
        kind: "input",
      }),
    );
  });

  it("browses every upload through the picker", async () => {
    const onPick = vi.fn();
    render(<RecentInputs onPick={onPick} />);
    fireEvent.click(await screen.findByRole("button", { name: "Browse all…" }));

    const dialog = await screen.findByRole("dialog", {
      name: "Choose an uploaded file",
    });
    const use = await within(dialog).findByRole("button", { name: "Use" });
    fireEvent.click(use);
    expect(onPick).toHaveBeenCalledWith(
      "data/uploads/f1__talk.mp4",
      "talk.mp4",
    );
  });
});

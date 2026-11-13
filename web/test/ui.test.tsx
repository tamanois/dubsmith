import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  Badge,
  Button,
  Disclosure,
  Empty,
  ErrorNote,
  Toggle,
  cn,
  formatBytes,
  formatDate,
  formatSeconds,
  list,
  num,
} from "../src/ui";

describe("format and parse helpers", () => {
  it("joins only truthy class names", () => {
    expect(cn("a", false, null, undefined, "b")).toBe("a b");
  });

  it("parses numbers and rejects blanks or NaN", () => {
    expect(num("")).toBeUndefined();
    expect(num("   ")).toBeUndefined();
    expect(num("12")).toBe(12);
    expect(num("-1.5")).toBe(-1.5);
    expect(num("1e3")).toBe(1000);
    expect(num("nope")).toBeUndefined();
    expect(num("Infinity")).toBeUndefined();
  });

  it("splits lists on commas and newlines", () => {
    expect(list("a, b\nc")).toEqual(["a", "b", "c"]);
    expect(list("  ")).toBeUndefined();
  });

  it("formats bytes at each boundary", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(1023)).toBe("1023 B");
    expect(formatBytes(1024)).toBe("1.0 KiB");
    expect(formatBytes(1024 ** 2)).toBe("1.0 MiB");
    expect(formatBytes(1024 ** 3)).toBe("1.00 GiB");
  });

  it("formats seconds and handles missing values", () => {
    expect(formatSeconds(null)).toBe("—");
    expect(formatSeconds(0)).toBe("0.0s");
    expect(formatSeconds(59.9)).toBe("59.9s");
    expect(formatSeconds(60)).toBe("1m 0s");
    expect(formatSeconds(125)).toBe("2m 5s");
  });

  it("formats dates and treats zero as missing", () => {
    expect(formatDate(null)).toBe("—");
    expect(formatDate(0)).toBe("—");
    expect(formatDate(1_700_000_000)).not.toBe("—");
  });
});

describe("Disclosure", () => {
  it("starts closed and toggles its content and expanded state", () => {
    render(
      <Disclosure title="Advanced">
        <div>hidden content</div>
      </Disclosure>,
    );
    const button = screen.getByRole("button", { name: /Advanced/ });
    expect(button).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("hidden content")).not.toBeInTheDocument();

    fireEvent.click(button);
    expect(button).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("hidden content")).toBeInTheDocument();

    fireEvent.click(button);
    expect(screen.queryByText("hidden content")).not.toBeInTheDocument();
  });

  it("can start open", () => {
    render(
      <Disclosure title="Advanced" defaultOpen>
        <div>visible content</div>
      </Disclosure>,
    );
    expect(screen.getByText("visible content")).toBeInTheDocument();
  });
});

describe("Toggle", () => {
  it("exposes its label and checked state", () => {
    const onChange = vi.fn();
    render(<Toggle label="Keep background" checked={false} onChange={onChange} />);
    const toggle = screen.getByRole("switch", { name: "Keep background" });
    expect(toggle).toHaveAttribute("aria-checked", "false");
    fireEvent.click(toggle);
    expect(onChange).toHaveBeenCalledWith(true);
  });
});

describe("small presentational components", () => {
  it("renders known and unknown job states", () => {
    const { rerender } = render(<Badge state="PROCESSING" />);
    expect(screen.getByText("PROCESSING")).toBeInTheDocument();
    rerender(<Badge state="SOMETHING" />);
    expect(screen.getByText("SOMETHING")).toBeInTheDocument();
  });

  it("renders empty and error messages", () => {
    render(
      <>
        <Empty text="Nothing here" />
        <ErrorNote message="Something failed" />
      </>,
    );
    expect(screen.getByText("Nothing here")).toBeInTheDocument();
    expect(screen.getByText("Something failed")).toBeInTheDocument();
  });

  it("passes the disabled attribute through Button", () => {
    render(<Button disabled>Run</Button>);
    expect(screen.getByRole("button", { name: "Run" })).toBeDisabled();
  });
});

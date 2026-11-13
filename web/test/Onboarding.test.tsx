import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { Onboarding } from "../src/components/Onboarding";

beforeEach(() => {
  window.localStorage.clear();
});

describe("Onboarding", () => {
  it("shows once and remembers the dismissal", () => {
    render(<Onboarding />);
    expect(
      screen.getByRole("dialog", { name: "Getting started" }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Got it" }));
    expect(
      screen.queryByRole("dialog", { name: "Getting started" }),
    ).not.toBeInTheDocument();
    expect(window.localStorage.getItem("dubsmith.onboarded")).toBe("1");
  });

  it("stays hidden after a previous dismissal", () => {
    window.localStorage.setItem("dubsmith.onboarded", "1");
    render(<Onboarding />);
    expect(
      screen.queryByRole("dialog", { name: "Getting started" }),
    ).not.toBeInTheDocument();
  });
});

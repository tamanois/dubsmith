import { cn } from "../ui";

const OPTIONS: { value: string; label: string }[] = [
  { value: "system", label: "System" },
  { value: "light", label: "Light" },
  { value: "dark", label: "Dark" },
];

/** Three-way theme switch. The resolved class is applied to <html> by App. */
export function ThemeToggle({
  theme,
  onChange,
}: {
  theme: string;
  onChange: (theme: string) => void;
}) {
  return (
    <div
      role="group"
      aria-label="Theme"
      className="flex rounded-lg border border-white/10 bg-ink-850 p-0.5 text-[11px]"
    >
      {OPTIONS.map((option) => (
        <button
          key={option.value}
          type="button"
          aria-pressed={theme === option.value}
          onClick={() => onChange(option.value)}
          className={cn(
            "rounded-md px-2 py-1 transition",
            theme === option.value
              ? "bg-accent-500 font-medium text-white"
              : "text-slate-400 hover:text-slate-200",
          )}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

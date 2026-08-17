export default function Home() {
  return (
    <main className="flex flex-1 flex-col items-center justify-center gap-6 px-6 text-center">
      <span className="rounded-full border border-hairline bg-surface px-4 py-1 text-sm text-muted">
        Scaffold — milestone 1
      </span>
      <h1 className="max-w-2xl text-5xl font-semibold tracking-tight">
        Your job search, on autopilot.
      </h1>
      <p className="max-w-xl text-lg text-muted">
        Upload your resume, refine your persona with a voice interview, and let
        the agent find — and apply to — the right jobs for you.
      </p>
      <div className="flex gap-3">
        <a
          href="/dashboard"
          className="rounded-(--radius-control) bg-accent px-6 py-3 font-medium text-white transition-colors hover:bg-accent-hover"
        >
          Open dashboard
        </a>
        <a
          href="#how-it-works"
          className="rounded-(--radius-control) border border-hairline bg-surface px-6 py-3 font-medium text-accent"
        >
          How it works
        </a>
      </div>
    </main>
  );
}

import Link from 'next/link';
import { Space_Grotesk } from 'next/font/google';
import { Github, Instagram } from 'lucide-react';

const space = Space_Grotesk({ subsets: ['latin'], weight: ['400', '500', '600', '700'] });

export default function LandingPage() {
  return (
    <main className={`min-h-screen bg-gradient-to-br from-slate-950 via-violet-950/50 to-slate-950 text-white relative overflow-hidden ${space.className}`}>
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute -top-24 -left-24 h-96 w-96 rounded-full bg-violet-600/20 blur-3xl" />
        <div className="absolute top-1/4 -right-24 h-96 w-96 rounded-full bg-indigo-600/20 blur-3xl" />
        <div className="absolute bottom-0 left-1/3 h-80 w-80 rounded-full bg-violet-500/10 blur-3xl" />
      </div>

      <div className="relative z-10 mx-auto max-w-6xl px-6 py-14">
        <header className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-2xl bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center shadow-xl">
              <span className="text-sm font-bold tracking-tight">SG</span>
            </div>
            <span className="text-lg font-semibold tracking-tight">SelfGPT Studio</span>
          </div>
          <Link
            href="/studio"
            className="text-sm text-slate-300 hover:text-white transition"
          >
            Enter Studio
          </Link>
        </header>

        <section className="mt-16 grid gap-10 lg:grid-cols-[1.2fr_0.8fr] items-center">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-violet-300/90">Self‑GPT powered IDE</p>
            <h1 className="mt-4 text-4xl md:text-6xl font-semibold leading-tight">
              Build, run, and ship full‑stack apps with your own AI studio.
            </h1>
            <p className="mt-4 text-slate-300 text-lg">
              SelfGPT Studio unifies multi‑file generation, live preview, terminals, agents, and versioning into one local‑first workspace.
            </p>
            <div className="mt-8 flex flex-wrap gap-4">
              <Link
                href="/studio"
                className="px-5 py-3 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 text-white font-medium hover:from-violet-500 hover:to-indigo-500 transition"
              >
                Get Started
              </Link>
              <div className="px-5 py-3 rounded-xl border border-slate-700 text-slate-200">
                Local‑first · Your keys · Your data
              </div>
            </div>
            <div className="mt-8 grid grid-cols-2 gap-4 text-sm text-slate-300">
              <div className="rounded-xl border border-slate-800/60 p-4 bg-white/5">
                Multi‑stack templates
              </div>
              <div className="rounded-xl border border-slate-800/60 p-4 bg-white/5">
                Agents for build + debug
              </div>
              <div className="rounded-xl border border-slate-800/60 p-4 bg-white/5">
                File‑level history + diff
              </div>
              <div className="rounded-xl border border-slate-800/60 p-4 bg-white/5">
                Local terminals
              </div>
            </div>
          </div>

          <div className="rounded-2xl border border-slate-800/60 bg-[#12121b]/80 p-6 shadow-2xl">
            <div className="flex items-center justify-between">
              <span className="text-xs uppercase tracking-[0.2em] text-slate-400">Workflow</span>
              <span className="text-xs text-violet-300">Local First</span>
            </div>
            <div className="mt-4 space-y-4 text-sm">
              <div className="rounded-xl border border-slate-800/60 p-4">
                1. Choose a stack (Next.js, FastAPI, MERN, Django, Flask)
              </div>
              <div className="rounded-xl border border-slate-800/60 p-4">
                2. Prompt your agent for scaffolds or fixes
              </div>
              <div className="rounded-xl border border-slate-800/60 p-4">
                3. Preview, run, diff, and export
              </div>
            </div>
            <div className="mt-6 rounded-xl bg-gradient-to-r from-slate-800 to-slate-900 p-4 text-xs text-slate-300">
              SelfGPT Studio routes to your own LLM keys. No vendor lock‑in.
            </div>
          </div>
        </section>

        <footer className="mt-16 border-t border-slate-800/60 pt-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 text-sm text-slate-400">
          <div>
            <p className="text-slate-300 font-medium">Designed & developed by Anurag</p>
          </div>
          <div className="flex flex-col sm:flex-row sm:items-center gap-3">
            <a
              href="https://github.com/anurag-m1"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 text-slate-300 hover:text-white transition"
            >
              <Github className="w-4 h-4" />
              <span className="text-xs">github.com/anurag-m1</span>
            </a>
            <a
              href="https://instagram.com/ca_anuragsingh"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 text-slate-300 hover:text-white transition"
            >
              <Instagram className="w-4 h-4" />
              <span className="text-xs">instagram.com/ca_anuragsingh</span>
            </a>
          </div>
        </footer>
      </div>
    </main>
  );
}

"use client";

import { useEffect, useRef, useState } from "react";

type Source = {
  n: number;
  title: string;
  date: string;
  url: string;
  timestamp: string;
  snippet: string;
  semantic_rank: number | null;
  keyword_rank: number | null;
};

type Turn = { role: "user" | "assistant"; text: string; sources?: Source[]; followups?: string[] };

function splitFollowups(raw: string): { answer: string; followups: string[] } {
  const idx = raw.indexOf(FOLLOWUPS_DELIM);
  if (idx === -1) return { answer: raw, followups: [] };
  const answer = raw.slice(0, idx).trimEnd();
  const followups = raw
    .slice(idx + FOLLOWUPS_DELIM.length)
    .split(/\n/)
    .map((l) => l.replace(/^\s*[-*\d.]+\s*/, "").trim())
    .filter((l) => l.length > 3)
    .slice(0, 3);
  return { answer, followups };
}

const GITHUB_URL = "https://github.com/nate-bowers/huberman-rag";
const FOLLOWUPS_DELIM = "###FOLLOWUPS###";

// Inline: **bold** and [n] citation links.
function inlineRender(text: string, sources: Source[] | undefined, key: string) {
  const tokens = text.split(/(\*\*[^*]+\*\*|\[\d+\])/g);
  return tokens.map((t, i) => {
    let m = t.match(/^\*\*([^*]+)\*\*$/);
    if (m) return <strong key={`${key}-${i}`}>{m[1]}</strong>;
    m = t.match(/^\[(\d+)\]$/);
    if (m) {
      const src = sources?.find((s) => s.n === Number(m![1]));
      if (src)
        return (
          <a key={`${key}-${i}`} className="cite" href={src.url} target="_blank" rel="noreferrer" title={src.title}>
            {t}
          </a>
        );
    }
    return <span key={`${key}-${i}`}>{t}</span>;
  });
}

// Light Markdown: paragraphs + bullet/numbered lists, with inline formatting.
function renderMarkdown(text: string, sources?: Source[]) {
  const lines = text.split(/\n/);
  type Block = { type: "p" | "ul" | "ol"; lines: string[] };
  const blocks: Block[] = [];
  let list: Block | null = null;
  const flush = () => { if (list) { blocks.push(list); list = null; } };
  for (const line of lines) {
    const ul = line.match(/^\s*[-*•]\s+(.*)/);
    const ol = line.match(/^\s*\d+\.\s+(.*)/);
    if (ul) { if (!list || list.type !== "ul") { flush(); list = { type: "ul", lines: [] }; } list.lines.push(ul[1]); }
    else if (ol) { if (!list || list.type !== "ol") { flush(); list = { type: "ol", lines: [] }; } list.lines.push(ol[1]); }
    else { flush(); if (line.trim()) blocks.push({ type: "p", lines: [line] }); }
  }
  flush();
  return blocks.map((b, i) => {
    if (b.type === "p") return <p key={i}>{inlineRender(b.lines[0], sources, `p${i}`)}</p>;
    const items = b.lines.map((l, j) => <li key={j}>{inlineRender(l, sources, `l${i}-${j}`)}</li>);
    return b.type === "ul" ? <ul key={i}>{items}</ul> : <ol key={i}>{items}</ol>;
  });
}

// Decorative animated EKG line for the hero background.
function HeroWave() {
  return (
    <svg className="herowave" viewBox="0 0 1200 200" preserveAspectRatio="none" aria-hidden>
      <path d="M0 100 H300 L330 100 L360 40 L395 165 L425 70 L450 130 L475 100 H720 L745 100 L770 55 L800 150 L828 85 L852 118 L876 100 H1200" />
    </svg>
  );
}

const EXAMPLES = [
  "What's the protocol for morning sunlight and circadian rhythm?",
  "How should I use deliberate cold exposure?",
  "When should I stop drinking caffeine to protect my sleep?",
  "How do I raise dopamine and motivation without crashing?",
  "What is non-sleep deep rest and how do I do it?",
  "What supplements does he recommend for sleep?",
  "How does zone 2 cardio improve my health?",
  "What's a fast way to calm down when I'm anxious?",
];

function Logo({ small = false }: { small?: boolean }) {
  return (
    <span className={`logo${small ? " small" : ""}`}>
      <span className="mark" aria-hidden>
        <svg viewBox="0 0 32 32" fill="none">
          {/* brain-wave / EEG pulse */}
          <path
            d="M2 16 H7 L10 16 L12.5 7 L16 25 L19 13 L21.5 19 L24 16 H30"
            stroke="#fff"
            strokeWidth="2.4"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </span>
      <span className="word">
        Huberman <span className="rag">GPT</span>
      </span>
    </span>
  );
}

export default function Home() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [placeholder, setPlaceholder] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const currentQRef = useRef(""); // full question currently shown in the typewriter

  const hero = turns.length === 0;

  // The typewriter cycles example questions on the home screen, and the latest
  // answer's follow-up suggestions once a conversation is going.
  const lastFollowups = [...turns].reverse().find((t) => t.role === "assistant" && t.followups?.length)?.followups;
  const cycle = !hero && lastFollowups && lastFollowups.length ? lastFollowups : EXAMPLES;
  const cycleKey = cycle.join("|");

  // Typewriter placeholder: type a question, pause, delete, next — on loop.
  // Pauses while the user is typing or a request is in flight.
  useEffect(() => {
    if (input || busy) return;
    let cancelled = false;
    let ei = 0;
    let ci = 0;
    let deleting = false;
    let timer: ReturnType<typeof setTimeout>;

    const schedule = (ms: number) => {
      if (!cancelled) timer = setTimeout(tick, ms);
    };
    function tick() {
      const q = cycle[ei % cycle.length];
      currentQRef.current = q;
      if (!deleting) {
        ci++;
        setPlaceholder(q.slice(0, ci));
        if (ci >= q.length) {
          deleting = true;
          schedule(1700);
        } else schedule(38 + Math.random() * 45);
      } else {
        ci--;
        setPlaceholder(q.slice(0, ci));
        if (ci <= 0) {
          deleting = false;
          ei++;
          schedule(320);
        } else schedule(22);
      }
    }
    schedule(450);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [input, busy, cycleKey]);

  async function ask(q: string) {
    if (!q.trim() || busy) return;
    setBusy(true);
    setInput("");
    setTurns((t) => [...t, { role: "user", text: q }, { role: "assistant", text: "" }]);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ query: q }),
      });

      let sources: Source[] | undefined;
      const b64 = res.headers.get("x-sources");
      if (b64) {
        try {
          sources = JSON.parse(atob(b64));
        } catch {}
      }
      setTurns((t) => {
        const c = [...t];
        c[c.length - 1] = { role: "assistant", text: "", sources };
        return c;
      });

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let acc = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        acc += decoder.decode(value, { stream: true });
        // Hide the follow-ups block while streaming (show only the answer part).
        const shown = acc.split(FOLLOWUPS_DELIM)[0];
        setTurns((t) => {
          const c = [...t];
          c[c.length - 1] = { ...c[c.length - 1], role: "assistant", text: shown, sources };
          return c;
        });
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
      }
      const { answer, followups } = splitFollowups(acc);
      setTurns((t) => {
        const c = [...t];
        c[c.length - 1] = { role: "assistant", text: answer, sources, followups };
        return c;
      });
    } catch {
      setTurns((t) => {
        const c = [...t];
        c[c.length - 1] = { role: "assistant", text: "Something went wrong. Please try again." };
        return c;
      });
    } finally {
      setBusy(false);
    }
  }

  const SearchBox = (
    <div className="searchwrap">
      <form
        className="searchbox"
        onSubmit={(e) => {
          e.preventDefault();
          // Empty input → ask the question currently shown in the typewriter.
          ask(input.trim() || currentQRef.current);
        }}
      >
        <span className="glass" aria-hidden>
          <svg width="19" height="19" viewBox="0 0 24 24" fill="none">
            <circle cx="11" cy="11" r="7" stroke="currentColor" strokeWidth="2" />
            <path d="M20 20l-3.2-3.2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
        </span>
        <input
          ref={inputRef}
          type="text"
          value={input}
          placeholder={placeholder || (hero ? "Ask about sleep, focus, dopamine…" : "Ask a follow-up…")}
          onChange={(e) => setInput(e.target.value)}
          disabled={busy}
          autoFocus
        />
        <button type="submit" disabled={busy}>
          {busy ? "…" : "Ask"}
        </button>
      </form>
    </div>
  );

  if (hero) {
    return (
      <div className="page">
        <HeroWave />
        <div className="hero">
          <Logo />
          <p className="tagline">
            Search <strong>800+ hours</strong> of the Huberman Lab podcast — answered with cited,
            timestamped sources, grounded in the transcripts.
          </p>
          {SearchBox}
          <div className="chips">
            {EXAMPLES.slice(0, 4).map((e) => (
              <button key={e} onClick={() => ask(e)}>
                {e}
              </button>
            ))}
          </div>
        </div>
        <div className="foot">
          <div className="disclaimer">
            Educational summaries of podcast content — not medical advice.
          </div>
          Hybrid (semantic + keyword) retrieval ·{" "}
          <a href={GITHUB_URL} target="_blank" rel="noreferrer">
            free &amp; open source
          </a>
        </div>
      </div>
    );
  }

  return (
    <div className="page">
      <div className="topbar">
        <Logo small />
        <button className="new" onClick={() => setTurns([])}>
          ＋ New chat
        </button>
      </div>

      <div className="thread">
        {turns.map((turn, i) => (
          <div className="msg" key={i}>
            <div className="role">{turn.role === "user" ? "You" : "Huberman RAG"}</div>
            <div className={`bubble ${turn.role}`}>
              {turn.role === "assistant"
                ? turn.text
                  ? renderMarkdown(turn.text, turn.sources)
                  : <span className="thinking"><span></span><span></span><span></span></span>
                : turn.text}
            </div>
            {turn.sources && turn.sources.length > 0 && (
              <div className="sources">
                {turn.sources.map((s) => (
                  <a className="source" key={s.n} href={s.url} target="_blank" rel="noreferrer">
                    <div className="top">
                      <span className="title">
                        <span className="cite">[{s.n}]</span> {s.title}
                      </span>
                      <span className="meta">
                        <span>{s.date}</span>
                        {s.timestamp && <span className="ts">▶ {s.timestamp}</span>}
                      </span>
                    </div>
                    <div className="snippet">{s.snippet}</div>
                  </a>
                ))}
              </div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="dock">
        {SearchBox}
        <div className="disclaimer dock-note">Educational summaries — not medical advice.</div>
      </div>
    </div>
  );
}

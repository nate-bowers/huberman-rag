"use client";

import { useRef, useState } from "react";

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

type Turn = {
  role: "user" | "assistant";
  text: string;
  sources?: Source[];
};

const EXAMPLES = [
  "What's the protocol for morning sunlight and circadian rhythm?",
  "How does Huberman recommend using cold exposure?",
  "What does he say about caffeine timing and sleep?",
  "How can I increase dopamine and motivation?",
];

export default function Home() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

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

      // Sources arrive in the header, available before the body streams.
      let sources: Source[] | undefined;
      const b64 = res.headers.get("x-sources");
      if (b64) {
        try {
          sources = JSON.parse(atob(b64));
        } catch {}
      }
      setTurns((t) => {
        const copy = [...t];
        copy[copy.length - 1] = { role: "assistant", text: "", sources };
        return copy;
      });

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let acc = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        acc += decoder.decode(value, { stream: true });
        setTurns((t) => {
          const copy = [...t];
          copy[copy.length - 1] = { ...copy[copy.length - 1], role: "assistant", text: acc, sources };
          return copy;
        });
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
      }
    } catch (e) {
      setTurns((t) => {
        const copy = [...t];
        copy[copy.length - 1] = { role: "assistant", text: "Something went wrong. Try again." };
        return copy;
      });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="wrap">
      <header>
        <h1>🧠 Huberman Lab RAG</h1>
        <p>
          Ask anything across 342 Huberman Lab episodes. Answers are grounded in transcript
          excerpts with cited, timestamped sources via hybrid (semantic + keyword) retrieval.
        </p>
      </header>

      {turns.length === 0 && (
        <div className="examples">
          {EXAMPLES.map((e) => (
            <button key={e} onClick={() => ask(e)}>
              {e}
            </button>
          ))}
        </div>
      )}

      {turns.map((turn, i) => (
        <div className="msg" key={i}>
          <div className="role">{turn.role}</div>
          <div className={`bubble ${turn.role}`}>
            {turn.text || (turn.role === "assistant" ? "…" : "")}
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
                      {s.date}
                      {s.timestamp && <span className="ts"> · ▶ {s.timestamp}</span>}
                      {s.semantic_rank != null && <span className="badge">sem #{s.semantic_rank}</span>}
                      {s.keyword_rank != null && <span className="badge">kw #{s.keyword_rank}</span>}
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

      <form
        onSubmit={(e) => {
          e.preventDefault();
          ask(input);
        }}
      >
        <div className="inputrow">
          <input
            type="text"
            value={input}
            placeholder="Ask about sleep, focus, dopamine, fitness…"
            onChange={(e) => setInput(e.target.value)}
            disabled={busy}
          />
          <button type="submit" disabled={busy || !input.trim()}>
            {busy ? "…" : "Ask"}
          </button>
        </div>
      </form>
    </div>
  );
}

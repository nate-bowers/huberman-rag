/**
 * Sliding-window rate limiter for the public /api/chat endpoint (it spends LLM
 * tokens, so it's an abuse/cost vector). Backed by Upstash Redis.
 *
 * Gracefully no-ops when Upstash isn't configured, so local dev and unconfigured
 * deploys still work (just without limiting).
 */
import { Ratelimit } from "@upstash/ratelimit";
import { Redis } from "@upstash/redis";

let limiter: Ratelimit | null | undefined;

function getLimiter(): Ratelimit | null {
  if (limiter !== undefined) return limiter;
  const url = process.env.UPSTASH_REDIS_REST_URL;
  const token = process.env.UPSTASH_REDIS_REST_TOKEN;
  limiter = url && token
    ? new Ratelimit({
        redis: new Redis({ url, token }),
        limiter: Ratelimit.slidingWindow(15, "60 s"), // 15 requests / minute / IP
        prefix: "huberman-rag",
        analytics: false,
      })
    : null;
  return limiter;
}

export async function checkRateLimit(ip: string): Promise<{ ok: boolean; remaining: number }> {
  const l = getLimiter();
  if (!l) return { ok: true, remaining: -1 }; // not configured → allow
  const { success, remaining } = await l.limit(ip);
  return { ok: success, remaining };
}

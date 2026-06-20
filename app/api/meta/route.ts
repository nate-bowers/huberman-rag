/**
 * Lightweight metadata endpoint: the most recent episode in the index.
 * Cached for an hour (the corpus changes rarely).
 */
import { publicClient } from "@/lib/supabase";

export const runtime = "nodejs";
export const revalidate = 3600;

export async function GET() {
  try {
    const { data, error } = await publicClient()
      .from("chunks")
      .select("date,title")
      .order("date", { ascending: false })
      .limit(1);
    if (error || !data?.length) return Response.json({ ok: false });
    return Response.json({ ok: true, date: data[0].date, title: data[0].title });
  } catch {
    return Response.json({ ok: false });
  }
}

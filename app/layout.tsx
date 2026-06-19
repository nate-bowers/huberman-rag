import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Huberman RAG",
  description: "Ask questions across every Huberman Lab podcast episode — answered with cited, timestamped sources.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

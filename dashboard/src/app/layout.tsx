import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Panorama Agent Dashboard",
  description: "Observability dashboard for the episodic-memory panorama agent",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-ctp-crust text-ctp-text">
        {children}
      </body>
    </html>
  );
}

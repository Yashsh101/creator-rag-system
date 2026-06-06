"use client";

import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { RagApiClient } from "@/lib/api";

export type RuntimeMode = "local" | "prod";

export type ConsoleSettings = {
  apiBaseUrl: string;
  apiKey: string;
  mode: RuntimeMode;
};

// Support NEXT_PUBLIC_API_URL (Render/Railway convention) with a
// fallback to the legacy NEXT_PUBLIC_RAG_API_BASE_URL name.
// This fixes the silent "API base URL is not configured" error that
// occurred whenever the Vercel env var used the new name but the
// code only read the old one.
const envBaseUrl =
  process.env.NEXT_PUBLIC_API_URL ||
  process.env.NEXT_PUBLIC_RAG_API_BASE_URL ||
  "";

const defaultSettings: ConsoleSettings = {
  apiBaseUrl: envBaseUrl,
  apiKey: process.env.NEXT_PUBLIC_RAG_API_KEY || "",
  mode: envBaseUrl ? "prod" : "local",
};

type SettingsContextValue = {
  settings: ConsoleSettings;
  updateSettings: (next: Partial<ConsoleSettings>) => void;
  client: RagApiClient;
};

const SettingsContext = createContext<SettingsContextValue | null>(null);

export function SettingsProvider({ children }: { children: React.ReactNode }) {
  const [settings, setSettings] = useState<ConsoleSettings>(defaultSettings);

  useEffect(() => {
    const stored = window.localStorage.getItem("rag-console-settings");
    if (stored) {
      try {
        setSettings({ ...defaultSettings, ...JSON.parse(stored) });
      } catch {
        // Corrupted localStorage entry — silently discard and use env defaults.
      }
    }
  }, []);

  const updateSettings = (next: Partial<ConsoleSettings>) => {
    setSettings((current) => {
      const merged = { ...current, ...next };
      window.localStorage.setItem("rag-console-settings", JSON.stringify(merged));
      return merged;
    });
  };

  // Narrow the dependency to the two values that actually change the client
  // instance.  The old [settings] dep re-created the client on every unrelated
  // state update (e.g. mode toggle), which caused in-flight requests to be
  // abandoned when the reference changed mid-flight.
  const client = useMemo(
    () => new RagApiClient({ baseUrl: settings.apiBaseUrl, apiKey: settings.apiKey }),
    [settings.apiBaseUrl, settings.apiKey]
  );

  return (
    <SettingsContext.Provider value={{ settings, updateSettings, client }}>
      {children}
    </SettingsContext.Provider>
  );
}

export function useConsoleSettings() {
  const value = useContext(SettingsContext);
  if (!value) throw new Error("useConsoleSettings must be used inside SettingsProvider");
  return value;
}

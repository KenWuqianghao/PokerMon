import type { NewGameResponse, ActionResponse } from "./types";

async function post<T>(url: string, body: Record<string, unknown> = {}): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

export function newGame(): Promise<NewGameResponse> {
  return post("/api/game/new", {});
}

export function sendAction(gameId: string, action: number): Promise<ActionResponse> {
  return post("/api/game/action", { game_id: gameId, action });
}

export function dealAgain(gameId: string): Promise<NewGameResponse> {
  return post("/api/game/deal", { game_id: gameId, action: 0 });
}

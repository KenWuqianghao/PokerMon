import type { ActionTaken } from "../types";

interface LogEntry {
  handNum: number;
  actions: ActionTaken[];
  result?: number;
}

interface GameLogProps {
  entries: LogEntry[];
  totalHands: number;
  totalResult: number;
}

export default function GameLog({ entries, totalHands, totalResult }: GameLogProps) {
  return (
    <div className="flex flex-col h-full">
      <div className="text-zinc-400 text-xs font-medium uppercase tracking-wider mb-3">
        Game Log
      </div>
      <div className="flex-1 overflow-y-auto space-y-3 min-h-0">
        {entries.map((entry, i) => (
          <div key={i} className="text-sm">
            <div className="text-zinc-500 text-xs mb-1">Hand #{entry.handNum}</div>
            {entry.actions.map((a, j) => (
              <div key={j} className="text-zinc-400 pl-2">
                <span className={a.player === "human" ? "text-zinc-300" : "text-zinc-500"}>
                  {a.player === "human" ? "You" : "AI"}
                </span>{" "}
                {a.label}
              </div>
            ))}
            {entry.result !== undefined && (
              <div
                className={`pl-2 font-medium ${entry.result > 0 ? "text-emerald-400" : entry.result < 0 ? "text-red-400" : "text-zinc-500"}`}
              >
                {entry.result > 0 ? "+" : ""}
                {entry.result.toLocaleString()}
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="border-t border-zinc-800 pt-3 mt-3 space-y-1">
        <div className="text-zinc-500 text-xs">
          Hands: <span className="text-zinc-300 tabular-nums">{totalHands}</span>
        </div>
        <div className="text-zinc-500 text-xs">
          Net:{" "}
          <span
            className={`tabular-nums ${totalResult > 0 ? "text-emerald-400" : totalResult < 0 ? "text-red-400" : "text-zinc-300"}`}
          >
            {totalResult > 0 ? "+" : ""}
            {totalResult.toLocaleString()}
          </span>
        </div>
      </div>
    </div>
  );
}

export type { LogEntry };

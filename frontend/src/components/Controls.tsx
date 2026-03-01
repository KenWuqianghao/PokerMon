import type { LegalAction } from "../types";

interface ControlsProps {
  legalActions: LegalAction[] | null;
  onAction: (action: number) => void;
  disabled: boolean;
  isTerminal: boolean;
  onDealAgain: () => void;
  onNewGame: () => void;
  result?: number;
  gameOver?: boolean;
}

function actionStyle(label: string): string {
  if (label === "Fold")
    return "bg-zinc-800 hover:bg-zinc-700 text-zinc-400";
  if (label.startsWith("All In"))
    return "bg-emerald-600 hover:bg-emerald-500 text-white ring-1 ring-emerald-400/30";
  return "bg-zinc-800 hover:bg-zinc-700 text-zinc-200";
}

export default function Controls({
  legalActions,
  onAction,
  disabled,
  isTerminal,
  onDealAgain,
  onNewGame,
  result,
  gameOver,
}: ControlsProps) {
  if (isTerminal) {
    return (
      <div className="flex items-center gap-4">
        <span
          className={`font-semibold tabular-nums ${result !== undefined && result > 0 ? "text-emerald-400" : result !== undefined && result < 0 ? "text-red-400" : "text-zinc-400"}`}
        >
          {result !== undefined
            ? result > 0
              ? `+${result.toLocaleString()}`
              : result.toLocaleString()
            : "Push"}
        </span>
        {gameOver ? (
          <>
            <span className="text-zinc-500 text-sm">
              {result !== undefined && result > 0 ? "AI is out of chips!" : "You're out of chips!"}
            </span>
            <button
              onClick={onNewGame}
              className="px-5 py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium transition-all active:scale-[0.98] hover:scale-[1.02]"
            >
              New Game
            </button>
          </>
        ) : (
          <button
            onClick={onDealAgain}
            className="px-5 py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium transition-all active:scale-[0.98] hover:scale-[1.02]"
          >
            Deal Again
          </button>
        )}
      </div>
    );
  }

  if (!legalActions || disabled) {
    return (
      <div className="text-zinc-600 text-sm tracking-wide uppercase">
        {disabled ? "Waiting..." : "Opponent's turn"}
      </div>
    );
  }

  return (
    <div className="flex gap-2 flex-wrap">
      {legalActions.map((a) => (
        <button
          key={a.action}
          onClick={() => onAction(a.action)}
          className={`px-4 py-2.5 rounded-lg font-medium text-sm transition-all active:scale-[0.98] hover:scale-[1.02] ${actionStyle(a.label)}`}
        >
          {a.label}
        </button>
      ))}
    </div>
  );
}

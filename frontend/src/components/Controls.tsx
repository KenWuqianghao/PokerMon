import { useState } from "react";
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

// Action codes 2-6 are bet/raise/all-in (0=Fold, 1=Check/Call, 2-5=Bet fractions, 6=All-In)
function findClosestBetAction(amount: number, actions: LegalAction[]): LegalAction | null {
  const betActions = actions.filter((a) => a.action >= 2);
  if (betActions.length === 0) return null;
  return betActions.reduce((best, a) =>
    Math.abs(a.amount - amount) < Math.abs(best.amount - amount) ? a : best,
  );
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
  const [customAmount, setCustomAmount] = useState("");

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

  if (disabled) {
    return (
      <div className="text-zinc-600 text-sm tracking-wide uppercase">Waiting...</div>
    );
  }

  if (!legalActions) {
    return (
      <div className="text-zinc-600 text-sm tracking-wide uppercase">Opponent&apos;s turn</div>
    );
  }

  if (legalActions.length === 0) {
    return (
      <div className="text-red-500 text-sm">Error: no legal actions. Please refresh.</div>
    );
  }

  const parsedAmount = parseInt(customAmount, 10);
  const closestAction =
    !isNaN(parsedAmount) && parsedAmount > 0
      ? findClosestBetAction(parsedAmount, legalActions)
      : null;

  return (
    <div className="flex flex-col gap-3">
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
      <div className="flex items-center gap-2">
        <input
          type="number"
          value={customAmount}
          onChange={(e) => setCustomAmount(e.target.value)}
          placeholder="Custom raise..."
          className="w-36 px-3 py-2 rounded-lg bg-zinc-900 border border-zinc-700 text-zinc-200 text-sm focus:outline-none focus:border-zinc-500 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
        />
        <button
          onClick={() => {
            if (closestAction) {
              onAction(closestAction.action);
              setCustomAmount("");
            }
          }}
          disabled={!closestAction}
          className={`px-4 py-2 rounded-lg font-medium text-sm transition-all ${
            closestAction
              ? "bg-zinc-700 hover:bg-zinc-600 text-zinc-200 active:scale-[0.98] hover:scale-[1.02]"
              : "bg-zinc-900 text-zinc-600 cursor-not-allowed"
          }`}
        >
          {closestAction
            ? closestAction.action === 6
              ? `All In ${closestAction.amount.toLocaleString()}`
              : `Raise ${closestAction.amount.toLocaleString()}`
            : "Raise"}
        </button>
      </div>
    </div>
  );
}

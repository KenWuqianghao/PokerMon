import Card from "./Card";

interface PlayerHandProps {
  label: string;
  cards: string[] | null;
  stack: number;
  bet: number;
  isButton: boolean;
  folded: boolean;
  allIn: boolean;
  handRank?: string;
  flip?: boolean;
}

export default function PlayerHand({
  label,
  cards,
  stack,
  bet,
  isButton,
  folded,
  allIn,
  handRank,
  flip,
}: PlayerHandProps) {
  const faceDown = !cards;

  return (
    <div className={`flex items-center gap-6 ${folded ? "opacity-40" : ""}`}>
      <div className="flex gap-2">
        {faceDown ? (
          <>
            <Card card={null} faceDown delay={0} />
            <Card card={null} faceDown delay={60} />
          </>
        ) : (
          cards.map((c, i) => (
            <Card key={c} card={c} delay={i * 60} flip={flip} />
          ))
        )}
      </div>
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-2">
          <span className="text-zinc-300 font-medium">{label}</span>
          {isButton && (
            <span className="text-[10px] font-semibold bg-zinc-700 text-zinc-300 px-1.5 py-0.5 rounded">
              BTN
            </span>
          )}
          {allIn && (
            <span className="text-[10px] font-semibold bg-emerald-900 text-emerald-300 px-1.5 py-0.5 rounded">
              ALL IN
            </span>
          )}
          {folded && (
            <span className="text-[10px] font-semibold bg-zinc-800 text-zinc-500 px-1.5 py-0.5 rounded">
              FOLD
            </span>
          )}
        </div>
        <div className="text-zinc-500 text-sm tabular-nums">
          Stack: {stack.toLocaleString()}
        </div>
        {bet > 0 && (
          <div className="text-zinc-400 text-sm tabular-nums">
            Bet: {bet.toLocaleString()}
          </div>
        )}
        {handRank && (
          <div className="text-emerald-400 text-sm font-medium">{handRank}</div>
        )}
      </div>
    </div>
  );
}

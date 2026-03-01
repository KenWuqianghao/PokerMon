const SUIT_SYMBOLS: Record<string, string> = {
  s: "\u2660",
  h: "\u2665",
  d: "\u2666",
  c: "\u2663",
};

const SUIT_COLORS: Record<string, string> = {
  s: "text-zinc-400",
  h: "text-red-500",
  d: "text-red-500",
  c: "text-zinc-400",
};

const RANK_DISPLAY: Record<string, string> = {
  T: "10",
};

interface CardProps {
  card: string | null;
  faceDown?: boolean;
  delay?: number;
  flip?: boolean;
}

export default function Card({ card, faceDown, delay = 0, flip }: CardProps) {
  if (faceDown || !card) {
    return (
      <div
        className="animate-deal-in w-[80px] h-[112px] rounded-lg border border-zinc-700 flex items-center justify-center"
        style={{
          animationDelay: `${delay}ms`,
          background:
            "repeating-linear-gradient(45deg, transparent, transparent 6px, rgb(52 211 153 / 0.15) 6px, rgb(52 211 153 / 0.15) 12px)",
          backgroundColor: "rgb(39 39 42)",
        }}
      />
    );
  }

  const rank = card[0];
  const suit = card[1];
  const displayRank = RANK_DISPLAY[rank] || rank;
  const suitSymbol = SUIT_SYMBOLS[suit];
  const colorClass = SUIT_COLORS[suit];

  return (
    <div
      className={`${flip ? "animate-card-flip" : "animate-deal-in"} w-[80px] h-[112px] rounded-lg bg-zinc-100 border border-zinc-300 flex flex-col justify-between p-2 select-none`}
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className={`text-sm font-semibold leading-none ${colorClass}`}>
        <div>{displayRank}</div>
        <div>{suitSymbol}</div>
      </div>
      <div className={`text-3xl leading-none self-center ${colorClass}`}>
        {suitSymbol}
      </div>
      <div
        className={`text-sm font-semibold leading-none self-end rotate-180 ${colorClass}`}
      >
        <div>{displayRank}</div>
        <div>{suitSymbol}</div>
      </div>
    </div>
  );
}

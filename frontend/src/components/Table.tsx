import Card from "./Card";

interface TableProps {
  communityCards: string[];
  pot: number;
  street: string;
}

export default function Table({ communityCards, pot, street }: TableProps) {
  // Show placeholder slots for cards not yet dealt
  const totalSlots = street === "preflop" ? 0 : street === "flop" ? 3 : street === "turn" ? 4 : 5;
  const cards = communityCards.slice(0, totalSlots);

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="flex gap-2 min-h-[112px] items-center">
        {totalSlots === 0 ? (
          <div className="text-zinc-600 text-sm tracking-wide uppercase">
            Pre-flop
          </div>
        ) : (
          cards.map((c, i) => <Card key={`${c}-${i}`} card={c} delay={i * 80} />)
        )}
      </div>
      <div className="text-emerald-400 font-semibold text-lg tabular-nums">
        Pot: {pot.toLocaleString()}
      </div>
    </div>
  );
}

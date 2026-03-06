import { useCallback, useEffect, useRef, useState } from "react";
import { newGame, sendAction, dealAgain } from "./api";
import type { GameStateResponse, ActionTaken } from "./types";
import Table from "./components/Table";
import PlayerHand from "./components/PlayerHand";
import Controls from "./components/Controls";
import ActionBanner from "./components/ActionBanner";
import GameLog, { type LogEntry } from "./components/GameLog";

const AI_THINK_MS = 200;
const BANNER_MS = 900;

export default function App() {
  const [gameId, setGameId] = useState<string | null>(null);
  const [state, setState] = useState<GameStateResponse | null>(null);
  const [waiting, setWaiting] = useState(false);
  const [logEntries, setLogEntries] = useState<LogEntry[]>([]);
  const [currentActions, setCurrentActions] = useState<ActionTaken[]>([]);
  const [totalResult, setTotalResult] = useState(0);
  const [showAiCards, setShowAiCards] = useState(false);
  const [actionBanner, setActionBanner] = useState<{
    text: string;
    key: number;
    isAi: boolean;
  } | null>(null);
  const bannerKeyRef = useRef(0);
  const startedRef = useRef(false);

  const showBannerSequence = useCallback(
    async (actions: ActionTaken[]) => {
      for (const act of actions) {
        const who = act.player === "ai" ? "AI" : "You";
        if (act.player === "ai") {
          bannerKeyRef.current += 1;
          setActionBanner({ text: "AI thinking...", key: bannerKeyRef.current, isAi: true });
          await new Promise((r) => setTimeout(r, AI_THINK_MS));
        }
        bannerKeyRef.current += 1;
        setActionBanner({
          text: `${who} \u00b7 ${act.label}`,
          key: bannerKeyRef.current,
          isAi: act.player === "ai",
        });
        await new Promise((r) => setTimeout(r, BANNER_MS));
      }
      setActionBanner(null);
    },
    [],
  );

  const startGame = useCallback(async () => {
    setWaiting(true);
    try {
      const res = await newGame();
      setGameId(res.game_id);
      setState(res.state);
      setCurrentActions(res.actions_taken);
      setShowAiCards(false);
      if (res.actions_taken.length > 0) {
        await showBannerSequence(res.actions_taken);
      }
    } finally {
      setWaiting(false);
    }
  }, [showBannerSequence]);

  useEffect(() => {
    if (!startedRef.current) {
      startedRef.current = true;
      startGame();
    }
  }, [startGame]);

  const handleAction = useCallback(
    async (action: number) => {
      if (!gameId || waiting) return;
      setWaiting(true);
      try {
        const res = await sendAction(gameId, action);
        const allActions = [...currentActions, ...res.actions_taken];

        setState(res.state);
        setCurrentActions(allActions);

        // Show banner sequence for all actions
        if (res.actions_taken.length > 0) {
          await showBannerSequence(res.actions_taken);
        }

        if (res.state.is_terminal) {
          setShowAiCards(true);
          const result = res.state.result ?? 0;
          setTotalResult((prev) => prev + result);
          setLogEntries((prev) => [
            ...prev,
            {
              handNum: res.state.hand_num,
              actions: allActions,
              result,
            },
          ]);
        }
      } finally {
        setWaiting(false);
      }
    },
    [gameId, waiting, currentActions, showBannerSequence],
  );

  const handleNewGame = useCallback(async () => {
    setLogEntries([]);
    setTotalResult(0);
    await startGame();
  }, [startGame]);

  const handleDealAgain = useCallback(async () => {
    if (!gameId || waiting) return;
    setWaiting(true);
    try {
      const res = await dealAgain(gameId);
      setState(res.state);
      setCurrentActions(res.actions_taken);
      setShowAiCards(false);
      if (res.actions_taken.length > 0) {
        await showBannerSequence(res.actions_taken);
      }
    } finally {
      setWaiting(false);
    }
  }, [gameId, waiting, showBannerSequence]);

  if (!state) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-zinc-500">Loading...</div>
      </div>
    );
  }

  const handCount = logEntries.length + (state.is_terminal ? 0 : 1);
  const gameOver = state.is_terminal && (state.human.stack === 0 || state.ai.stack === 0);

  return (
    <div className="min-h-screen p-6">
      <div className="max-w-[1100px] mx-auto grid grid-cols-[1fr_280px] gap-6 h-[calc(100vh-48px)]">
        {/* Main game area */}
        <div className="flex flex-col justify-between">
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <h1 className="text-zinc-400 text-sm font-medium tracking-wider uppercase">
              PokerMon
            </h1>
            <div className="text-zinc-600 text-sm">
              Hand #{state.hand_num}
            </div>
          </div>

          {/* AI hand */}
          <div className="mb-8">
            <PlayerHand
              label="AI"
              cards={showAiCards ? state.ai.cards : null}
              stack={state.ai.stack}
              bet={state.ai.bet}
              isButton={state.button === 1}
              folded={state.ai.folded}
              allIn={state.ai.all_in}
              handRank={state.is_terminal ? state.hand_info?.ai : undefined}
              flip={showAiCards && !!state.ai.cards}
            />
          </div>

          {/* Community cards + pot */}
          <div className="flex-1 flex items-center justify-center relative">
            <Table
              communityCards={state.community_cards}
              pot={state.pot}
              street={state.street}
            />
            {actionBanner && (
              <ActionBanner
                text={actionBanner.text}
                bannerKey={actionBanner.key}
                isAi={actionBanner.isAi}
              />
            )}
          </div>

          {/* Human hand */}
          <div className="mt-8 mb-4">
            <PlayerHand
              label="You"
              cards={state.human.cards}
              stack={state.human.stack}
              bet={state.human.bet}
              isButton={state.button === 0}
              folded={state.human.folded}
              allIn={state.human.all_in}
              handRank={state.is_terminal ? state.hand_info?.human : undefined}
            />
          </div>

          {/* Controls */}
          <div className="py-4">
            <Controls
              legalActions={state.legal_actions}
              onAction={handleAction}
              disabled={waiting}
              isTerminal={state.is_terminal}
              onDealAgain={handleDealAgain}
              onNewGame={handleNewGame}
              result={state.result}
              gameOver={gameOver}
            />
          </div>
        </div>

        {/* Sidebar */}
        <div className="border-l border-zinc-800 pl-6 overflow-hidden">
          <GameLog
            entries={logEntries}
            totalHands={handCount}
            totalResult={totalResult}
          />
        </div>
      </div>
    </div>
  );
}

export interface PlayerInfo {
  cards: string[] | null;
  stack: number;
  bet: number;
  folded: boolean;
  all_in: boolean;
}

export interface LegalAction {
  action: number;
  label: string;
  amount: number;
}

export interface HandInfo {
  human?: string;
  ai?: string;
}

export interface GameStateResponse {
  game_id: string;
  hand_num: number;
  street: string;
  community_cards: string[];
  pot: number;
  human: PlayerInfo;
  ai: PlayerInfo;
  current_player: number;
  is_terminal: boolean;
  legal_actions: LegalAction[] | null;
  button: number;
  payoffs?: number[];
  result?: number;
  hand_info?: HandInfo;
}

export interface ActionTaken {
  player: "human" | "ai";
  action: number;
  label: string;
  amount: number;
}

export interface NewGameResponse {
  game_id: string;
  state: GameStateResponse;
  actions_taken: ActionTaken[];
}

export interface ActionResponse {
  state: GameStateResponse;
  actions_taken: ActionTaken[];
}

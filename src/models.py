from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class PlayerWeek:
    player_id: str
    week: int
    stats: Dict[str, float]

@dataclass
class Player:
    id: str
    name: str
    position: str
    team: str
    weekly: List[PlayerWeek] = field(default_factory=list)

@dataclass
class Team:
    id: str
    name: str
    roster: List[str]  # player ids
    starting: List[str]  # player ids for each week; simplify as starters set
    wins: int = 0
    losses: int = 0
    ties: int = 0

@dataclass
class ScoringRules:
    # rules may contain numeric multipliers (stat -> points per unit)
    # and additional keys for special rules (e.g., "reception_tiers": [{"min":5,"points":1}, ...])
    rules: Dict[str, Any]

@dataclass
class SeasonData:
    players: Dict[str, Player]
    teams: Dict[str, Team]
    weeks: int
    player_weeks: List[PlayerWeek]
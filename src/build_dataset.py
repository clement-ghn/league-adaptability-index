"""
Build the transfer dataset from API-Football.

Steps (resumable: every response is cached in data/raw/, so a re-run only fetches what is missing):
1. Team lists for the five leagues (season 2023).
2. Transfers of every team (one call per team, full history).
3. Keep permanent transfers between two of the five leagues in the chosen windows.
4. Player season stats before and after each transfer (club games only).
5. Write data/transfers_dataset.csv from the cache.

Each run stops before exhausting the daily quota (100 requests on the free plan).
Run it again on the next day to continue.
"""

import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from api_football import ApiFootball, PROJECT_DIR

OUTPUT_PATH = PROJECT_DIR / "data" / "transfers_dataset.csv"

LEAGUES = {
    'Premier League': 39,
    'La Liga': 140,
    'Serie A': 135,
    'Bundesliga': 78,
    'Ligue 1': 61,
}
TEAM_LIST_SEASON = 2023

# Transfer windows used for the dataset: (first day, last day). A transfer in year Y
# compares season Y-1 (before) with season Y (after). Start with summer 2023; add windows later.
TRANSFER_WINDOWS = [
    (date(2023, 6, 1), date(2023, 9, 30)),
]

# Minimum minutes before and after the move for a player to be kept (about 10 full matches).
MIN_MINUTES = 900

# API position names -> positions used by the model (the API does not distinguish full-backs or wingers).
POSITION_MAP = {
    'Goalkeeper': 'Goalkeeper',
    'Defender': 'Defender',
    'Midfielder': 'Midfielder',
    'Attacker': 'Forward',
}


def fetch_team_league(api: ApiFootball) -> Dict[int, str]:
    """Map team id -> league name for the five leagues."""
    team_league = {}
    for league_name, league_id in LEAGUES.items():
        body = api.get('teams', league=league_id, season=TEAM_LIST_SEASON)
        for item in body['response']:
            team_league[item['team']['id']] = league_name
    return team_league


def collect_transfers(api: ApiFootball, team_league: Dict[int, str]) -> List[Dict]:
    """Permanent inter-league transfers in the chosen windows, read from the cache only."""
    moves = {}
    for team_id in team_league:
        body = api.cached('transfers', team=team_id)
        if body is None:
            continue
        for entry in body['response']:
            player = entry['player']
            for t in entry['transfers']:
                out_id = t['teams']['out']['id']
                in_id = t['teams']['in']['id']
                if out_id not in team_league or in_id not in team_league:
                    continue
                if team_league[out_id] == team_league[in_id]:
                    continue
                if 'loan' in (t['type'] or '').lower():
                    continue
                moved_on = date.fromisoformat(t['date'])
                if not any(start <= moved_on <= end for start, end in TRANSFER_WINDOWS):
                    continue
                moves[(player['id'], t['date'])] = {
                    'player_id': player['id'],
                    'player_name': player['name'],
                    'transfer_date': t['date'],
                    'transfer_type': t['type'],
                    'from_club': t['teams']['out']['name'],
                    'to_club': t['teams']['in']['name'],
                    'from_league': team_league[out_id],
                    'to_league': team_league[in_id],
                }
    return sorted(moves.values(), key=lambda m: m['transfer_date'])


def season_for(transfer_date: str) -> int:
    """Season the transfer leads into: a transfer in 2023 -> season 2023 (the 2023-24 season)."""
    return int(transfer_date[:4])


def summarise_season(response: List[Dict]) -> Dict:
    """Sum a player's club stats for one season. National-team games are excluded."""
    if not response:
        return {'minutes': 0, 'goals': 0, 'assists': 0, 'position': None, 'birth': None}

    player = response[0]['player']
    nationality = player.get('nationality')
    minutes = goals = assists = 0
    position = None
    for entry in response[0]['statistics']:
        if entry['team']['name'] == nationality:  # national team games (World Cup, Nations League, friendlies)
            continue
        minutes += entry['games']['minutes'] or 0
        goals += entry['goals']['total'] or 0
        assists += entry['goals']['assists'] or 0
        if position is None and entry['games']['position']:
            position = entry['games']['position']

    return {
        'minutes': minutes,
        'goals': goals,
        'assists': assists,
        'position': position,
        'birth': (player.get('birth') or {}).get('date'),
    }


def age_at(birth: Optional[str], on: str) -> Optional[float]:
    if not birth:
        return None
    return round((date.fromisoformat(on) - date.fromisoformat(birth)).days / 365.25, 1)


def contributions_per_90(goals: float, assists: float, minutes: float) -> float:
    return (goals + assists) / minutes * 90 if minutes > 0 else 0.0


def success_label(pre: Dict, post: Dict) -> int:
    """
    Default rule (version 0, to be refined):
    - the player keeps at least half of his minutes after the move, and
    - if he produced goals or assists before the move, his goal contributions per 90 minutes
      after the move are at least 75% of the value before.
    """
    if post['minutes'] < 0.5 * pre['minutes']:
        return 0
    pre_rate = contributions_per_90(pre['goals'], pre['assists'], pre['minutes'])
    post_rate = contributions_per_90(post['goals'], post['assists'], post['minutes'])
    if pre_rate > 0 and post_rate < 0.75 * pre_rate:
        return 0
    return 1


def build_rows(api: ApiFootball, moves: List[Dict]) -> List[Dict]:
    """Attach pre/post season stats and the label. Moves whose stats are not cached yet are skipped."""
    rows = []
    for move in moves:
        post_season = season_for(move['transfer_date'])
        pre_season = post_season - 1
        pre_body = api.cached('players', id=move['player_id'], season=pre_season)
        post_body = api.cached('players', id=move['player_id'], season=post_season)
        if pre_body is None or post_body is None:
            continue
        pre = summarise_season(pre_body['response'])
        post = summarise_season(post_body['response'])
        if pre['position'] not in POSITION_MAP:
            continue
        if pre['minutes'] < MIN_MINUTES or post['minutes'] < MIN_MINUTES:
            continue
        rows.append({
            'player_id': move['player_id'],
            'player_name': move['player_name'],
            'transfer_date': move['transfer_date'],
            'transfer_type': move['transfer_type'],
            'from_club': move['from_club'],
            'to_club': move['to_club'],
            'from_league': move['from_league'],
            'to_league': move['to_league'],
            'age': age_at(pre['birth'], move['transfer_date']),
            'position': POSITION_MAP[pre['position']],
            'pre_season': pre_season,
            'post_season': post_season,
            'pre_minutes': pre['minutes'],
            'pre_goals': pre['goals'],
            'pre_assists': pre['assists'],
            'post_minutes': post['minutes'],
            'post_goals': post['goals'],
            'post_assists': post['assists'],
            'success': success_label(pre, post),
        })
    return rows


def main():
    api = ApiFootball()
    quota = api.status()["requests"]
    # Calls allowed in this run, counting the status call above, with one call of margin.
    budget = quota["limit_day"] - quota["current"] - 1
    print(f"Quota: {quota['current']}/{quota['limit_day']} used, budget for this run: {budget}", flush=True)

    team_league = fetch_team_league(api)
    print(f"Teams: {len(team_league)}", flush=True)

    # Step 2: team transfers
    missing_teams = [t for t in team_league if api.cached('transfers', team=t) is None]
    print(f"Team transfer lists still missing: {len(missing_teams)}", flush=True)
    for team_id in missing_teams:
        if api.calls_made >= budget:
            print("Budget reached during team transfers, re-run later.", flush=True)
            break
        api.get('transfers', team=team_id)

    moves = collect_transfers(api, team_league)
    print(f"Inter-league permanent transfers in the windows: {len(moves)}", flush=True)

    # Step 4: player stats for the moves (pre and post season)
    needed = []
    for move in moves:
        post_season = season_for(move['transfer_date'])
        needed += [(move['player_id'], post_season - 1), (move['player_id'], post_season)]
    needed = list(dict.fromkeys(needed))
    missing = [(pid, s) for pid, s in needed if api.cached('players', id=pid, season=s) is None]
    print(f"Player-season stats still missing: {len(missing)} of {len(needed)}", flush=True)
    for player_id, season in missing:
        if api.calls_made >= budget:
            print("Budget reached during player stats, re-run later.", flush=True)
            break
        api.get('players', id=player_id, season=season)

    # Step 5: dataset from the cache
    rows = build_rows(api, moves)
    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    successes = int(df['success'].sum()) if len(df) else 0
    print(f"Dataset written: {OUTPUT_PATH} ({len(df)} rows, {successes} with success=1)", flush=True)
    print(f"API calls this run: {api.calls_made}", flush=True)


if __name__ == "__main__":
    main()

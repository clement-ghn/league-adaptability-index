"""
Exemple d'amélioration des critères avec données réelles
"""

def get_real_league_data():
    """
    Exemple de comment récupérer de vraies données pour les critères
    """
    
    # Données exemple qu'on pourrait récupérer de FBRef/Opta
    real_stats_2023_24 = {
        'Premier League': {
            'avg_distance_per_game': 114.2,  # km parcourus moyenne
            'avg_sprints_per_game': 47.3,    # sprints > 25km/h
            'avg_duels_per_game': 29.1,      # duels physiques
            'pass_completion_rate': 82.4,     # % passes réussies
            'ppda': 8.7,                      # passes per defensive action
            'avg_goals_per_game': 2.84,       # buts par match
            'fouls_per_game': 21.8           # fautes par match
        },
        'La Liga': {
            'avg_distance_per_game': 108.9,
            'avg_sprints_per_game': 39.2,
            'avg_duels_per_game': 24.6,
            'pass_completion_rate': 86.7,
            'ppda': 11.2,
            'avg_goals_per_game': 2.41,
            'fouls_per_game': 19.3
        },
        'Serie A': {
            'avg_distance_per_game': 110.1,
            'avg_sprints_per_game': 41.8,
            'avg_duels_per_game': 26.3,
            'pass_completion_rate': 84.9,
            'ppda': 10.1,
            'avg_goals_per_game': 2.67,
            'fouls_per_game': 20.1
        }
        # etc.
    }
    
    return real_stats_2023_24

def calculate_normalized_criteria(stats_dict):
    """
    Convertit les vraies stats en critères normalisés 0-1
    """
    
    # Trouve les valeurs max pour normalisation
    max_distance = max([s['avg_distance_per_game'] for s in stats_dict.values()])
    max_sprints = max([s['avg_sprints_per_game'] for s in stats_dict.values()])
    max_duels = max([s['avg_duels_per_game'] for s in stats_dict.values()])
    max_pass_rate = max([s['pass_completion_rate'] for s in stats_dict.values()])
    min_ppda = min([s['ppda'] for s in stats_dict.values()])
    
    normalized = {}
    
    for league, stats in stats_dict.items():
        normalized[league] = {
            # Intensité = distance + rythme général
            'intensity': (stats['avg_distance_per_game'] / max_distance) * 0.7 + 
                        (stats['avg_sprints_per_game'] / max_sprints) * 0.3,
            
            # Rythme = sprints + pressing
            'pace': (stats['avg_sprints_per_game'] / max_sprints) * 0.6 + 
                   (min_ppda / stats['ppda']) * 0.4,
            
            # Technique = taux de passes + goals/game
            'technical': (stats['pass_completion_rate'] / max_pass_rate) * 0.8 + 
                        (stats['avg_goals_per_game'] / 3.0) * 0.2,
            
            # Physicalité = duels + fautes
            'physicality': stats['avg_duels_per_game'] / max_duels,
            
            # Pressing = inverse de PPDA (moins de passes = plus de pressing)
            'pressing': min_ppda / stats['ppda']
        }
    
    return normalized

# Exemple d'utilisation
if __name__ == "__main__":
    real_data = get_real_league_data()
    normalized_criteria = calculate_normalized_criteria(real_data)
    
    print("Critères basés sur vraies données:")
    for league, criteria in normalized_criteria.items():
        print(f"\n{league}:")
        for criterion, value in criteria.items():
            print(f"  {criterion}: {value:.3f}")
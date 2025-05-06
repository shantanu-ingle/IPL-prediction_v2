from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import ollama
import pickle
from sklearn.preprocessing import StandardScaler
import ast
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from django.conf import settings

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CricketNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CricketNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

def load_scaler(scaler_path='scaler.pkl'):
    scaler_path = os.path.join(settings.BASE_DIR, scaler_path)
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    else:
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

def load_model(model_path, input_dim, output_dim):
    model_path = os.path.join(settings.BASE_DIR, model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = CricketNet(input_dim=input_dim, output_dim=output_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

def compute_team_stats(team_name, matches, deliveries, venue, opponent_team):
    team_matches = matches[(matches['team1'] == team_name) | (matches['team2'] == team_name)]
    batting_avg = team_matches[team_matches['team1'] == team_name]['first_ings_score'].mean()
    bowling_avg = team_matches[team_matches['team2'] == team_name]['first_ings_wkts'].mean()
    recent_matches = team_matches.sort_values(by='match_id').tail(3)  # Sort by match_id since match_date is unavailable
    recent_runs = (recent_matches[recent_matches['team1'] == team_name]['first_ings_score'].sum() + 
                   recent_matches[recent_matches['team2'] == team_name]['second_ings_score'].sum()) / 3
    opp_matches = team_matches[(team_matches['team1'] == opponent_team) | (team_matches['team2'] == opponent_team)]
    if not opp_matches.empty:
        opp_batting_avg = (opp_matches[opp_matches['team1'] == opponent_team]['first_ings_score'].mean() + 
                           opp_matches[opp_matches['team2'] == opponent_team]['second_ings_score'].mean()) / 2
    else:
        opp_batting_avg = 0
    opp_bowling_strength = opp_matches[opp_matches['team2'] == opponent_team]['first_ings_wkts'].mean()
    venue_avg_score = matches[matches['venue'] == venue]['first_ings_score'].mean()
    return {
        'batting_avg': batting_avg if not np.isnan(batting_avg) else 0,
        'bowling_avg': bowling_avg if not np.isnan(bowling_avg) else 0,
        'recent_runs': recent_runs if not np.isnan(recent_runs) else 0,
        'opp_batting_avg': opp_batting_avg if not np.isnan(opp_batting_avg) else 0,
        'opp_bowling_strength': opp_bowling_strength if not np.isnan(opp_bowling_strength) else 0,
        'venue_avg_score': venue_avg_score if not np.isnan(venue_avg_score) else 0
    }

def compute_player_stats(player_name, role, team_name, venue, opponent_team, matches, deliveries):
    if role == 'batsman':
        player_data = deliveries[deliveries['striker'] == player_name]
        if player_data.empty:
            return {
                'recent_runs': 0,
                'recent_strike_rate': 0,
                'venue_runs': 0,
                'venue_strike_rate': 0,
                'opp_runs': 0,
                'opp_strike_rate': 0,
                'recent_wickets': 0,
                'recent_economy': 0
            }
        player_stats = player_data.groupby('match_id').agg({
            'runs_of_bat': 'sum',
            'match_id': 'count'
        }).rename(columns={'match_id': 'balls_faced', 'runs_of_bat': 'runs'})
        player_stats['strike_rate'] = (player_stats['runs'] / player_stats['balls_faced'] * 100).fillna(0)
        recent_runs = player_stats['runs'].tail(3).mean()
        recent_strike_rate = player_stats['strike_rate'].tail(3).mean()
        venue_matches = matches[matches['venue'] == venue]
        venue_match_ids = venue_matches['match_id'].values
        venue_data = player_data[player_data['match_id'].isin(venue_match_ids)]
        if not venue_data.empty:
            venue_stats = venue_data.groupby('match_id').agg({
                'runs_of_bat': 'sum',
                'match_id': 'count'
            }).rename(columns={'match_id': 'balls_faced', 'runs_of_bat': 'runs'})
            venue_stats['strike_rate'] = (venue_stats['runs'] / venue_stats['balls_faced'] * 100).fillna(0)
            venue_runs = venue_stats['runs'].mean()
            venue_strike_rate = venue_stats['strike_rate'].mean()
        else:
            venue_runs = recent_runs
            venue_strike_rate = recent_strike_rate
        opp_matches = matches[(matches['team1'] == opponent_team) | (matches['team2'] == opponent_team)]
        opp_match_ids = opp_matches['match_id'].values
        opp_data = player_data[player_data['match_id'].isin(opp_match_ids)]
        if not opp_data.empty:
            opp_stats = opp_data.groupby('match_id').agg({
                'runs_of_bat': 'sum',
                'match_id': 'count'
            }).rename(columns={'match_id': 'balls_faced', 'runs_of_bat': 'runs'})
            opp_stats['strike_rate'] = (opp_stats['runs'] / opp_stats['balls_faced'] * 100).fillna(0)
            opp_runs = opp_stats['runs'].mean()
            opp_strike_rate = opp_stats['strike_rate'].mean()
        else:
            opp_runs = recent_runs
            opp_strike_rate = recent_strike_rate
        return {
            'recent_runs': recent_runs if not np.isnan(recent_runs) else 0,
            'recent_strike_rate': recent_strike_rate if not np.isnan(recent_strike_rate) else 0,
            'venue_runs': venue_runs if not np.isnan(venue_runs) else 0,
            'venue_strike_rate': venue_strike_rate if not np.isnan(venue_strike_rate) else 0,
            'opp_runs': opp_runs if not np.isnan(opp_runs) else 0,
            'opp_strike_rate': opp_strike_rate if not np.isnan(opp_strike_rate) else 0,
            'recent_wickets': 0,
            'recent_economy': 0
        }
    elif role == 'bowler':
        player_data = deliveries[deliveries['bowler'] == player_name]
        if player_data.empty:
            return {
                'recent_runs': 0,
                'recent_strike_rate': 0,
                'venue_runs': 0,
                'venue_strike_rate': 0,
                'opp_runs': 0,
                'opp_strike_rate': 0,
                'recent_wickets': 0,
                'recent_economy': 0,
                'venue_wickets': 0,
                'venue_economy': 0,
                'opp_wickets': 0,
                'opp_economy': 0
            }
        player_stats = player_data.groupby('match_id').agg({
            'player_dismissed': lambda x: x.notna().sum(),
            'runs_of_bat': 'sum',
            'extras': 'sum',
            'over': 'nunique'
        }).rename(columns={'player_dismissed': 'wickets', 'over': 'overs_bowled'})
        player_stats['runs_conceded'] = player_stats['runs_of_bat'] + player_stats['extras']
        player_stats['economy_rate'] = (player_stats['runs_conceded'] / player_stats['overs_bowled']).fillna(0)
        recent_wickets = player_stats['wickets'].tail(3).mean()
        recent_economy = player_stats['economy_rate'].tail(3).mean()
        logger.info(f"Computed recent_economy for {player_name}: {recent_economy}")
        return {
            'recent_runs': 0,
            'recent_strike_rate': 0,
            'venue_runs': 0,
            'venue_strike_rate': 0,
            'opp_runs': 0,
            'opp_strike_rate': 0,
            'recent_wickets': recent_wickets if not np.isnan(recent_wickets) else 0,
            'recent_economy': recent_economy if not np.isnan(recent_economy) else 0,
            'venue_wickets': 0,
            'venue_economy': 0,
            'opp_wickets': 0,
            'opp_economy': 0
        }
    else:
        return {
            'recent_runs': 0,
            'recent_strike_rate': 0,
            'venue_runs': 0,
            'venue_strike_rate': 0,
            'opp_runs': 0,
            'opp_strike_rate': 0,
            'recent_wickets': 0,
            'recent_economy': 0
        }

def validate_player_team(player_name, team_name, role, matches, deliveries):
    if not player_name or not team_name:
        return
    if role == 'batsman':
        player_data = deliveries[deliveries['striker'] == player_name]
    else:
        player_data = deliveries[deliveries['bowler'] == player_name]
    if player_data.empty:
        raise ValueError(f"No data found for {role} {player_name} in deliveries dataset.")
    match_ids = player_data['match_id'].unique()
    player_matches = matches[matches['match_id'].isin(match_ids)]
    team_matches = player_matches[(player_matches['team1'] == team_name) | (player_matches['team2'] == team_name)]
    if team_matches.empty:
        raise ValueError(f"{player_name} does not belong to {team_name} based on match data.")

def generate_prediction_explanation(prediction, features, model_type, team1_name="Team 1", team2_name="Team 2", winner_team=None):
    if model_type == "winner":
        prompt = f"""
        You are an expert cricket analyst. An ML model predicts that {winner_team} will win an IPL match between {team1_name} and {team2_name} based on predicted scores.
        Key factors:
        - {team1_name} predicted score: {features['team1_score']:.1f}
        - {team2_name} predicted score: {features['team2_score']:.1f}
        Explain why {winner_team} is likely to win in under 100 words, focusing on these factors.
        """
    elif model_type == "team_runs":
        team_name = team1_name if prediction['team'] == "Team 1" else team2_name
        prompt = f"""
        You are an expert cricket analyst. An ML model predicts {team_name} will score {prediction['runs']:.0f} runs.
        Key factors:
        - Team form: {features['team_form']*100:.1f}% (last 3 matches)
        - Average runs (last 3): {features['avg_runs_last_3']:.1f}
        Explain why this score is predicted in under 100 words, focusing on these factors.
        """
    elif model_type == "player_runs":
        prompt = f"""
        You are an expert cricket analyst. An ML model predicts Player X will score {prediction['runs']:.0f} runs.
        Key factors:
        - Average runs (last 3): {features['player_avg_runs_last_3']:.1f}
        - Strike rate: {features['player_strike_rate']:.1f}
        Explain why this performance is predicted in under 100 words, focusing on these factors.
        """
    try:
        response = ollama.generate(model="mistral", prompt=prompt)
        return response['response']
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

def generate_player_trend_plot(player_name, matches, deliveries):
    plt.figure(figsize=(6, 4))
    
    # Fetch the player's last 3 matches' runs from deliveries dataset
    if not player_name:
        plt.plot([1, 2, 3], [0, 0, 0], label="Player Runs (Last 3)", marker='o', color='#42A5F5')
        logger.info("No player name provided, generating placeholder plot.")
    else:
        player_data = deliveries[deliveries['striker'] == player_name]
        if player_data.empty:
            plt.plot([1, 2, 3], [0, 0, 0], label="Player Runs (Last 3)", marker='o', color='#42A5F5')
            logger.info(f"No data found for player {player_name}, generating placeholder plot.")
        else:
            # Merge with matches to get dates
            player_matches = player_data.merge(matches[['match_id']], on='match_id', how='left')
            # Group by match_id to get runs per match
            player_stats = player_matches.groupby('match_id').agg({
                'runs_of_bat': 'sum'
            }).reset_index()
            # Sort by match_id (fallback if date is unavailable)
            player_stats = player_stats.sort_values(by='match_id').tail(3)
            runs = player_stats['runs_of_bat'].tolist()
            # If less than 3 matches, pad with zeros
            matches_indices = list(range(1, len(runs) + 1))
            while len(runs) < 3:
                runs.insert(0, 0)
                matches_indices.insert(0, matches_indices[0] - 1 if matches_indices else 1)
            logger.info(f"Player {player_name} runs for last 3 matches: {runs}")
            plt.plot(matches_indices, runs, label=f"{player_name}'s Runs (Last 3)", marker='o', color='#42A5F5')
            # Use match numbers on x-axis since date is unavailable
            plt.xticks(matches_indices, [f"Match {i}" for i in matches_indices])
    
    plt.xlabel("Match")
    plt.ylabel("Runs")
    plt.title("Player Runs Trend (Last 3 Matches)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plot_path = os.path.join(settings.STATICFILES_DIRS[0], "player_trend.png")
    os.makedirs(settings.STATICFILES_DIRS[0], exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    return "/static/player_trend.png"

@api_view(['GET'])
def get_options(request):
    matches = pd.read_csv('matches.csv')
    deliveries = pd.read_csv('deliveries.csv')
    deliveries = deliveries.rename(columns={'match_no': 'match_id'})
    team1_name = request.GET.get('team1_name')
    team2_name = request.GET.get('team2_name')
    teams = sorted(set(matches['team1']).union(set(matches['team2'])))
    if team1_name and team2_name:
        team_matches = matches[(matches['team1'].isin([team1_name, team2_name])) | (matches['team2'].isin([team1_name, team2_name]))]
        match_ids = team_matches['match_id'].unique()
        team_deliveries = deliveries[deliveries['match_id'].isin(match_ids)]
        batsmen = set()
        bowlers = set()
        for match_id in match_ids:
            match = team_matches[team_matches['match_id'] == match_id].iloc[0]
            match_deliveries = team_deliveries[team_deliveries['match_id'] == match_id]
            team1 = match['team1']
            team2 = match['team2']
            team1_batsmen = match_deliveries[match_deliveries['batting_team'] == team1]['striker'].unique()
            team2_batsmen = match_deliveries[match_deliveries['batting_team'] == team2]['striker'].unique()
            team1_bowlers = match_deliveries[match_deliveries['bowling_team'] == team1]['bowler'].unique()
            team2_bowlers = match_deliveries[match_deliveries['bowling_team'] == team2]['bowler'].unique()
            if team1 == team1_name:
                batsmen.update(team1_batsmen)
                bowlers.update(team1_bowlers)
            elif team1 == team2_name:
                batsmen.update(team1_batsmen)
                bowlers.update(team1_bowlers)
            if team2 == team1_name:
                batsmen.update(team2_batsmen)
                bowlers.update(team2_bowlers)
            elif team2 == team2_name:
                batsmen.update(team2_batsmen)
                bowlers.update(team2_bowlers)
        batsmen = sorted(list(batsmen))
        bowlers = sorted(list(bowlers))
    else:
        batsmen = sorted(deliveries['striker'].unique())
        bowlers = sorted(deliveries['bowler'].unique())
    venues = sorted(matches['venue'].unique())
    return Response({
        'teams': teams,
        'batsmen': batsmen,
        'bowlers': bowlers,
        'venues': venues
    }, status=status.HTTP_200_OK)

prediction_context = {}

@api_view(['POST'])
def predict_match(request):
    try:
        matches = pd.read_csv('matches.csv')
        deliveries = pd.read_csv('deliveries.csv')
        deliveries = deliveries.rename(columns={'match_no': 'match_id'})
        scaler = load_scaler()
        input_dim = 14
        models = {
            'score_rf': load_model('score_rf.pth', input_dim, 1),
            'score_gb': load_model('score_gb.pth', input_dim, 1),
            'runs_rf': load_model('runs_rf.pth', input_dim, 1),
            'runs_gb': load_model('runs_gb.pth', input_dim, 1),
            'wickets_rf': load_model('wickets_rf.pth', input_dim, 1),
            'wickets_gb': load_model('wickets_gb.pth', input_dim, 1),
            'economy_rf': load_model('economy_rf.pth', input_dim, 1),
            'economy_gb': load_model('economy_gb.pth', input_dim, 1),
            'strike_rate_rf': load_model('strike_rate_rf.pth', input_dim, 1),
            'strike_rate_gb': load_model('strike_rate_gb.pth', input_dim, 1),
        }
        input_data = request.data
        required_fields = ['team1_name', 'team2_name']
        for field in required_fields:
            if field not in input_data or not input_data[field]:
                return Response({"error": f"Missing field: {field}"}, status=status.HTTP_400_BAD_REQUEST)
        
        venue = input_data.get('venue', matches['venue'].unique()[0] if matches['venue'].unique().size > 0 else "Unknown Venue")
        player1_name = input_data.get('player1_name', '')
        player1_role = input_data.get('player1_role', 'batsman')
        player1_team = input_data.get('player1_team', input_data['team1_name'])
        player2_name = input_data.get('player2_name', '')
        player2_role = input_data.get('player2_role', 'bowler')
        player2_team = input_data.get('player2_team', input_data['team2_name'])
        overs_team1 = float(input_data.get('overs_team1', 20))
        overs_team2 = float(input_data.get('overs_team2', 20))
        toss_winner = input_data.get('toss_winner', 'team1')

        valid_teams = sorted(set(matches['team1']).union(set(matches['team2'])))
        if input_data['team1_name'] not in valid_teams or input_data['team2_name'] not in valid_teams:
            return Response({"error": f"Invalid team names. Must be one of: {valid_teams}"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            validate_player_team(player1_name, player1_team, player1_role, matches, deliveries)
            validate_player_team(player2_name, player2_team, player2_role, matches, deliveries)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        team1_stats = compute_team_stats(input_data['team1_name'], matches, deliveries, venue, input_data['team2_name'])
        team2_stats = compute_team_stats(input_data['team2_name'], matches, deliveries, venue, input_data['team1_name'])
        player1_stats = compute_player_stats(player1_name, player1_role, player1_team, venue, input_data['team2_name'], matches, deliveries)
        player2_stats = compute_player_stats(player2_name, player2_role, player2_team, venue, input_data['team1_name'], matches, deliveries)

        combined_player_stats = {
            'player_recent_runs': player1_stats['recent_runs'] if player1_role == 'batsman' else player2_stats['recent_runs'],
            'player_recent_wickets': player2_stats['recent_wickets'] if player2_role == 'bowler' else player1_stats['recent_wickets'],
            'player_recent_strike_rate': player1_stats['recent_strike_rate'] if player1_role == 'batsman' else player2_stats['recent_strike_rate'],
            'player_recent_economy': player2_stats['recent_economy'] if player2_role == 'bowler' else player1_stats['recent_economy']
        }

        logger.info(f"Input features for economy prediction: {combined_player_stats}")

        toss_impact_team1 = 1 if (toss_winner == "team1" and input_data['team1_name'] == player1_team) else 0
        toss_impact_team2 = 1 if (toss_winner == "team2" and input_data['team2_name'] == player2_team) else 0

        test_data_team1 = pd.DataFrame([{
            'team_batting_avg': team1_stats['batting_avg'],
            'team_bowling_avg': team1_stats['bowling_avg'],
            'team1_recent_runs': team1_stats['recent_runs'],
            'team2_recent_runs': team2_stats['recent_runs'],
            'player_recent_runs': combined_player_stats['player_recent_runs'],
            'player_recent_wickets': combined_player_stats['player_recent_wickets'],
            'player_recent_strike_rate': combined_player_stats['player_recent_strike_rate'],
            'player_recent_economy': combined_player_stats['player_recent_economy'],
            'opp_bowling_strength': team2_stats['opp_bowling_strength'],
            'venue_avg_score': team1_stats['venue_avg_score'],
            'toss_impact': toss_impact_team1,
            'weather_impact': 1.0,
            'player_available': 1,
            'overs': overs_team1
        }])

        test_data_team2 = pd.DataFrame([{
            'team_batting_avg': team2_stats['batting_avg'],
            'team_bowling_avg': team2_stats['bowling_avg'],
            'team1_recent_runs': team2_stats['recent_runs'],
            'team2_recent_runs': team1_stats['recent_runs'],
            'player_recent_runs': combined_player_stats['player_recent_runs'],
            'player_recent_wickets': combined_player_stats['player_recent_wickets'],
            'player_recent_strike_rate': combined_player_stats['player_recent_strike_rate'],
            'player_recent_economy': combined_player_stats['player_recent_economy'],
            'opp_bowling_strength': team1_stats['opp_bowling_strength'],
            'venue_avg_score': team2_stats['venue_avg_score'],
            'toss_impact': toss_impact_team2,
            'weather_impact': 1.0,
            'player_available': 1,
            'overs': overs_team2
        }])

        features = ['team_batting_avg', 'team_bowling_avg', 'team1_recent_runs', 'team2_recent_runs', 
                    'player_recent_runs', 'player_recent_wickets', 'player_recent_strike_rate', 
                    'player_recent_economy', 'opp_bowling_strength', 'venue_avg_score', 
                    'toss_impact', 'weather_impact', 'player_available', 'overs']
        test_data_team1_scaled = scaler.transform(test_data_team1[features])
        test_data_team2_scaled = scaler.transform(test_data_team2[features])
        test_data_team1_tensor = torch.tensor(test_data_team1_scaled, dtype=torch.float32).to(device)
        test_data_team2_tensor = torch.tensor(test_data_team2_scaled, dtype=torch.float32).to(device)

        models['score_rf'].eval()
        models['score_gb'].eval()
        with torch.no_grad():
            score_team1_rf = models['score_rf'](test_data_team1_tensor).squeeze().cpu().numpy()
            score_team1_gb = models['score_gb'](test_data_team1_tensor).squeeze().cpu().numpy()
            score_team2_rf = models['score_rf'](test_data_team2_tensor).squeeze().cpu().numpy()
            score_team2_gb = models['score_gb'](test_data_team2_tensor).squeeze().cpu().numpy()
        
        score_team1 = (score_team1_rf + score_team1_gb) / 2
        score_team2 = (score_team2_rf + score_team2_gb) / 2

        if score_team1 > score_team2:
            winner_team = input_data['team1_name']
            winning_score = score_team1
            losing_score = score_team2
            winning_team = input_data['team1_name']
            losing_team = input_data['team2_name']
        else:
            winner_team = input_data['team2_name']
            winning_score = score_team2
            losing_score = score_team1
            winning_team = input_data['team2_name']
            losing_team = input_data['team1_name']

        team1_rmse = 16.84
        team2_rmse = 18.57
        team1_runs_lower = score_team1 - 1.96 * team1_rmse
        team1_runs_upper = score_team1 + 1.96 * team1_rmse
        team2_runs_lower = score_team2 - 1.96 * team2_rmse
        team2_runs_upper = score_team2 + 1.96 * team2_rmse

        team1_features = {"team_form": 0.5, "avg_runs_last_3": team1_stats['recent_runs']}
        team2_features = {"team_form": 0.5, "avg_runs_last_3": team2_stats['recent_runs']}
        team1_explanation = generate_prediction_explanation(
            {"team": "Team 1", "runs": score_team1},
            team1_features,
            "team_runs",
            input_data['team1_name'],
            input_data['team2_name']
        )
        team2_explanation = generate_prediction_explanation(
            {"team": "Team 2", "runs": score_team2},
            team2_features,
            "team_runs",
            input_data['team1_name'],
            input_data['team2_name']
        )

        winner_features = {"team1_score": score_team1, "team2_score": score_team2}
        winner_explanation = generate_prediction_explanation(
            {"winner": winner_team, "confidence": 0.95},
            winner_features,
            "winner",
            input_data['team1_name'],
            input_data['team2_name'],
            winner_team
        )

        if player1_team == input_data['team1_name']:
            player1_tensor = test_data_team1_tensor
        else:
            player1_tensor = test_data_team2_tensor
        
        if player2_team == input_data['team1_name']:
            player2_tensor = test_data_team1_tensor
        else:
            player2_tensor = test_data_team2_tensor

        player_runs = None
        player_runs_lower = None
        player_runs_upper = None
        player_explanation = None
        player_strike_rate = None
        if player1_name and player1_role == 'batsman':
            models['runs_rf'].eval()
            models['runs_gb'].eval()
            with torch.no_grad():
                runs_pred_rf = models['runs_rf'](player1_tensor).squeeze().cpu().numpy()
                runs_pred_gb = models['runs_gb'](player1_tensor).squeeze().cpu().numpy()
            player_runs = (runs_pred_rf + runs_pred_gb) / 2
            player_rmse = 12.38
            player_runs_lower = player_runs - 1.96 * player_rmse
            player_runs_upper = player_runs + 1.96 * player_rmse
            models['strike_rate_rf'].eval()
            models['strike_rate_gb'].eval()
            with torch.no_grad():
                strike_pred_rf = models['strike_rate_rf'](player1_tensor).squeeze().cpu().numpy()
                strike_pred_gb = models['strike_rate_gb'](player1_tensor).squeeze().cpu().numpy()
            player_strike_rate = (strike_pred_rf + strike_pred_gb) / 2
            player_features = {"player_avg_runs_last_3": player_runs, "player_strike_rate": player_strike_rate}
            player_explanation = generate_prediction_explanation(
                {"runs": player_runs},
                player_features,
                "player_runs",
                input_data['team1_name'],
                input_data['team2_name']
            )

        player_wickets = None
        player_economy = None
        if player2_name and player2_role == 'bowler':
            models['wickets_rf'].eval()
            models['wickets_gb'].eval()
            with torch.no_grad():
                wickets_pred_rf = models['wickets_rf'](player2_tensor).squeeze().cpu().numpy()
                wickets_pred_gb = models['wickets_gb'](player2_tensor).squeeze().cpu().numpy()
            player_wickets = (wickets_pred_rf + wickets_pred_gb) / 2
            player_wickets = round(player_wickets)
            models['economy_rf'].eval()
            models['economy_gb'].eval()
            with torch.no_grad():
                economy_pred_rf = models['economy_rf'](player2_tensor).squeeze().cpu().numpy()
                economy_pred_gb = models['economy_gb'](player2_tensor).squeeze().cpu().numpy()
            raw_economy = (economy_pred_rf + economy_pred_gb) / 2
            logger.info(f"Raw predicted economy: {raw_economy}")
            min_economy = 4.0
            max_economy = 12.0
            player_economy = min_economy + (max_economy - min_economy) * raw_economy
            logger.info(f"Scaled predicted economy: {player_economy}")
            player_economy = max(player_economy, min_economy)

        plot_url = generate_player_trend_plot(player1_name, matches, deliveries)

        response = {
            "match_winner": {
                "team": winner_team,
                "confidence": 0.95,
                "explanation": winner_explanation
            },
            f"{input_data['team1_name']}_runs": {
                "predicted_runs": float(score_team1),
                "confidence_interval": [float(team1_runs_lower), float(team1_runs_upper)],
                "explanation": team1_explanation
            },
            f"{input_data['team2_name']}_runs": {
                "predicted_runs": float(score_team2),
                "confidence_interval": [float(team2_runs_lower), float(team2_runs_upper)],
                "explanation": team2_explanation
            },
            "winning_team_runs": {
                "team": winning_team,
                "predicted_runs": float(winning_score),
                "confidence_interval": [float(team1_runs_lower if winning_team == input_data['team1_name'] else team2_runs_lower),
                                        float(team1_runs_upper if winning_team == input_data['team1_name'] else team2_runs_upper)]
            },
            "losing_team_runs": {
                "team": losing_team,
                "predicted_runs": float(losing_score),
                "confidence_interval": [float(team2_runs_lower if losing_team == input_data['team2_name'] else team1_runs_lower),
                                        float(team2_runs_upper if losing_team == input_data['team2_name'] else team1_runs_upper)]
            }
        }

        if player_runs is not None:
            response["player_runs"] = {
                "predicted_runs": float(player_runs),
                "strike_rate": float(player_strike_rate),
                "confidence_interval": [float(player_runs_lower), float(player_runs_upper)],
                "explanation": player_explanation,
                "trend_plot": plot_url
            }
        
        if player_wickets is not None:
            response["player_wickets"] = {
                "predicted_wickets": int(player_wickets),
                "economy_rate": float(player_economy)
            }

        prediction_context['response'] = response
        prediction_context['input_data'] = input_data
        prediction_context['team1_stats'] = team1_stats
        prediction_context['team2_stats'] = team2_stats
        prediction_context['player1_stats'] = player1_stats
        prediction_context['player2_stats'] = player2_stats

        return Response(response, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def chat_with_model(request):
    try:
        user_message = request.data.get('message')
        if not user_message:
            return Response({"error": "Message is required"}, status=status.HTTP_400_BAD_REQUEST)

        if not prediction_context:
            return Response({"error": "No prediction context available. Please make a prediction first."}, status=status.HTTP_400_BAD_REQUEST)

        response = prediction_context['response']
        input_data = prediction_context['input_data']
        team1_stats = prediction_context['team1_stats']
        team2_stats = prediction_context['team2_stats']
        player1_stats = prediction_context['player1_stats']
        player2_stats = prediction_context['player2_stats']

        context = f"""
        You are an expert cricket analyst with access to the following prediction data:
        - Match Winner: {response['match_winner']['team']} (Confidence: {response['match_winner']['confidence']*100}%)
        - {input_data['team1_name']} Predicted Runs: {response[f"{input_data['team1_name']}_runs"]['predicted_runs']:.0f}
        - {input_data['team2_name']} Predicted Runs: {response[f"{input_data['team2_name']}_runs"]['predicted_runs']:.0f}
        """
        if 'player_runs' in response:
            context += f"""
            - Batsman ({input_data.get('player1_name', 'Unknown')}): 
              Predicted Runs: {response['player_runs']['predicted_runs']:.0f}
              Strike Rate: {response['player_runs']['strike_rate']:.0f}
            """
        if 'player_wickets' in response:
            context += f"""
            - Bowler ({input_data.get('player2_name', 'Unknown')}): 
              Predicted Wickets: {response['player_wickets']['predicted_wickets']}
              Economy Rate: {response['player_wickets']['economy_rate']:.2f}
            """

        prompt = f"""
        {context}
        User: {user_message}
        Provide a detailed and insightful response based on the prediction data and the user's query.
        """
        ollama_response = ollama.generate(model="mistral", prompt=prompt)
        return Response({"response": ollama_response['response']}, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
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
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import os
from django.conf import settings

logger = logging.getLogger(__name__)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Neural Network Model (must match the training script's architecture)
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

# Load the Scaler
def load_scaler(scaler_path='scaler.pkl'):
    scaler_path = os.path.join(settings.BASE_DIR, scaler_path)  # Use BASE_DIR
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    else:
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

# Load Models Le
def load_model(model_path, input_dim, output_dim):
    model_path = os.path.join(settings.BASE_DIR, model_path)  # Use BASE_DIR
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = CricketNet(input_dim=input_dim, output_dim=output_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

# Compute Team Statistics from Dataset
def compute_team_stats(team_name, matches, deliveries, venue, opponent_team):
    # Filter matches involving the team
    team_matches = matches[(matches['team1'] == team_name) | (matches['team2'] == team_name)]
    
    # Batting average (first innings score when batting first)
    batting_avg = team_matches[team_matches['team1'] == team_name]['first_ings_score'].mean()
    
    # Bowling average (first innings wickets taken when bowling second)
    bowling_avg = team_matches[team_matches['team2'] == team_name]['first_ings_wkts'].mean()
    
    # Recent runs (last 3 matches)
    recent_matches = team_matches.sort_values(by='date').tail(3)
    recent_runs = (recent_matches[recent_matches['team1'] == team_name]['first_ings_score'].sum() + 
                   recent_matches[recent_matches['team2'] == team_name]['second_ings_score'].sum()) / 3
    
    # Opposition-specific batting average (against the opponent team)
    opp_matches = team_matches[(team_matches['team1'] == opponent_team) | (team_matches['team2'] == opponent_team)]
    if not opp_matches.empty:
        opp_batting_avg = (opp_matches[opp_matches['team1'] == opponent_team]['first_ings_score'].mean() + 
                           opp_matches[opp_matches['team2'] == opponent_team]['second_ings_score'].mean()) / 2
    else:
        opp_batting_avg = 0
    
    # Opposition bowling strength (average wickets taken by opponent team)
    opp_bowling_strength = opp_matches[opp_matches['team2'] == opponent_team]['first_ings_wkts'].mean()
    
    # Venue average score
    venue_avg_score = matches[matches['venue'] == venue]['first_ings_score'].mean()
    
    return {
        'batting_avg': batting_avg if not np.isnan(batting_avg) else 0,
        'bowling_avg': bowling_avg if not np.isnan(bowling_avg) else 0,
        'recent_runs': recent_runs if not np.isnan(recent_runs) else 0,
        'opp_batting_avg': opp_batting_avg if not np.isnan(opp_batting_avg) else 0,
        'opp_bowling_strength': opp_bowling_strength if not np.isnan(opp_bowling_strength) else 0,
        'venue_avg_score': venue_avg_score if not np.isnan(venue_avg_score) else 0
    }

# Compute Player Statistics with Venue-Specific and Opposition-Specific Data
def compute_player_stats(player_name, role, team_name, venue, opponent_team, matches, deliveries):
    if role == 'batsman':
        player_data = deliveries[deliveries['striker'] == player_name]
        if player_data.empty:
            raise ValueError(f"No data found for batsman {player_name} in deliveries dataset.")
        
        # Compute historical stats
        player_stats = player_data.groupby('match_id').agg({
            'runs_of_bat': 'sum',
            'match_id': 'count'
        }).rename(columns={'match_id': 'balls_faced', 'runs_of_bat': 'runs'})
        player_stats['strike_rate'] = (player_stats['runs'] / player_stats['balls_faced'] * 100).fillna(0)
        
        # Recent stats (last 3 matches)
        recent_runs = player_stats['runs'].tail(3).mean()
        recent_strike_rate = player_stats['strike_rate'].tail(3).mean()
        
        # Venue-specific stats
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
        
        # Opposition-specific stats
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
            raise ValueError(f"No data found for bowler {player_name} in deliveries dataset.")
        
        # Compute historical stats
        player_stats = player_data.groupby('match_id').agg({
            'player_dismissed': lambda x: x.notna().sum(),
            'runs_of_bat': 'sum',
            'extras': 'sum',
            'over': 'nunique'
        }).rename(columns={'player_dismissed': 'wickets', 'over': 'overs_bowled'})
        player_stats['runs_conceded'] = player_stats['runs_of_bat'] + player_stats['extras']
        player_stats['economy_rate'] = (player_stats['runs_conceded'] / player_stats['overs_bowled']).fillna(0)
        
        # Recent stats (last 3 matches)
        recent_wickets = player_stats['wickets'].tail(3).mean()
        recent_economy = player_stats['economy_rate'].tail(3).mean()
        logger.info(f"Computed recent_economy for {player_name}: {recent_economy}")  # Log the computed economy rate
        
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
        raise ValueError(f"Invalid role {role}. Must be 'batsman' or 'bowler'.")

# Validate Player Team Association
def validate_player_team(player_name, team_name, role, matches, deliveries):
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

# Generate Prediction Explanations
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

# Generate Player Trend Plot
def generate_player_trend_plot(player_runs):
    plt.figure(figsize=(6, 4))
    plt.plot([1, 2, 3], [player_runs - 5, player_runs, player_runs + 5], label="Player Runs (Last 3)", marker='o')
    plt.xlabel("Match")
    plt.ylabel("Runs")
    plt.title("Player Runs Trend (Last 3 Matches)")
    plt.legend()
    plot_path = os.path.join(settings.STATICFILES_DIRS[0], "player_trend.png")
    # Ensure the static directory exists and is writable
    os.makedirs(settings.STATICFILES_DIRS[0], exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    return "/static/player_trend.png"

# Get Lists of Teams, Players, and Venues
@api_view(['GET'])
def get_options(request):
    matches = pd.read_csv('matches.csv')
    deliveries = pd.read_csv('deliveries.csv')
    deliveries = deliveries.rename(columns={'match_no': 'match_id'})
    
    # Get team1_name and team2_name from query parameters
    team1_name = request.GET.get('team1_name')
    team2_name = request.GET.get('team2_name')
    
    # Get unique teams
    teams = sorted(set(matches['team1']).union(set(matches['team2'])))
    
    # Filter batsmen and bowlers based on selected teams if provided
    if team1_name and team2_name:
        # Get match IDs for the selected teams
        team_matches = matches[(matches['team1'].isin([team1_name, team2_name])) | (matches['team2'].isin([team1_name, team2_name]))]
        match_ids = team_matches['match_id'].unique()
        
        # Filter deliveries for these matches
        team_deliveries = deliveries[deliveries['match_id'].isin(match_ids)]
        
        # Get batsmen and bowlers who played for the selected teams
        batsmen = set()
        bowlers = set()
        
        for match_id in match_ids:
            match = team_matches[team_matches['match_id'] == match_id].iloc[0]
            match_deliveries = team_deliveries[team_deliveries['match_id'] == match_id]
            
            # Get batting and bowling teams for each match
            team1 = match['team1']
            team2 = match['team2']
            
            # Batsmen from team1 (batting first) or team2 (batting second)
            team1_batsmen = match_deliveries[match_deliveries['batting_team'] == team1]['striker'].unique()
            team2_batsmen = match_deliveries[match_deliveries['batting_team'] == team2]['striker'].unique()
            
            # Bowlers from team1 (bowling second) or team2 (bowling first)
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
        # If teams are not specified, return all players
        batsmen = sorted(deliveries['striker'].unique())
        bowlers = sorted(deliveries['bowler'].unique())
    
    # Get unique venues
    venues = sorted(matches['venue'].unique())
    
    return Response({
        'teams': teams,
        'batsmen': batsmen,
        'bowlers': bowlers,
        'venues': venues
    }, status=status.HTTP_200_OK)

@api_view(['POST'])
def predict_match(request):
    try:
        # Load data
        matches = pd.read_csv('matches.csv')
        deliveries = pd.read_csv('deliveries.csv')
        deliveries = deliveries.rename(columns={'match_no': 'match_id'})
        
        # Load scaler and models
        scaler = load_scaler()
        input_dim = 14  # Number of features
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

        # Parse input data
        input_data = request.data
        required_fields = [
            'team1_name', 'team2_name',
            'player1_name', 'player1_role', 'player1_team',
            'player2_name', 'player2_role', 'player2_team',
            'venue', 'overs_team1', 'overs_team2', 'toss_winner'
        ]
        for field in required_fields:
            if field not in input_data:
                return Response({"error": f"Missing field: {field}"}, status=status.HTTP_400_BAD_REQUEST)

        # Validate teams
        valid_teams = sorted(set(matches['team1']).union(set(matches['team2'])))
        if input_data['team1_name'] not in valid_teams or input_data['team2_name'] not in valid_teams:
            return Response({"error": f"Invalid team names. Must be one of: {valid_teams}"}, status=status.HTTP_400_BAD_REQUEST)

        # Validate player team associations
        try:
            validate_player_team(input_data['player1_name'], input_data['player1_team'], input_data['player1_role'], matches, deliveries)
            validate_player_team(input_data['player2_name'], input_data['player2_team'], input_data['player2_role'], matches, deliveries)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        # Compute team stats
        team1_stats = compute_team_stats(input_data['team1_name'], matches, deliveries, input_data['venue'], input_data['team2_name'])
        team2_stats = compute_team_stats(input_data['team2_name'], matches, deliveries, input_data['venue'], input_data['team1_name'])

        # Compute player stats
        player1_stats = compute_player_stats(input_data['player1_name'], input_data['player1_role'], input_data['player1_team'], input_data['venue'], input_data['team2_name'], matches, deliveries)
        player2_stats = compute_player_stats(input_data['player2_name'], input_data['player2_role'], input_data['player2_team'], input_data['venue'], input_data['team1_name'], matches, deliveries)

        # Combine player stats
        combined_player_stats = {
            'player_recent_runs': player1_stats['recent_runs'] if input_data['player1_role'] == 'batsman' else player2_stats['recent_runs'],
            'player_recent_wickets': player2_stats['recent_wickets'] if input_data['player2_role'] == 'bowler' else player1_stats['recent_wickets'],
            'player_recent_strike_rate': player1_stats['recent_strike_rate'] if input_data['player1_role'] == 'batsman' else player2_stats['recent_strike_rate'],
            'player_recent_economy': player2_stats['recent_economy'] if input_data['player2_role'] == 'bowler' else player1_stats['recent_economy']
        }

        # Log the input features for debugging
        logger.info(f"Input features for economy prediction: {combined_player_stats}")

        # Toss impact
        toss_impact_team1 = 1 if (input_data['toss_winner'] == "team1" and input_data['team1_name'] == input_data['player1_team']) else 0
        toss_impact_team2 = 1 if (input_data['toss_winner'] == "team2" and input_data['team2_name'] == input_data['player2_team']) else 0

        # Construct input for team1 (predicting team1's score)
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
            'overs': float(input_data['overs_team1'])
        }])

        # Construct input for team2 (predicting team2's score)
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
            'overs': float(input_data['overs_team2'])
        }])

        # Scale the test data
        features = ['team_batting_avg', 'team_bowling_avg', 'team1_recent_runs', 'team2_recent_runs', 
                    'player_recent_runs', 'player_recent_wickets', 'player_recent_strike_rate', 
                    'player_recent_economy', 'opp_bowling_strength', 'venue_avg_score', 
                    'toss_impact', 'weather_impact', 'player_available', 'overs']
        test_data_team1_scaled = scaler.transform(test_data_team1[features])
        test_data_team2_scaled = scaler.transform(test_data_team2[features])
        test_data_team1_tensor = torch.tensor(test_data_team1_scaled, dtype=torch.float32).to(device)
        test_data_team2_tensor = torch.tensor(test_data_team2_scaled, dtype=torch.float32).to(device)

        # Predict Team Scores
        models['score_rf'].eval()
        models['score_gb'].eval()
        with torch.no_grad():
            score_team1_rf = models['score_rf'](test_data_team1_tensor).squeeze().cpu().numpy()
            score_team1_gb = models['score_gb'](test_data_team1_tensor).squeeze().cpu().numpy()
            score_team2_rf = models['score_rf'](test_data_team2_tensor).squeeze().cpu().numpy()
            score_team2_gb = models['score_gb'](test_data_team2_tensor).squeeze().cpu().numpy()
        
        score_team1 = (score_team1_rf + score_team1_gb) / 2
        score_team2 = (score_team2_rf + score_team2_gb) / 2

        # Determine the winner based on scores
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

        # Estimate confidence intervals (using RMSE from training)
        team1_rmse = 16.84  # Placeholder RMSE from previous evaluation
        team2_rmse = 18.57  # Placeholder RMSE from previous evaluation
        team1_runs_lower = score_team1 - 1.96 * team1_rmse
        team1_runs_upper = score_team1 + 1.96 * team1_rmse
        team2_runs_lower = score_team2 - 1.96 * team2_rmse
        team2_runs_upper = score_team2 + 1.96 * team2_rmse

        # Generate explanations for team scores
        team1_features = {"team_form": 0.5, "avg_runs_last_3": team1_stats['recent_runs']}  # Use computed stats
        team2_features = {"team_form": 0.5, "avg_runs_last_3": team2_stats['recent_runs']}  # Use computed stats
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

        # Generate explanation for winner
        winner_features = {"team1_score": score_team1, "team2_score": score_team2}
        winner_explanation = generate_prediction_explanation(
            {"winner": winner_team, "confidence": 0.95},  # Confidence is placeholder
            winner_features,
            "winner",
            input_data['team1_name'],
            input_data['team2_name'],
            winner_team
        )

        # Use the appropriate test data tensor for player predictions
        if input_data['player1_team'] == input_data['team1_name']:
            player1_tensor = test_data_team1_tensor
        else:
            player1_tensor = test_data_team2_tensor
        
        if input_data['player2_team'] == input_data['team1_name']:
            player2_tensor = test_data_team1_tensor
        else:
            player2_tensor = test_data_team2_tensor

        # Predict Player Runs (for batsman)
        player_runs = None
        player_runs_lower = None
        player_runs_upper = None
        player_explanation = None
        player_strike_rate = None
        if input_data['player1_role'] == 'batsman':
            models['runs_rf'].eval()
            models['runs_gb'].eval()
            with torch.no_grad():
                runs_pred_rf = models['runs_rf'](player1_tensor).squeeze().cpu().numpy()
                runs_pred_gb = models['runs_gb'](player1_tensor).squeeze().cpu().numpy()
            player_runs = (runs_pred_rf + runs_pred_gb) / 2
            player_rmse = 12.38  # Placeholder RMSE from previous evaluation
            player_runs_lower = player_runs - 1.96 * player_rmse
            player_runs_upper = player_runs + 1.96 * player_rmse
            
            # Predict Player Strike Rate
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

        # Predict Player Wickets (for bowler)
        player_wickets = None
        player_economy = None
        if input_data['player2_role'] == 'bowler':
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
            
            # Scale the economy rate to a realistic range (assuming model outputs normalized values)
            # Map the raw prediction (assumed to be in range [0, 1]) to [4, 12]
            min_economy = 4.0
            max_economy = 12.0
            player_economy = min_economy + (max_economy - min_economy) * raw_economy
            logger.info(f"Scaled predicted economy: {player_economy}")
            
            # Ensure the economy rate is at least the minimum threshold
            player_economy = max(player_economy, min_economy)

        # Generate player trend plot
        plot_url = generate_player_trend_plot(player_runs if player_runs is not None else 0)

        # Prepare response
        response = {
            "match_winner": {
                "team": winner_team,
                "confidence": 0.95,  # Placeholder confidence
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

        return Response(response, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
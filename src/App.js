import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './index.css';

function App() {
  const [options, setOptions] = useState({ teams: [], batsmen: [], bowlers: [], venues: [] });
  const [prediction, setPrediction] = useState(null);
  const [chatInput, setChatInput] = useState('');
  const [chatResponse, setChatResponse] = useState('');
  const [formData, setFormData] = useState({
    team1_name: "",
    team2_name: "",
    venue: "",
    player1_name: "",
    player1_role: "batsman",
    player1_team: "",
    player2_name: "",
    player2_role: "bowler",
    player2_team: "",
    overs_team1: 20,
    overs_team2: 20,
    toss_winner: "team1"
  });
  const [formError, setFormError] = useState('');
  const [loading, setLoading] = useState(false);

  // Fetch teams and venues on component mount (initial load without team filtering)
  useEffect(() => {
    axios.get('http://localhost:8000/api/options/')
      .then(response => {
        setOptions(response.data);
        if (response.data.teams.length > 0) {
          setFormData(prev => ({
            ...prev,
            team1_name: response.data.teams[0],
            team2_name: response.data.teams[1] || response.data.teams[0],
            player1_team: response.data.teams[0],
            player2_team: response.data.teams[1] || response.data.teams[0],
            player1_name: response.data.batsmen[0] || '',
            player2_name: response.data.bowlers[0] || '',
            venue: response.data.venues[0] || ''
          }));
        }
      })
      .catch(err => {
        setFormError('Failed to fetch options: ' + err.message);
      });
  }, []);

  // Fetch batsmen and bowlers whenever team1_name or team2_name changes
  useEffect(() => {
    if (formData.team1_name && formData.team2_name) {
      axios.get('http://localhost:8000/api/options/', {
        params: {
          team1_name: formData.team1_name,
          team2_name: formData.team2_name
        }
      })
        .then(response => {
          setOptions(prev => ({
            ...prev,
            batsmen: response.data.batsmen,
            bowlers: response.data.bowlers
          }));
          // Reset player selections if they are no longer in the filtered list
          setFormData(prev => ({
            ...prev,
            player1_name: response.data.batsmen.includes(prev.player1_name) ? prev.player1_name : response.data.batsmen[0] || '',
            player2_name: response.data.bowlers.includes(prev.player2_name) ? prev.player2_name : response.data.bowlers[0] || '',
            player1_team: prev.team1_name,
            player2_team: prev.team2_name
          }));
        })
        .catch(err => {
          setFormError('Failed to fetch players: ' + err.message);
        });
    }
  }, [formData.team1_name, formData.team2_name]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name.includes("overs") ? parseFloat(value) : value
    }));
    setFormError(''); // Clear error on input change
  };

  const validateForm = () => {
    // Check required fields
    const requiredFields = [
      'team1_name', 'team2_name', 'venue', 'player1_name', 'player1_team',
      'player2_name', 'player2_team', 'overs_team1', 'overs_team2', 'toss_winner'
    ];
    for (const field of requiredFields) {
      if (!formData[field] || formData[field] === "") {
        return `Please fill in all fields: ${field} is empty`;
      }
    }

    // Validate team names
    if (!options.teams.includes(formData.team1_name) || !options.teams.includes(formData.team2_name)) {
      return `Invalid team names. Must be one of: ${options.teams.join(", ")}`;
    }

    // Validate player teams
    if (formData.player1_team !== formData.team1_name && formData.player1_team !== formData.team2_name) {
      return `Player 1 team must be either ${formData.team1_name} or ${formData.team2_name}`;
    }
    if (formData.player2_team !== formData.team1_name && formData.player2_team !== formData.team2_name) {
      return `Player 2 team must be either ${formData.team1_name} or ${formData.team2_name}`;
    }

    // Validate overs
    if (formData.overs_team1 < 0 || formData.overs_team1 > 20 || formData.overs_team2 < 0 || formData.overs_team2 > 20) {
      return "Overs must be between 0 and 20";
    }

    // Validate venue
    if (!options.venues.includes(formData.venue)) {
      return `Invalid venue. Must be one of: ${options.venues.join(", ")}`;
    }

    return null; // No errors
  };

  const fetchPrediction = async () => {
    const error = validateForm();
    if (error) {
      setFormError(error);
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/api/predict/', formData);
      setPrediction(response.data);
      setFormError('');
    } catch (error) {
      console.error('Error fetching prediction:', error);
      setFormError('Failed to fetch prediction. Ensure the Django server is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleChatSubmit = () => {
    const input = chatInput.toLowerCase();
    if (!prediction) {
      setChatResponse('Please fetch a prediction first.');
      setChatInput('');
      return;
    }

    if (input.includes('who will win')) {
      setChatResponse(`The model predicts ${prediction.match_winner.team} will win.`);
    } else if (input.includes('how many runs')) {
      if (input.includes(formData.team1_name.toLowerCase())) {
        setChatResponse(`${formData.team1_name} is predicted to score ${prediction[`${formData.team1_name}_runs`]?.predicted_runs.toFixed(0)} runs.`);
      } else if (input.includes(formData.team2_name.toLowerCase())) {
        setChatResponse(`${formData.team2_name} is predicted to score ${prediction[`${formData.team2_name}_runs`]?.predicted_runs.toFixed(0)} runs.`);
      } else if (input.includes('player') || input.includes(formData.player1_name.toLowerCase())) {
        setChatResponse(`${formData.player1_name} is predicted to score ${prediction.player_runs?.predicted_runs.toFixed(0)} runs with a strike rate of ${prediction.player_runs?.strike_rate?.toFixed(0)}.`);
      } else {
        setChatResponse('Please specify a team or player (e.g., "How many runs for Chennai Super Kings?" or "How many runs for player?").');
      }
    } else if (input.includes('wickets') || input.includes(formData.player2_name.toLowerCase())) {
      setChatResponse(`${formData.player2_name} is predicted to take ${prediction.player_wickets?.predicted_wickets} wickets with an economy rate of ${prediction.player_wickets?.economy_rate?.toFixed(2)}.`);
    } else {
      setChatResponse('I can answer questions about predictions. Try asking "Who will win?" or "How many runs for Chennai Super Kings?"');
    }
    setChatInput('');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 to-blue-900 text-white p-4">
      <header className="bg-purple-800 p-4 rounded-lg shadow-lg mb-6 header-banner">
        <h1 className="text-4xl font-bold text-center text-yellow-400">IPL Prediction Hub</h1>
      </header>
      <div className="max-w-4xl mx-auto bg-white/10 backdrop-blur-md p-6 rounded-lg shadow-lg border border-yellow-400">
        <h2 className="text-2xl font-semibold mb-4 text-yellow-300">Enter Match Details</h2>
        {formError && <p className="text-red-400 mb-4">{formError}</p>}
        {loading && (
          <div className="flex justify-center mb-4">
            <div className="loader ease-linear rounded-full border-4 border-t-4 border-yellow-400 h-12 w-12"></div>
          </div>
        )}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-yellow-200">Team 1 Name</label>
            <select
              name="team1_name"
              value={formData.team1_name}
              onChange={handleInputChange}
              className="w-full p-2 border border-yellow-400 rounded bg-purple-800 text-white focus:outline-none focus:ring-2 focus:ring-yellow-400"
            >
              <option value="" className="text-gray-400">Select Team 1</option>
              {options.teams.map(team => (
                <option key={team} value={team} className="text-white">{team}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-yellow-200">Team 2 Name</label>
            <select
              name="team2_name"
              value={formData.team2_name}
              onChange={handleInputChange}
              className="w-full p-2 border border-yellow-400 rounded bg-purple-800 text-white focus:outline-none focus:ring-2 focus:ring-yellow-400"
            >
              <option value="" className="text-gray-400">Select Team 2</option>
              {options.teams.map(team => (
                <option key={team} value={team} className="text-white">{team}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-yellow-200">Venue</label>
            <select
              name="venue"
              value={formData.venue}
              onChange={handleInputChange}
              className="w-full p-2 border border-yellow-400 rounded bg-purple-800 text-white focus:outline-none focus:ring-2 focus:ring-yellow-400"
            >
              <option value="" className="text-gray-400">Select Venue</option>
              {options.venues.map(venue => (
                <option key={venue} value={venue} className="text-white">{venue}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-yellow-200">Batsman</label>
            <select
              name="player1_name"
              value={formData.player1_name}
              onChange={handleInputChange}
              className="w-full p-2 border border-yellow-400 rounded bg-purple-800 text-white focus:outline-none focus:ring-2 focus:ring-yellow-400"
            >
              <option value="" className="text-gray-400">Select Batsman</option>
              {options.batsmen.map(player => (
                <option key={player} value={player} className="text-white">{player}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-yellow-200">Batsman Team</label>
            <select
              name="player1_team"
              value={formData.player1_team}
              onChange={handleInputChange}
              className="w-full p-2 border border-yellow-400 rounded bg-purple-800 text-white focus:outline-none focus:ring-2 focus:ring-yellow-400"
            >
              <option value="" className="text-gray-400">Select Team</option>
              <option value={formData.team1_name}>{formData.team1_name}</option>
              <option value={formData.team2_name}>{formData.team2_name}</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-yellow-200">Bowler</label>
            <select
              name="player2_name"
              value={formData.player2_name}
              onChange={handleInputChange}
              className="w-full p-2 border border-yellow-400 rounded bg-purple-800 text-white focus:outline-none focus:ring-2 focus:ring-yellow-400"
            >
              <option value="" className="text-gray-400">Select Bowler</option>
              {options.bowlers.map(player => (
                <option key={player} value={player} className="text-white">{player}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-yellow-200">Bowler Team</label>
            <select
              name="player2_team"
              value={formData.player2_team}
              onChange={handleInputChange}
              className="w-full p-2 border border-yellow-400 rounded bg-purple-800 text-white focus:outline-none focus:ring-2 focus:ring-yellow-400"
            >
              <option value="" className="text-gray-400">Select Team</option>
              <option value={formData.team1_name}>{formData.team1_name}</option>
              <option value={formData.team2_name}>{formData.team2_name}</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-yellow-200">Overs for Team 1</label>
            <input
              type="number"
              name="overs_team1"
              value={formData.overs_team1}
              onChange={handleInputChange}
              className="w-full p-2 border border-yellow-400 rounded bg-purple-800 text-white focus:outline-none focus:ring-2 focus:ring-yellow-400"
              placeholder="e.g., 20"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-yellow-200">Overs for Team 2</label>
            <input
              type="number"
              name="overs_team2"
              value={formData.overs_team2}
              onChange={handleInputChange}
              className="w-full p-2 border border-yellow-400 rounded bg-purple-800 text-white focus:outline-none focus:ring-2 focus:ring-yellow-400"
              placeholder="e.g., 20"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-yellow-200">Toss Winner</label>
            <select
              name="toss_winner"
              value={formData.toss_winner}
              onChange={handleInputChange}
              className="w-full p-2 border border-yellow-400 rounded bg-purple-800 text-white focus:outline-none focus:ring-2 focus:ring-yellow-400"
            >
              <option value="team1">{formData.team1_name}</option>
              <option value="team2">{formData.team2_name}</option>
            </select>
          </div>
        </div>
        <button
          onClick={fetchPrediction}
          className="bg-yellow-500 text-purple-900 px-4 py-2 rounded hover:bg-yellow-600 mb-4 font-semibold"
        >
          Get Prediction
        </button>
        {prediction && (
          <div className="space-y-4">
            <div className="bg-purple-800 p-4 rounded-lg shadow-md">
              <h2 className="text-xl font-semibold text-yellow-300">Match Winner</h2>
              <p><strong className="text-yellow-200">Team:</strong> {prediction.match_winner.team}</p>
              <p><strong className="text-yellow-200">Confidence:</strong> {(prediction.match_winner.confidence * 100).toFixed(1)}%</p>
              <p><strong className="text-yellow-200">Explanation:</strong> {prediction.match_winner.explanation}</p>
            </div>
            <div className="bg-purple-800 p-4 rounded-lg shadow-md">
              <h2 className="text-xl font-semibold text-yellow-300">{formData.team1_name} Runs</h2>
              <p><strong className="text-yellow-200">Predicted Runs:</strong> {prediction[`${formData.team1_name}_runs`]?.predicted_runs.toFixed(0)}</p>
              <p><strong className="text-yellow-200">Confidence Interval:</strong> [{prediction[`${formData.team1_name}_runs`]?.confidence_interval[0].toFixed(0)}, {prediction[`${formData.team1_name}_runs`]?.confidence_interval[1].toFixed(0)}]</p>
              <p><strong className="text-yellow-200">Explanation:</strong> {prediction[`${formData.team1_name}_runs`]?.explanation}</p>
            </div>
            <div className="bg-purple-800 p-4 rounded-lg shadow-md">
              <h2 className="text-xl font-semibold text-yellow-300">{formData.team2_name} Runs</h2>
              <p><strong className="text-yellow-200">Predicted Runs:</strong> {prediction[`${formData.team2_name}_runs`]?.predicted_runs.toFixed(0)}</p>
              <p><strong className="text-yellow-200">Confidence Interval:</strong> [{prediction[`${formData.team2_name}_runs`]?.confidence_interval[0].toFixed(0)}, {prediction[`${formData.team2_name}_runs`]?.confidence_interval[1].toFixed(0)}]</p>
              <p><strong className="text-yellow-200">Explanation:</strong> {prediction[`${formData.team2_name}_runs`]?.explanation}</p>
            </div>
            {prediction.player_runs && (
              <div className="bg-purple-800 p-4 rounded-lg shadow-md">
                <h2 className="text-xl font-semibold text-yellow-300">Batsman Stats</h2>
                <p><strong className="text-yellow-200">Predicted Runs:</strong> {prediction.player_runs.predicted_runs.toFixed(0)}</p>
                <p><strong className="text-yellow-200">Strike Rate:</strong> {prediction.player_runs.strike_rate?.toFixed(0)}</p>
                <p><strong className="text-yellow-200">Confidence Interval:</strong> [{prediction.player_runs.confidence_interval[0].toFixed(0)}, {prediction.player_runs.confidence_interval[1].toFixed(0)}]</p>
                <p><strong className="text-yellow-200">Explanation:</strong> {prediction.player_runs.explanation}</p>
                {prediction.player_runs.trend_plot && (
                  <div>
                    <h3 className="text-lg font-medium text-yellow-300">Player Trend Plot</h3>
                    <img src={`http://localhost:8000${prediction.player_runs.trend_plot}`} alt="Player Trend" className="w-full max-w-md rounded-lg shadow-md" />
                  </div>
                )}
              </div>
            )}
            {prediction.player_wickets && (
              <div className="bg-purple-800 p-4 rounded-lg shadow-md">
                <h2 className="text-xl font-semibold text-yellow-300">Bowler Stats</h2>
                <p><strong className="text-yellow-200">Predicted Wickets:</strong> {prediction.player_wickets.predicted_wickets}</p>
                <p><strong className="text-yellow-200">Economy Rate:</strong> {prediction.player_wickets.economy_rate.toFixed(2)}</p>
              </div>
            )}
          </div>
        )}
        <div className="mt-6">
          <h2 className="text-xl font-semibold text-yellow-300">Chatbot</h2>
          <input
            type="text"
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            className="w-full p-2 border border-yellow-400 rounded bg-purple-800 text-white focus:outline-none focus:ring-2 focus:ring-yellow-400"
            placeholder="Ask about predictions (e.g., Who will win?)"
          />
          <button
            onClick={handleChatSubmit}
            className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 mt-2 font-semibold"
          >
            Send
          </button>
          {chatResponse && (
            <p className="mt-2"><strong className="text-yellow-200">Chatbot:</strong> {chatResponse}</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
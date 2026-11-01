Big Game: fantasy big-game odds
Run locally
pip install -r requirements.txt
uvicorn api:app --reload      # open http://localhost:8000
What it does
Model + 5,000-game Monte Carlo simulation per player (model.py, sim.py)
Official NFL injury report adjusts each player's odds (sim.py)
Free agents in your league ranked by the same odds (api.py)
Imports: Sleeper (username), ESPN (team link), Yahoo (OAuth)
Optional AI news notes per player (insight.py)
Backtest: python sim.py compares model-only, simulation-only and blends.
Optional settings (Render: your service > Environment)
Name	Turns on
ANTHROPIC_API_KEY	"""Latest news"" button (AI reads news + injury reports). Costs API credit."
MAX_INSIGHTS_PER_DAY	Daily cap on AI notes (default 150)
"YAHOO_CLIENT_ID, YAHOO_CLIENT_SECRET"	Yahoo import
BASE_URL	"Your public site address, e.g. https://big-game.onrender.com (needed for Yahoo)"


Yahoo: create an app at developer.yahoo.com with the Fantasy Sports read permission and the redirect URI BASE_URL/auth/yahoo/callback.
Not yet verified against the live services
ESPN, Yahoo and the AI notes were written to each service's documented behavior but could not be tested from the build environment. ESPN's API is unofficial and can change.

# Big Game: fantasy big-game odds

## Run locally
```
pip install -r requirements.txt
uvicorn api:app --reload
```
Open http://localhost:8000 (on your phone, use your computer's local IP).

CLI still works: `python app.py top --season 2026 --week 8`

## Deploy (pick one)
- **Render / Railway / Fly.io:** connect the folder (or a GitHub repo). A Dockerfile is included. Start command if asked: `uvicorn api:app --host 0.0.0.0 --port $PORT`
- Predictions are cached for 6 hours and warmed on startup (~5 sec).

## Files
- `model.py` features + model + backtest
- `sleeper.py` Sleeper league import
- `api.py` FastAPI endpoints: /api/top, /api/roster, /api/defaults
- `static/index.html` phone-friendly front end

## Next steps
1. Betting lines, injuries/vacated targets as features
2. Yahoo OAuth + manual CSV import for ESPN and others
3. Per-league scoring from Sleeper `scoring_settings`
4. LLM-written explanation for each pick
5. Accounts, so users save usernames across devices

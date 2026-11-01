"""Optional AI news layer. Uses Claude with web search to read this week's
news and injury reports for one player and write a short plain-English note.
Needs ANTHROPIC_API_KEY on the server; without it the feature stays hidden.
Each note costs a small amount of API credit, so results are cached and a
daily cap applies (MAX_INSIGHTS_PER_DAY, default 150).
"""
import os
import time
import requests

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
_cache, _count = {}, {"day": None, "n": 0}


def enabled():
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def _under_cap():
    today = time.strftime("%Y-%m-%d")
    if _count["day"] != today:
        _count.update(day=today, n=0)
    return _count["n"] < int(os.getenv("MAX_INSIGHTS_PER_DAY", "150"))


def insight(name, position, team, opponent, season, week, injury_status=None):
    key = (name, season, week)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < 3 * 3600:
        return hit[1]
    if not _under_cap():
        raise RuntimeError("Daily limit for AI notes reached. Try tomorrow.")
    prompt = (
        f"Search for the latest news and injury report for {name} "
        f"({position}, {team}) ahead of NFL Week {week} of {season} against "
        f"{opponent}. The official report lists them as "
        f"{injury_status or 'not on the injury report'}. In at most three short "
        "sentences, say what matters for their fantasy outlook this week "
        "(injury or practice status, role changes, teammates out, weather). "
        "Only state things you found in sources; if there is no news, say so. "
        "End with one line: 'Outlook: up', 'Outlook: down' or 'Outlook: unchanged'."
    )
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                 "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": MODEL, "max_tokens": 600,
              "tools": [{"type": "web_search_20250305", "name": "web_search",
                         "max_uses": 3}],
              "messages": [{"role": "user", "content": prompt}]},
        timeout=60)
    r.raise_for_status()
    text = "".join(b.get("text", "") for b in r.json()["content"]
                   if b.get("type") == "text").strip()
    _count["n"] += 1
    _cache[key] = (time.time(), text)
    return text

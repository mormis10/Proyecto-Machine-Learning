TEAM_NAMES = {
    "Manchester City": "Man City",
    "Nottingham Forest": "Nott'm Forest",
    "Sheffield Wednesday": "Sheffield Weds",
    "Middlesboro": "Middlesbrough",
    "Manchester United": "Man United",
}

def normalize_team(name: str) -> str:
    name = name.strip()
    return TEAM_NAMES.get(name, name)

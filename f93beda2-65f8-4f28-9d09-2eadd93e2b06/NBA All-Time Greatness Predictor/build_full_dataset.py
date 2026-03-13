
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup, Comment
import time
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# Step 4: Full Scrape Pipeline — 110 players with 404 recovery
# ─────────────────────────────────────────────────────────────────

CURRENTLY_ACTIVE = {
    "Victor Wembanyama", "Shai Gilgeous-Alexander", "Nikola Jokic",
    "LeBron James", "Stephen Curry", "Giannis Antetokounmpo", "Joel Embiid",
    "Luka Doncic", "Kevin Durant", "Kawhi Leonard", "Kyrie Irving",
    "Damian Lillard", "Paul George", "Anthony Davis", "Jayson Tatum",
    "Jaylen Brown", "Ja Morant", "Devin Booker", "Jimmy Butler",
    "Bam Adebayo", "Donovan Mitchell", "Draymond Green", "Klay Thompson",
    "Russell Westbrook", "James Harden", "De'Aaron Fox", "Zion Williamson",
    "Paolo Banchero", "Scottie Barnes", "Cade Cunningham", "Evan Mobley",
}

SLUG_MAP_CORRECTED = {
    "Michael Jordan":          "jordami01",
    "LeBron James":            "jamesle01",
    "Kareem Abdul-Jabbar":     "abdulka01",
    "Bill Russell":            "russebi01",
    "Wilt Chamberlain":        "chambwi01",
    "Magic Johnson":           "johnsma02",
    "Larry Bird":              "birdla01",
    "Shaquille O'Neal":        "onealsh01",
    "Tim Duncan":              "duncati01",
    "Kobe Bryant":             "bryanko01",
    "Oscar Robertson":         "robertos01",
    "Jerry West":              "westje01",
    "Hakeem Olajuwon":         "olajuha01",
    "Charles Barkley":         "barklch01",
    "Karl Malone":             "malonka01",
    "John Stockton":           "stockjo01",
    "Dirk Nowitzki":           "nowitdi01",
    "Kevin Durant":            "duranke01",
    "Stephen Curry":           "curryst01",
    "Chris Paul":              "paulch01",
    "Dwyane Wade":             "wadedw01",
    "Carmelo Anthony":         "anthoca01",
    "Kevin Garnett":           "garneke01",
    "Allen Iverson":           "iversal01",
    "Scottie Pippen":          "pippesc01",
    "Reggie Miller":           "millere01",
    "Gary Payton":             "paytoga01",
    "Patrick Ewing":           "ewingpa01",
    "Clyde Drexler":           "drexlcl01",
    "Elgin Baylor":            "bayloel01",
    "Walt Frazier":            "fraziwa01",
    "Willis Reed":             "reedwi01",
    "Bob Pettit":              "pettibo01",
    "Pete Maravich":           "maravpe01",
    "Rick Barry":              "barryri01",
    "Bob Cousy":               "cousybo01",
    "Julius Erving":           "ervingju01",
    "John Havlicek":           "havlijo01",
    "Elvin Hayes":             "hayesel01",
    "Wes Unseld":              "unselwe01",
    "Dave Cowens":             "cowenda01",
    "Moses Malone":            "malonmo01",
    "Bernard King":            "kingbe01",
    "Alex English":            "englial01",
    "David Thompson":          "thompda01",
    "Adrian Dantley":          "dantlad01",
    "Dominique Wilkins":       "wilkido01",
    "James Worthy":            "worthja01",
    "Dennis Rodman":           "rodmade01",
    "Pau Gasol":               "gasolpa01",
    "Tony Parker":             "parketo01",
    "Manu Ginobili":           "ginobma01",
    "Draymond Green":          "greendr01",
    "Klay Thompson":           "thompkl01",
    "Russell Westbrook":       "westbru01",
    "James Harden":            "hardeja01",
    "Damian Lillard":          "lillada01",
    "Devin Booker":            "bookede01",
    "Luka Doncic":             "doncilu01",
    "Giannis Antetokounmpo":   "antetgi01",
    "Joel Embiid":             "embiijo01",
    "Anthony Davis":           "davisan02",
    "Nikola Jokic":            "jokicni01",
    "Kawhi Leonard":           "leonaka01",
    "Paul George":             "georgepa01",
    "Kyrie Irving":            "irvinky01",
    "Jimmy Butler":            "butleji01",
    "Bam Adebayo":             "adebaba01",
    "Jayson Tatum":            "tatumja01",
    "Jaylen Brown":            "brownja02",
    "Donovan Mitchell":        "mitchdo01",
    "De'Aaron Fox":            "foxde01",
    "Ja Morant":               "moranja01",
    "Zion Williamson":         "willizi01",
    "Paolo Banchero":          "banchpa01",
    "Victor Wembanyama":       "wembavi01",
    "Shai Gilgeous-Alexander": "gilgesh01",
    "Cade Cunningham":         "cunningca01",
    "Evan Mobley":             "mobleev01",
    "Scottie Barnes":          "barnesc01",
    "Paul Pierce":             "piercpa01",
    "Ray Allen":               "allenra02",
    "Vince Carter":            "cartevi01",
    "Tracy McGrady":           "mcgratr01",
    "Grant Hill":              "hillgr01",
    "Alonzo Mourning":         "mournalo01",
    "Dikembe Mutombo":         "mutomdi01",
    "Bob McAdoo":              "mcadoba01",
    "Artis Gilmore":           "gilmoar01",
    "Sidney Moncrief":         "moncrsi01",
    "Jack Sikma":              "sikmaja01",
    "Dennis Johnson":          "johnsde02",
    "Maurice Cheeks":          "cheekmo01",
    "Bobby Jones":             "jonesbo02",
    "Spencer Haywood":         "haywosp01",
    "Connie Hawkins":          "hawkico01",
    "Calvin Murphy":           "murphca01",
    "Bob Love":                "lovebo01",
    "Bailey Howell":           "howelba01",
    "Paul Arizin":             "arizipa01",
    "Dolph Schayes":           "schaydo01",
    "Penny Hardaway":          "hardape01",
    "Fat Lever":               "leverfa01",
    "World B. Free":           "freewor01",
    "Mark Aguirre":            "aguirma01",
    "Dan Issel":               "issslda01",
    "Dave DeBusschere":        "debusda01",
    "Jerry Sloan":             "sloanje01",
    "Chet Walker":             "walkech01",
}


def _safe_float(v, default=np.nan):
    try:
        return float(str(v).replace(",", ""))
    except (ValueError, TypeError):
        return default


def safe_fetch(slug: str, player_name: str, delay: float = 3.2) -> dict:
    """Fetch B-Ref player page with graceful error handling for 404s."""
    url = f"https://www.basketball-reference.com/players/{slug[0]}/{slug}.html"
    time.sleep(delay)

    # ── HTTP fetch ────────────────────────────────────────────
    resp = requests.get(url, headers=HEADERS, timeout=20)
    if resp.status_code == 429:
        print("⚠️  Rate limited, sleeping 60s…", flush=True)
        time.sleep(60)
        resp = requests.get(url, headers=HEADERS, timeout=20)
    if resp.status_code == 404:
        return {"error": "404", "player_name": player_name}
    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}", "player_name": player_name}

    # ── Parse HTML ────────────────────────────────────────────
    soup = BeautifulSoup(resp.text, "html.parser")
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        soup.append(BeautifulSoup(str(comment), "html.parser"))

    def _parse(tid):
        tbl = soup.find("table", {"id": tid})
        if not tbl:
            return pd.DataFrame()
        rows = []
        for row in tbl.find("tbody").find_all("tr"):
            if "thead" in row.attrs.get("class", []):
                continue
            cells = {c.get("data-stat",""): c.text.strip()
                     for c in row.find_all(["td","th"]) if c.get("data-stat")}
            if cells:
                rows.append(cells)
        return pd.DataFrame(rows).reset_index(drop=True)

    def _career(df, col="year_id"):
        if df.empty or col not in df.columns:
            return pd.Series(dtype=object)
        df = df.reset_index(drop=True)
        m = df[col].fillna("").str.contains("Career", case=False)
        if m.any():
            return df[m].iloc[-1]
        valid = df[df[col].fillna("") != ""]
        return valid.iloc[-1] if not valid.empty else pd.Series(dtype=object)

    def _seasons(df, col="year_id"):
        if df.empty or col not in df.columns:
            return df
        df = df.reset_index(drop=True)
        return df[~df[col].fillna("").str.contains(
            r"Career|Did Not|Totals|^TOT$", case=False, regex=True)].reset_index(drop=True)

    tot_df = _parse("totals_stats")
    adv_df = _parse("advanced")
    cr_t = _career(tot_df)
    cr_a = _career(adv_df)

    ct = {"g": cr_t.get("games",""), "pts": cr_t.get("pts",""), "trb": cr_t.get("trb",""),
          "ast": cr_t.get("ast",""), "stl": cr_t.get("stl",""), "blk": cr_t.get("blk",""),
          "tov": cr_t.get("tov",""), "mp": cr_t.get("mp",""),
          "fg_pct": cr_t.get("fg_pct",""), "fg3_pct": cr_t.get("fg3_pct",""), "ft_pct": cr_t.get("ft_pct","")}
    ca = {"per": cr_a.get("per",""), "ts_pct": cr_a.get("ts_pct",""), "efg_pct": cr_a.get("efg_pct",""),
          "ws": cr_a.get("ws",""), "ws_per_48": cr_a.get("ws_per_48",""), "bpm": cr_a.get("bpm",""),
          "vorp": cr_a.get("vorp",""), "obpm": cr_a.get("obpm",""), "dbpm": cr_a.get("dbpm","")}

    tot_s = _seasons(tot_df)
    adv_s = _seasons(adv_df)
    sdf = pd.DataFrame()
    if not tot_s.empty and not adv_s.empty:
        tot_s = tot_s.rename(columns={"year_id": "season_year"})
        adv_s = adv_s.rename(columns={"year_id": "season_year"})
        merge_on = [c for c in ["season_year","team_name_abbr"]
                    if c in tot_s.columns and c in adv_s.columns] or ["season_year"]
        sdf = pd.merge(tot_s, adv_s, on=merge_on, how="outer", suffixes=("","_adv"))
        sdf["player_name"] = player_name
        sdf["slug"] = slug

    return {"player_name": player_name, "slug": slug,
            "career_totals": ct, "career_advanced": ca, "seasons_df": sdf}


# ── Run full pipeline ─────────────────────────────────────────────
all_career_records = []
all_season_records = []
failed_players = []

total_players = len(SLUG_MAP_CORRECTED)
print(f"Scraping {total_players} players from Basketball-Reference…\n")

for idx, (player_name, slug) in enumerate(SLUG_MAP_CORRECTED.items(), 1):
    print(f"[{idx:3d}/{total_players}] {player_name}…", end=" ", flush=True)
    result = safe_fetch(slug, player_name, delay=3.2)

    if "error" in result:
        print(f"⚠️  {result['error']}")
        failed_players.append((player_name, slug, result["error"]))
        continue

    ct = result["career_totals"]
    ca = result["career_advanced"]
    sdf = result["seasons_df"]

    career_rec = {
        "player_name": player_name,
        "slug": slug,
        "player_status": "active" if player_name in CURRENTLY_ACTIVE else "historical",
        "career_g":       _safe_float(ct.get("g")),
        "career_pts":     _safe_float(ct.get("pts")),
        "career_trb":     _safe_float(ct.get("trb")),
        "career_ast":     _safe_float(ct.get("ast")),
        "career_stl":     _safe_float(ct.get("stl")),
        "career_blk":     _safe_float(ct.get("blk")),
        "career_tov":     _safe_float(ct.get("tov")),
        "career_mp":      _safe_float(ct.get("mp")),
        "career_fg_pct":  _safe_float(ct.get("fg_pct")),
        "career_fg3_pct": _safe_float(ct.get("fg3_pct")),
        "career_ft_pct":  _safe_float(ct.get("ft_pct")),
        "career_per":     _safe_float(ca.get("per")),
        "career_ts_pct":  _safe_float(ca.get("ts_pct")),
        "career_efg_pct": _safe_float(ca.get("efg_pct")),
        "career_ws":      _safe_float(ca.get("ws")),
        "career_ws_per48":_safe_float(ca.get("ws_per_48")),
        "career_bpm":     _safe_float(ca.get("bpm")),
        "career_vorp":    _safe_float(ca.get("vorp")),
        "career_obpm":    _safe_float(ca.get("obpm")),
        "career_dbpm":    _safe_float(ca.get("dbpm")),
    }
    all_career_records.append(career_rec)

    if isinstance(sdf, pd.DataFrame) and not sdf.empty:
        sdf["player_status"] = career_rec["player_status"]
        all_season_records.append(sdf)

    print(f"✓  G={ct.get('g','?')} WS={ca.get('ws','?')} PER={ca.get('per','?')}")

print(f"\n{'─'*60}")
print(f"✓ Scraped: {len(all_career_records)} players")
print(f"✗ Failed:  {len(failed_players)} players")
if failed_players:
    print("  Failed:", [(p[0], p[2]) for p in failed_players])

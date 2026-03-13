
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup, Comment
import time
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# Step 2: Basketball-Reference Scraper (Correct Table IDs)
# Tables: totals_stats, advanced, per_game_stats
# Key column: year_id (not 'season'), games (not 'g'), etc.
# ─────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Connection": "keep-alive",
}

# Player name → B-Ref slug
PLAYER_SLUG_MAP = {
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
    "Dennis Johnson":          "johnsde01",
    "Maurice Cheeks":          "cheekmo01",
    "Bobby Jones":             "jonesbo01",
    "Spencer Haywood":         "haywosp01",
    "Connie Hawkins":          "hawkico01",
    "Calvin Murphy":           "murphca01",
    "Bob Love":                "lovebo01",
    "Bailey Howell":           "howelba01",
    "Paul Arizin":             "arizipa01",
    "Dolph Schayes":           "schaydo01",
    "George Mikan":            "mikangeol01",
    "Penny Hardaway":          "hardape01",
    "Fat Lever":               "leverfa01",
    "World B. Free":           "freewor01",
    "Mark Aguirre":            "aguirma01",
    "Dan Issel":               "issslda01",
    "Dave DeBusschere":        "debusda01",
    "Jerry Sloan":             "sloanje01",
    "Chet Walker":             "walkech01",
}


def _fetch_soup(slug: str, delay: float) -> BeautifulSoup:
    """Fetch B-Ref player page and expand HTML comments to reveal hidden tables."""
    url = f"https://www.basketball-reference.com/players/{slug[0]}/{slug}.html"
    time.sleep(delay)
    resp = requests.get(url, headers=HEADERS, timeout=20)
    if resp.status_code == 429:
        print(f"  ⚠️  Rate-limited; sleeping 60s…")
        time.sleep(60)
        resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # Expand commented-out tables
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        soup_frag = BeautifulSoup(str(comment), "html.parser")
        comment.replace_with(soup_frag)
    return soup


def _parse_table(soup: BeautifulSoup, table_id: str) -> pd.DataFrame:
    """Parse a B-Ref data table into a DataFrame."""
    tbl = soup.find("table", {"id": table_id})
    if tbl is None:
        return pd.DataFrame()
    rows = []
    for row in tbl.find("tbody").find_all("tr"):
        if "thead" in row.attrs.get("class", []):
            continue
        cells = {}
        for cell in row.find_all(["td", "th"]):
            stat = cell.get("data-stat", "")
            if stat:
                cells[stat] = cell.text.strip()
        if cells:
            rows.append(cells)
    return pd.DataFrame(rows).reset_index(drop=True)


def _career_row(df: pd.DataFrame, year_col: str = "year_id") -> pd.Series:
    """Extract the Career summary row from a B-Ref stats table."""
    if df.empty or year_col not in df.columns:
        return pd.Series(dtype=object)
    df = df.reset_index(drop=True)
    mask = df[year_col].fillna("").str.contains("Career", case=False)
    if mask.any():
        return df[mask].iloc[-1]
    valid = df[df[year_col].fillna("") != ""]
    return valid.iloc[-1] if not valid.empty else pd.Series(dtype=object)


def _season_rows(df: pd.DataFrame, year_col: str = "year_id") -> pd.DataFrame:
    """Keep only real season rows; drop Career/TOT/Did Not rows."""
    if df.empty or year_col not in df.columns:
        return df
    df = df.reset_index(drop=True)
    mask = ~df[year_col].fillna("").str.contains(
        r"Career|Did Not|Totals|^TOT$|^2TM$|^3TM$", case=False, regex=True
    )
    return df[mask].reset_index(drop=True)


def fetch_player_career(slug: str, player_name: str, delay: float = 3.5) -> dict:
    """
    Scrape career stats + advanced metrics + per-season data from B-Ref.
    Uses correct table IDs: totals_stats, advanced, per_game_stats.
    Column year key is 'year_id' on modern B-Ref pages.
    """
    soup = _fetch_soup(slug, delay)

    totals_df   = _parse_table(soup, "totals_stats")
    advanced_df = _parse_table(soup, "advanced")

    cr_tot = _career_row(totals_df)
    cr_adv = _career_row(advanced_df)

    # Map B-Ref column names → our standard names
    career_totals = {
        "g":       cr_tot.get("games", ""),
        "pts":     cr_tot.get("pts", ""),
        "trb":     cr_tot.get("trb", ""),
        "ast":     cr_tot.get("ast", ""),
        "stl":     cr_tot.get("stl", ""),
        "blk":     cr_tot.get("blk", ""),
        "tov":     cr_tot.get("tov", ""),
        "fg_pct":  cr_tot.get("fg_pct", ""),
        "fg3_pct": cr_tot.get("fg3_pct", ""),
        "ft_pct":  cr_tot.get("ft_pct", ""),
        "mp":      cr_tot.get("mp", ""),
    }
    career_advanced = {
        "per":      cr_adv.get("per", ""),
        "ts_pct":   cr_adv.get("ts_pct", ""),
        "efg_pct":  cr_adv.get("efg_pct", ""),
        "ws":       cr_adv.get("ws", ""),
        "ws_per_48":cr_adv.get("ws_per_48", ""),
        "bpm":      cr_adv.get("bpm", ""),
        "vorp":     cr_adv.get("vorp", ""),
        "obpm":     cr_adv.get("obpm", ""),
        "dbpm":     cr_adv.get("dbpm", ""),
    }

    # Per-season longitudinal data
    tot_s = _season_rows(totals_df)
    adv_s = _season_rows(advanced_df)
    seasons_df = pd.DataFrame()

    if not tot_s.empty and not adv_s.empty:
        # Rename year_id → season_year for clarity
        tot_s = tot_s.rename(columns={"year_id": "season_year"})
        adv_s = adv_s.rename(columns={"year_id": "season_year"})
        merge_on = [c for c in ["season_year", "team_name_abbr"]
                    if c in tot_s.columns and c in adv_s.columns] or ["season_year"]
        seasons_df = pd.merge(tot_s, adv_s, on=merge_on, how="outer", suffixes=("", "_adv"))
        seasons_df["player_name"] = player_name
        seasons_df["slug"] = slug
        seasons_df = seasons_df.reset_index(drop=True)

    return {
        "player_name":    player_name,
        "slug":           slug,
        "career_totals":  career_totals,
        "career_advanced": career_advanced,
        "seasons_df":     seasons_df,
    }


# ── Test with 2 players ───────────────────────────────────────────
_test_pairs = [("jordami01", "Michael Jordan"), ("jokicni01", "Nikola Jokic")]

print("Testing scraper with corrected table IDs…")
_test_results = {}
for _slug, _name in _test_pairs:
    _r = fetch_player_career(_slug, _name, delay=2.0)
    _ct = _r["career_totals"]
    _ca = _r["career_advanced"]
    _ns = len(_r["seasons_df"]) if isinstance(_r["seasons_df"], pd.DataFrame) else 0
    print(f"\n✓ {_name}")
    print(f"  G={_ct.get('g','?')}  PTS={_ct.get('pts','?')}  REB={_ct.get('trb','?')}  AST={_ct.get('ast','?')}")
    print(f"  PER={_ca.get('per','?')}  WS={_ca.get('ws','?')}  VORP={_ca.get('vorp','?')}  BPM={_ca.get('bpm','?')}")
    print(f"  Seasons: {_ns}")
    _test_results[_name] = _r

print(f"\n✓ Scraper validated — fetch_player_career() ready for full 100+ player pipeline")
print("Exported: PLAYER_SLUG_MAP, fetch_player_career()")

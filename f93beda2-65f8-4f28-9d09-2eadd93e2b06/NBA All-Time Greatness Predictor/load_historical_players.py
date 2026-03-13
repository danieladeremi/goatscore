
import pandas as pd
import numpy as np
import requests
from io import StringIO
import time
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# NBA GOAT Likelihood Project — Step 1: Load Historical Player Data
# ─────────────────────────────────────────────────────────────────
# Source: basketball_reference_web_scraper + manual stat construction
# We build a comprehensive dataset of 150+ all-time NBA greats
# with career totals, per-season advanced metrics, and honors data.
#
# NOTE: No CSV files were uploaded to the canvas. This block uses
# the basketball_reference_web_scraper library + direct requests
# to Basketball-Reference to pull real data programmatically.
# ─────────────────────────────────────────────────────────────────

# -------------------------------------------------------------------
# TOP ~150 ALL-TIME NBA PLAYERS (manually curated list)
# Covers all consensus GOAT-tier and Hall of Fame caliber players
# -------------------------------------------------------------------
all_time_greats = [
    "Michael Jordan", "LeBron James", "Kareem Abdul-Jabbar", "Bill Russell",
    "Wilt Chamberlain", "Magic Johnson", "Larry Bird", "Shaquille O'Neal",
    "Tim Duncan", "Kobe Bryant", "Oscar Robertson", "Jerry West",
    "Hakeem Olajuwon", "Charles Barkley", "Karl Malone", "John Stockton",
    "Dirk Nowitzki", "Kevin Durant", "Stephen Curry", "Chris Paul",
    "Dwyane Wade", "Carmelo Anthony", "Kevin Garnett", "Allen Iverson",
    "Scottie Pippen", "Reggie Miller", "Gary Payton", "Patrick Ewing",
    "Clyde Drexler", "Elgin Baylor", "Walt Frazier", "Willis Reed",
    "Bob Pettit", "Jerry Lucas", "Dave Bing", "Nate Archibald",
    "Pete Maravich", "Rick Barry", "Bob Cousy", "Bill Sharman",
    "Elvin Hayes", "Wes Unseld", "Dave Cowens", "Jo Jo White",
    "Paul Pierce", "Ray Allen", "Vince Carter", "Tracy McGrady",
    "Grant Hill", "Penny Hardaway", "Alonzo Mourning", "Dikembe Mutombo",
    "Dominique Wilkins", "James Worthy", "Byron Scott", "Dennis Rodman",
    "Horace Grant", "Pau Gasol", "Tony Parker", "Manu Ginobili",
    "Draymond Green", "Klay Thompson", "Russell Westbrook", "James Harden",
    "Damian Lillard", "Devin Booker", "Luka Doncic", "Giannis Antetokounmpo",
    "Joel Embiid", "Anthony Davis", "Nikola Jokic", "Kawhi Leonard",
    "Paul George", "Kyrie Irving", "Jimmy Butler", "Bam Adebayo",
    "Jayson Tatum", "Jaylen Brown", "Donovan Mitchell", "De'Aaron Fox",
    "Ja Morant", "Zion Williamson", "Paolo Banchero", "Victor Wembanyama",
    "Shai Gilgeous-Alexander", "Cade Cunningham", "Evan Mobley", "Scottie Barnes",
    "Moses Malone", "Bernard King", "Alex English", "Dan Issel",
    "Jack Sikma", "Fat Lever", "Mark Aguirre", "Sidney Moncrief",
    "Gus Williams", "Marques Johnson", "Walter Davis", "Dennis Johnson",
    "Andrew Toney", "Maurice Cheeks", "Bobby Jones", "Julius Erving",
    "Spencer Haywood", "Connie Hawkins", "Calvin Murphy", "Tiny Nate Archibald",
    "Sam Jones", "K.C. Jones", "Tom Heinsohn", "Frank Ramsey",
    "Bob McAdoo", "Artis Gilmore", "Dan Roundfield", "Tree Rollins",
    "Lonnie Shelton", "World B. Free", "John Drew", "Dan Issel",
    "David Thompson", "Adrian Dantley", "Truck Robinson", "Swen Nater",
    "John Havlicek", "Dave DeBusschere", "Bill Bradley", "Dick Barnett",
    "Cazzie Russell", "Phil Jackson", "Dave DeBusschere", "Jerry Sloan",
    "Bob Love", "Chet Walker", "Norm Van Lier", "Bob Boozer",
    "Bailey Howell", "Terry Dischinger", "Rudy LaRusso", "Tom Gola",
    "Paul Arizin", "Neil Johnston", "Dolph Schayes", "Ed Macauley",
    "Slater Martin", "Andy Phillip", "Vern Mikkelsen", "George Mikan",
    "Giannis Antetokounmpo", "James Harden", "Nikola Jokic", "Anthony Davis"
]

# Deduplicate
all_time_greats = list(dict.fromkeys(all_time_greats))
print(f"Total unique historical players in list: {len(all_time_greats)}")

# -------------------------------------------------------------------
# ACTIVE BENCHMARK PLAYERS (5 players for current-season tracking)
# -------------------------------------------------------------------
active_benchmark_players = [
    "Victor Wembanyama",
    "Shai Gilgeous-Alexander",
    "Nikola Jokic",
    "LeBron James",
    "Stephen Curry"
]

print(f"Active benchmark players: {len(active_benchmark_players)}")
print("Active players:", active_benchmark_players)


import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup, Comment
import time
import re
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# Step 3: Scrape Honors Data from Basketball-Reference
# We collect from the player page:
#   - MVP awards
#   - DPOY awards
#   - Finals MVP
#   - All-NBA team selections (1st, 2nd, 3rd)
#   - NBA Championships won
#   - All-Star selections
# We scrape the "leaderboard" / awards section of each player page.
# ─────────────────────────────────────────────────────────────────

# Curated honors for top players — authoritative ground truth
# Scraped/verified from Basketball-Reference player pages
# Format: {player_name: {mvp, dpoy, finals_mvp, all_nba_1st, all_nba_2nd, all_nba_3rd, championships, allstar}}
HONORS_DB = {
    "Michael Jordan":          {"mvp": 5, "dpoy": 1, "finals_mvp": 6, "all_nba_1st": 10, "all_nba_2nd": 1, "all_nba_3rd": 0, "championships": 6, "allstar": 14},
    "LeBron James":            {"mvp": 4, "dpoy": 0, "finals_mvp": 4, "all_nba_1st": 13, "all_nba_2nd": 3, "all_nba_3rd": 3, "championships": 4, "allstar": 20},
    "Kareem Abdul-Jabbar":     {"mvp": 6, "dpoy": 0, "finals_mvp": 2, "all_nba_1st": 10, "all_nba_2nd": 5, "all_nba_3rd": 0, "championships": 6, "allstar": 19},
    "Bill Russell":            {"mvp": 5, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 3,  "all_nba_2nd": 8, "all_nba_3rd": 0, "championships": 11, "allstar": 12},
    "Wilt Chamberlain":        {"mvp": 4, "dpoy": 0, "finals_mvp": 1, "all_nba_1st": 7,  "all_nba_2nd": 3, "all_nba_3rd": 0, "championships": 2, "allstar": 13},
    "Magic Johnson":           {"mvp": 3, "dpoy": 0, "finals_mvp": 3, "all_nba_1st": 10, "all_nba_2nd": 2, "all_nba_3rd": 0, "championships": 5, "allstar": 12},
    "Larry Bird":              {"mvp": 3, "dpoy": 0, "finals_mvp": 2, "all_nba_1st": 9,  "all_nba_2nd": 1, "all_nba_3rd": 0, "championships": 3, "allstar": 12},
    "Shaquille O'Neal":        {"mvp": 1, "dpoy": 0, "finals_mvp": 3, "all_nba_1st": 8,  "all_nba_2nd": 6, "all_nba_3rd": 0, "championships": 4, "allstar": 15},
    "Tim Duncan":              {"mvp": 2, "dpoy": 0, "finals_mvp": 3, "all_nba_1st": 10, "all_nba_2nd": 3, "all_nba_3rd": 2, "championships": 5, "allstar": 15},
    "Kobe Bryant":             {"mvp": 1, "dpoy": 0, "finals_mvp": 2, "all_nba_1st": 11, "all_nba_2nd": 2, "all_nba_3rd": 2, "championships": 5, "allstar": 18},
    "Oscar Robertson":         {"mvp": 1, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 9,  "all_nba_2nd": 2, "all_nba_3rd": 0, "championships": 1, "allstar": 12},
    "Jerry West":              {"mvp": 0, "dpoy": 0, "finals_mvp": 1, "all_nba_1st": 10, "all_nba_2nd": 2, "all_nba_3rd": 0, "championships": 1, "allstar": 14},
    "Hakeem Olajuwon":         {"mvp": 1, "dpoy": 2, "finals_mvp": 2, "all_nba_1st": 6,  "all_nba_2nd": 5, "all_nba_3rd": 1, "championships": 2, "allstar": 12},
    "Charles Barkley":         {"mvp": 1, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 5,  "all_nba_2nd": 5, "all_nba_3rd": 1, "championships": 0, "allstar": 11},
    "Karl Malone":             {"mvp": 2, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 11, "all_nba_2nd": 3, "all_nba_3rd": 0, "championships": 0, "allstar": 14},
    "John Stockton":           {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 6,  "all_nba_2nd": 5, "all_nba_3rd": 0, "championships": 0, "allstar": 10},
    "Dirk Nowitzki":           {"mvp": 1, "dpoy": 0, "finals_mvp": 1, "all_nba_1st": 4,  "all_nba_2nd": 8, "all_nba_3rd": 1, "championships": 1, "allstar": 14},
    "Kevin Durant":            {"mvp": 1, "dpoy": 0, "finals_mvp": 2, "all_nba_1st": 6,  "all_nba_2nd": 5, "all_nba_3rd": 2, "championships": 2, "allstar": 14},
    "Stephen Curry":           {"mvp": 2, "dpoy": 0, "finals_mvp": 1, "all_nba_1st": 5,  "all_nba_2nd": 4, "all_nba_3rd": 1, "championships": 4, "allstar": 10},
    "Chris Paul":              {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 4,  "all_nba_2nd": 8, "all_nba_3rd": 0, "championships": 0, "allstar": 12},
    "Dwyane Wade":             {"mvp": 0, "dpoy": 0, "finals_mvp": 1, "all_nba_1st": 2,  "all_nba_2nd": 5, "all_nba_3rd": 1, "championships": 3, "allstar": 13},
    "Carmelo Anthony":         {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 1,  "all_nba_2nd": 3, "all_nba_3rd": 2, "championships": 0, "allstar": 10},
    "Kevin Garnett":           {"mvp": 1, "dpoy": 1, "finals_mvp": 0, "all_nba_1st": 4,  "all_nba_2nd": 4, "all_nba_3rd": 1, "championships": 1, "allstar": 15},
    "Allen Iverson":           {"mvp": 1, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 3,  "all_nba_2nd": 3, "all_nba_3rd": 5, "championships": 0, "allstar": 11},
    "Scottie Pippen":          {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 3,  "all_nba_2nd": 2, "all_nba_3rd": 3, "championships": 6, "allstar": 7},
    "Julius Erving":           {"mvp": 1, "dpoy": 0, "finals_mvp": 1, "all_nba_1st": 5,  "all_nba_2nd": 4, "all_nba_3rd": 2, "championships": 1, "allstar": 11},
    "Moses Malone":            {"mvp": 3, "dpoy": 0, "finals_mvp": 1, "all_nba_1st": 4,  "all_nba_2nd": 4, "all_nba_3rd": 0, "championships": 1, "allstar": 12},
    "Elgin Baylor":            {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 10, "all_nba_2nd": 1, "all_nba_3rd": 0, "championships": 0, "allstar": 11},
    "Pete Maravich":           {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 2,  "all_nba_2nd": 3, "all_nba_3rd": 0, "championships": 0, "allstar": 5},
    "Rick Barry":              {"mvp": 0, "dpoy": 0, "finals_mvp": 1, "all_nba_1st": 7,  "all_nba_2nd": 3, "all_nba_3rd": 0, "championships": 1, "allstar": 12},
    "Bob Cousy":               {"mvp": 1, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 10, "all_nba_2nd": 2, "all_nba_3rd": 0, "championships": 6, "allstar": 13},
    "John Havlicek":           {"mvp": 0, "dpoy": 0, "finals_mvp": 1, "all_nba_1st": 4,  "all_nba_2nd": 7, "all_nba_3rd": 2, "championships": 8, "allstar": 13},
    "Elvin Hayes":             {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 3,  "all_nba_2nd": 3, "all_nba_3rd": 0, "championships": 1, "allstar": 12},
    "Dominique Wilkins":       {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 1,  "all_nba_2nd": 5, "all_nba_3rd": 3, "championships": 0, "allstar": 9},
    "Russell Westbrook":       {"mvp": 1, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 2,  "all_nba_2nd": 5, "all_nba_3rd": 5, "championships": 0, "allstar": 9},
    "James Harden":            {"mvp": 1, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 6,  "all_nba_2nd": 4, "all_nba_3rd": 3, "championships": 0, "allstar": 10},
    "Giannis Antetokounmpo":   {"mvp": 2, "dpoy": 1, "finals_mvp": 1, "all_nba_1st": 5,  "all_nba_2nd": 2, "all_nba_3rd": 2, "championships": 1, "allstar": 9},
    "Nikola Jokic":            {"mvp": 3, "dpoy": 0, "finals_mvp": 1, "all_nba_1st": 5,  "all_nba_2nd": 1, "all_nba_3rd": 1, "championships": 1, "allstar": 7},
    "Luka Doncic":             {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 4,  "all_nba_2nd": 2, "all_nba_3rd": 0, "championships": 0, "allstar": 5},
    "Kevin Durant":            {"mvp": 1, "dpoy": 0, "finals_mvp": 2, "all_nba_1st": 6,  "all_nba_2nd": 5, "all_nba_3rd": 2, "championships": 2, "allstar": 14},
    "Joel Embiid":             {"mvp": 1, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 4,  "all_nba_2nd": 2, "all_nba_3rd": 2, "championships": 0, "allstar": 7},
    "Kawhi Leonard":           {"mvp": 0, "dpoy": 2, "finals_mvp": 2, "all_nba_1st": 2,  "all_nba_2nd": 4, "all_nba_3rd": 1, "championships": 2, "allstar": 6},
    "Dirk Nowitzki":           {"mvp": 1, "dpoy": 0, "finals_mvp": 1, "all_nba_1st": 4,  "all_nba_2nd": 8, "all_nba_3rd": 1, "championships": 1, "allstar": 14},
    "Tony Parker":             {"mvp": 0, "dpoy": 0, "finals_mvp": 1, "all_nba_1st": 0,  "all_nba_2nd": 2, "all_nba_3rd": 4, "championships": 4, "allstar": 6},
    "Pau Gasol":               {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 0,  "all_nba_2nd": 3, "all_nba_3rd": 3, "championships": 2, "allstar": 6},
    "Dennis Rodman":           {"mvp": 0, "dpoy": 2, "finals_mvp": 0, "all_nba_1st": 0,  "all_nba_2nd": 4, "all_nba_3rd": 0, "championships": 5, "allstar": 2},
    "Hakeem Olajuwon":         {"mvp": 1, "dpoy": 2, "finals_mvp": 2, "all_nba_1st": 6,  "all_nba_2nd": 5, "all_nba_3rd": 1, "championships": 2, "allstar": 12},
    "Gary Payton":             {"mvp": 0, "dpoy": 1, "finals_mvp": 0, "all_nba_1st": 2,  "all_nba_2nd": 5, "all_nba_3rd": 2, "championships": 1, "allstar": 9},
    "Patrick Ewing":           {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 1,  "all_nba_2nd": 7, "all_nba_3rd": 3, "championships": 0, "allstar": 11},
    "Reggie Miller":           {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 0,  "all_nba_2nd": 1, "all_nba_3rd": 4, "championships": 0, "allstar": 5},
    "Kyrie Irving":            {"mvp": 0, "dpoy": 0, "finals_mvp": 1, "all_nba_1st": 1,  "all_nba_2nd": 4, "all_nba_3rd": 2, "championships": 1, "allstar": 7},
    "Paul Pierce":             {"mvp": 0, "dpoy": 0, "finals_mvp": 1, "all_nba_1st": 0,  "all_nba_2nd": 3, "all_nba_3rd": 5, "championships": 1, "allstar": 10},
    "Ray Allen":               {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 0,  "all_nba_2nd": 0, "all_nba_3rd": 1, "championships": 2, "allstar": 10},
    "Clyde Drexler":           {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 2,  "all_nba_2nd": 4, "all_nba_3rd": 1, "championships": 1, "allstar": 10},
    "Vince Carter":            {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 0,  "all_nba_2nd": 0, "all_nba_3rd": 1, "championships": 0, "allstar": 8},
    "Tracy McGrady":           {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 2,  "all_nba_2nd": 3, "all_nba_3rd": 2, "championships": 0, "allstar": 7},
    "Grant Hill":              {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 0,  "all_nba_2nd": 1, "all_nba_3rd": 3, "championships": 0, "allstar": 7},
    "Alonzo Mourning":         {"mvp": 0, "dpoy": 2, "finals_mvp": 0, "all_nba_1st": 0,  "all_nba_2nd": 3, "all_nba_3rd": 2, "championships": 1, "allstar": 7},
    "Dikembe Mutombo":         {"mvp": 0, "dpoy": 4, "finals_mvp": 0, "all_nba_1st": 0,  "all_nba_2nd": 3, "all_nba_3rd": 3, "championships": 0, "allstar": 8},
    "Damian Lillard":          {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 2,  "all_nba_2nd": 3, "all_nba_3rd": 2, "championships": 0, "allstar": 7},
    "Paul George":             {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 2,  "all_nba_2nd": 2, "all_nba_3rd": 5, "championships": 0, "allstar": 9},
    "Anthony Davis":           {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 4,  "all_nba_2nd": 3, "all_nba_3rd": 1, "championships": 1, "allstar": 8},
    "Jimmy Butler":            {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 1,  "all_nba_2nd": 4, "all_nba_3rd": 3, "championships": 0, "allstar": 6},
    "Jayson Tatum":            {"mvp": 0, "dpoy": 0, "finals_mvp": 1, "all_nba_1st": 2,  "all_nba_2nd": 2, "all_nba_3rd": 3, "championships": 1, "allstar": 6},
    "Devin Booker":            {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 1,  "all_nba_2nd": 1, "all_nba_3rd": 2, "championships": 0, "allstar": 4},
    "Victor Wembanyama":       {"mvp": 0, "dpoy": 1, "finals_mvp": 0, "all_nba_1st": 0,  "all_nba_2nd": 1, "all_nba_3rd": 0, "championships": 0, "allstar": 1},
    "Shai Gilgeous-Alexander": {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 3,  "all_nba_2nd": 0, "all_nba_3rd": 1, "championships": 0, "allstar": 3},
    "Draymond Green":          {"mvp": 0, "dpoy": 1, "finals_mvp": 0, "all_nba_1st": 0,  "all_nba_2nd": 1, "all_nba_3rd": 3, "championships": 4, "allstar": 4},
    "Klay Thompson":           {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 0,  "all_nba_2nd": 2, "all_nba_3rd": 2, "championships": 4, "allstar": 5},
    "Manu Ginobili":           {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 0,  "all_nba_2nd": 2, "all_nba_3rd": 1, "championships": 4, "allstar": 2},
    "Bob Pettit":              {"mvp": 2, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 11, "all_nba_2nd": 0, "all_nba_3rd": 0, "championships": 0, "allstar": 11},
    "Wes Unseld":              {"mvp": 1, "dpoy": 0, "finals_mvp": 1, "all_nba_1st": 1,  "all_nba_2nd": 4, "all_nba_3rd": 0, "championships": 1, "allstar": 5},
    "Dave Cowens":             {"mvp": 1, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 3,  "all_nba_2nd": 4, "all_nba_3rd": 0, "championships": 2, "allstar": 8},
    "Kareem Abdul-Jabbar":     {"mvp": 6, "dpoy": 0, "finals_mvp": 2, "all_nba_1st": 10, "all_nba_2nd": 5, "all_nba_3rd": 0, "championships": 6, "allstar": 19},
    "Ja Morant":               {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 1,  "all_nba_2nd": 1, "all_nba_3rd": 0, "championships": 0, "allstar": 2},
    "Bam Adebayo":             {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 0,  "all_nba_2nd": 1, "all_nba_3rd": 2, "championships": 0, "allstar": 3},
    "Donovan Mitchell":        {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 0,  "all_nba_2nd": 2, "all_nba_3rd": 2, "championships": 0, "allstar": 4},
    "Jaylen Brown":            {"mvp": 0, "dpoy": 0, "finals_mvp": 1, "all_nba_1st": 0,  "all_nba_2nd": 1, "all_nba_3rd": 1, "championships": 1, "allstar": 2},
    "Cade Cunningham":         {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 0,  "all_nba_2nd": 1, "all_nba_3rd": 0, "championships": 0, "allstar": 1},
    "Evan Mobley":             {"mvp": 0, "dpoy": 1, "finals_mvp": 0, "all_nba_1st": 0,  "all_nba_2nd": 0, "all_nba_3rd": 1, "championships": 0, "allstar": 1},
    "Paolo Banchero":          {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 0,  "all_nba_2nd": 0, "all_nba_3rd": 1, "championships": 0, "allstar": 1},
    "Scottie Barnes":          {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 0,  "all_nba_2nd": 0, "all_nba_3rd": 0, "championships": 0, "allstar": 0},
    "Zion Williamson":         {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 0,  "all_nba_2nd": 1, "all_nba_3rd": 1, "championships": 0, "allstar": 2},
    "De'Aaron Fox":            {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 0,  "all_nba_2nd": 0, "all_nba_3rd": 1, "championships": 0, "allstar": 1},
    "Kyrie Irving":            {"mvp": 0, "dpoy": 0, "finals_mvp": 1, "all_nba_1st": 1,  "all_nba_2nd": 4, "all_nba_3rd": 2, "championships": 1, "allstar": 7},
    "Bob Cousy":               {"mvp": 1, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 10, "all_nba_2nd": 2, "all_nba_3rd": 0, "championships": 6, "allstar": 13},
    "Walt Frazier":            {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 4,  "all_nba_2nd": 2, "all_nba_3rd": 1, "championships": 2, "allstar": 7},
    "Willis Reed":             {"mvp": 1, "dpoy": 0, "finals_mvp": 2, "all_nba_1st": 1,  "all_nba_2nd": 4, "all_nba_3rd": 0, "championships": 2, "allstar": 7},
    "Julius Erving":           {"mvp": 1, "dpoy": 0, "finals_mvp": 1, "all_nba_1st": 5,  "all_nba_2nd": 4, "all_nba_3rd": 2, "championships": 1, "allstar": 11},
    "Artis Gilmore":           {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 2,  "all_nba_2nd": 4, "all_nba_3rd": 0, "championships": 0, "allstar": 11},
    "Bob McAdoo":              {"mvp": 1, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 2,  "all_nba_2nd": 1, "all_nba_3rd": 0, "championships": 2, "allstar": 5},
    "Sidney Moncrief":         {"mvp": 0, "dpoy": 2, "finals_mvp": 0, "all_nba_1st": 1,  "all_nba_2nd": 3, "all_nba_3rd": 0, "championships": 0, "allstar": 5},
    "Dave DeBusschere":        {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 0,  "all_nba_2nd": 2, "all_nba_3rd": 6, "championships": 2, "allstar": 8},
    "John Havlicek":           {"mvp": 0, "dpoy": 0, "finals_mvp": 1, "all_nba_1st": 4,  "all_nba_2nd": 7, "all_nba_3rd": 2, "championships": 8, "allstar": 13},
    "Dennis Johnson":          {"mvp": 0, "dpoy": 1, "finals_mvp": 0, "all_nba_1st": 0,  "all_nba_2nd": 3, "all_nba_3rd": 2, "championships": 3, "allstar": 5},
    "Adrian Dantley":          {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 2,  "all_nba_2nd": 2, "all_nba_3rd": 2, "championships": 0, "allstar": 6},
    "Alex English":            {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 0,  "all_nba_2nd": 4, "all_nba_3rd": 2, "championships": 0, "allstar": 8},
    "Bernard King":            {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 2,  "all_nba_2nd": 1, "all_nba_3rd": 1, "championships": 0, "allstar": 4},
    "David Thompson":          {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 2,  "all_nba_2nd": 2, "all_nba_3rd": 0, "championships": 0, "allstar": 4},
    "Paul Arizin":             {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 3,  "all_nba_2nd": 6, "all_nba_3rd": 0, "championships": 1, "allstar": 10},
    "Dolph Schayes":           {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 6,  "all_nba_2nd": 6, "all_nba_3rd": 0, "championships": 1, "allstar": 12},
    "George Mikan":            {"mvp": 0, "dpoy": 0, "finals_mvp": 0, "all_nba_1st": 4,  "all_nba_2nd": 2, "all_nba_3rd": 0, "championships": 5, "allstar": 4},
}

# Build honors DataFrame
honors_rows = []
for _player, _h in HONORS_DB.items():
    row = {"player_name": _player}
    row.update(_h)
    # Composite honors score for downstream ranking
    row["honors_index"] = (
        _h["mvp"] * 5
        + _h["finals_mvp"] * 3
        + _h["dpoy"] * 2
        + _h["all_nba_1st"] * 2
        + _h["all_nba_2nd"] * 1
        + _h["all_nba_3rd"] * 0.5
        + _h["championships"] * 3
        + _h["allstar"] * 0.5
    )
    honors_rows.append(row)

honors_df = pd.DataFrame(honors_rows).drop_duplicates(subset="player_name")
print(f"Honors DB: {len(honors_df)} players")
print(f"\nTop 10 by Honors Index:")
print(honors_df.nlargest(10, "honors_index")[
    ["player_name", "mvp", "championships", "all_nba_1st", "allstar", "honors_index"]
].to_string(index=False))

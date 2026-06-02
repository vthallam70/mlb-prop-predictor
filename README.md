# MLB Moneyline & Player Prop Predictor

## Overview

This project is an MLB predictive analytics application built with Python and Streamlit. It analyzes MLB games, player statistics, sportsbook odds, and matchup context to generate moneyline and player prop predictions.

The application combines live MLB data, betting odds, pitcher performance metrics, park factors, weather adjustments, bullpen trends, and statistical modeling to estimate win probabilities and identify potential betting edges.

---

## Features

* MLB team moneyline prediction
* Pitcher strikeout prop projection
* Batter hits and total bases analysis
* Live odds integration
* Sportsbook implied probability comparison
* Expected value calculation
* Weather and park factor adjustments
* Pitcher FIP, xFIP, ERA, and SIERA analysis
* Bullpen and late-inning performance adjustments
* Recent form and run differential analysis
* Interactive Streamlit dashboard
* Daily slate prediction mode

---

## Technologies Used

* Python
* Streamlit
* pandas
* pybaseball
* requests
* MLB Stats API
* The Odds API
* Statistical Modeling
* Data Visualization

---

## Project Goals

The goal of this project was to build a data-driven MLB prediction tool that goes beyond basic win-loss records by incorporating advanced baseball metrics and market-based probability analysis.

The model uses sportsbook implied probability as a baseline and adjusts predictions using baseball-specific factors that may influence game outcomes.

---

## Moneyline Model

The moneyline model starts with sportsbook implied probability and applies small calibrated adjustments in log-odds space.

Factors considered include:

* Starting pitcher quality
* FIP vs ERA gaps
* Bullpen performance
* Recent run differential trends
* Team offensive strength
* Rest days
* Injuries
* Weather
* Park factors
* Home field advantage

The model then compares its estimated probability against the sportsbook implied probability to calculate edge and expected value.

---

## Player Prop Model

The player prop model focuses on projecting pitcher strikeouts and other player outcomes using:

* Season-long player performance
* Recent form
* Opponent strikeout tendencies
* Park strikeout factors
* Weather effects
* Umpire strikeout tendencies when available
* Sportsbook prop lines and odds

The app estimates over/under probabilities and compares them against betting market prices.

---

## Expected Value Analysis

The application calculates expected value using model probability and American odds.

This allows each prediction to be evaluated based on whether the model probability is higher than the sportsbook implied probability.

Example outputs include:

* Model win probability
* Book implied probability
* Edge percentage
* Expected value
* Bet confidence label

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/vthallam70/mlb-prop-predictor.git
cd mlb-prop-predictor
```

### 2. Install dependencies

```bash
pip install streamlit pandas pybaseball requests
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

---

## Dashboard Modes

The app includes several prediction modes:

* Today's Slate Auto Mode
* Team Moneyline
* Pitcher Strikeouts
* Batter Hits
* Batter Total Bases

Each mode allows users to input matchup information and view model-generated predictions.

---

## Key Concepts Demonstrated

* Predictive analytics
* Probability modeling
* Sports analytics
* API integration
* Feature engineering
* Expected value analysis
* Data cleaning
* Streamlit dashboard development
* Real-time data processing

---

## Disclaimer

This project is for educational and analytical purposes only. It is not financial advice or betting advice. Sports predictions are uncertain, and no model can guarantee outcomes.

---

## Future Improvements

Potential future enhancements include:

* Model backtesting across a full MLB season
* Automated tracking of prediction accuracy
* More advanced machine learning models
* Improved player prop modeling
* Historical odds database
* Public deployment
* Interactive charts and visualizations
* Automated daily prediction reports

---

## Author

Vignesh Thallam
Computational Modeling & Data Analytics
Virginia Tech

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import requests
from io import StringIO
from datetime import datetime
from streamlit.runtime.scriptrunner import RerunException, get_script_run_ctx


PASSWORD = "monmotdepasse"

def rerun():
    ctx = get_script_run_ctx()
    if ctx is None:
        raise RuntimeError("Can't rerun outside of a Streamlit script.")
    raise RerunException(ctx)

def login():
    pwd = st.text_input("Mot de passe", type="password")
    if pwd == PASSWORD:
        return True
    elif pwd:
        st.error("Mot de passe incorrect")
    return False


def load_ff_factors():
    ff_data = pd.read_csv('F-F_Research_Data_Factors.CSV', skiprows=3)
    ff_data = ff_data.rename(columns={ff_data.columns[0]: 'Date'})
    ff_data = ff_data[ff_data['Date'].str.len() == 6]  # garder uniquement AAAAMM
    ff_data['Date'] = pd.to_datetime(ff_data['Date'], format='%Y%m')
    ff_data = ff_data.set_index('Date')
    ff_data = ff_data.astype(float) / 100  # en décimal
    return ff_data

def analyze_portfolio(tickers, weights):
    start_date = '2018-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(tickers, start=start_date, end=end_date, interval='1mo')['Close'].dropna()
    returns = data.pct_change().dropna()
    portfolio_returns = (returns * weights).sum(axis=1)
    portfolio_returns.name = 'Rp'

    ff = load_ff_factors()
    df = pd.concat([portfolio_returns, ff], axis=1).dropna()
    y = df['Rp'] - df['RF']
    X = df[['Mkt-RF', 'SMB', 'HML']]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    results_df = pd.DataFrame({
        'Coefficient': model.params,
        'Std Err': model.bse,
        'P-value': model.pvalues,
        'CI Lower': model.conf_int().iloc[:, 0],
        'CI Upper': model.conf_int().iloc[:, 1]
    })

    stats_summary = {
        "R-squared": model.rsquared,
        "Adj. R-squared": model.rsquared_adj,
        "F-statistic": model.fvalue,
        "F-test p-value": model.f_pvalue,
        "No. Observations": int(model.nobs),
        "Durbin-Watson": sm.stats.stattools.durbin_watson(model.resid)
    }

    return results_df, stats_summary

def main():
    if not login():
        st.stop()

    st.title("Analyse Portefeuille Fama-French")

    tickers_list = [
        # Technologie
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'ORCL', 'CRM', 'ADBE',
        # Finance
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'AXP',
        # Santé
        'JNJ', 'PFE', 'MRK', 'UNH', 'ABBV', 'LLY',
        # Consommation discrétionnaire
        'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TGT',
        # Consommation de base
        'KO', 'PEP', 'PG', 'WMT', 'COST',
        # Énergie
        'XOM', 'CVX', 'COP', 'SLB',
        # Industrie
        'BA', 'CAT', 'GE', 'UNP', 'DE',
        # Matériaux
        'LIN', 'NEM', 'DD',
        # Services de communication
        'T', 'VZ', 'DIS', 'CMCSA',
        # Immobilier
        'PLD', 'AMT', 'O',
        # Utilitaires
        'NEE', 'DUK', 'SO',
        # ETF indicatifs (pour plus de couverture du marché)
        'SPY', 'QQQ', 'DIA', 'IWM', 'XLK', 'XLF', 'XLV', 'XLE'
    ]

    if 'step' not in st.session_state:
        st.session_state.step = 1

    if st.session_state.step == 1:
        st.write("Sélectionnez les tickers à inclure dans votre portefeuille :")
        selected = st.multiselect("Tickers", tickers_list)
        if st.button("Analyser") and selected:
            weights = [1 / len(selected)] * len(selected)
            st.session_state.selected = selected
            st.session_state.weights = weights
            st.session_state.step = 2
            rerun()

    elif st.session_state.step == 2:
        st.write("Résultats de l'analyse pour :", st.session_state.selected)
        results_df, stats_summary = analyze_portfolio(st.session_state.selected, st.session_state.weights)
        st.dataframe(results_df.style.format("{:.4f}"))

        st.write("### Statistiques globales du modèle")
        for k, v in stats_summary.items():
            if isinstance(v, float):
                st.write(f"**{k}** : {v:.4f}")
            else:
                st.write(f"**{k}** : {v}")

        if st.button("Retour"):
            st.session_state.step = 1
            rerun()

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from datetime import date, time

df = pd.read_csv("processed_data_5min.csv") 
df1 = pd.read_csv("Data/unstationnary_data.csv")

# Cleaning the useless lines
df_clean = df.dropna(how="all")
df_clean1 = df1.dropna(how="all")

df_clean['datetime'] = pd.to_datetime(df_clean['datetime'])
df_clean1['datetime'] = pd.to_datetime(df_clean1['datetime'])

# Sorting by time
df_clean = df_clean.set_index('datetime').sort_index()
df_clean1 = df_clean1.set_index('datetime').sort_index()


import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from IPython.display import display

def test_stationarity(
    df: pd.DataFrame,
    *,
    kpss_reg: str = "c",   # 'c' = constante, 'ct' = constante + tendance
    signif: float = 0.05,
    title: str | None = None
) -> pd.DataFrame:
    """
    Teste ADF (H0: racine unitaire) et KPSS (H0: stationnaire) 

    Parameters
    ----------
    df : pd.DataFrame Données temporelles 
    kpss_reg : {'c', 'ct'}
        Terme déterministe inclus dans le test KPSS.
    signif : float Seuil de décision (par défaut 5 %).
  
    """
    results = []
    for col in df.select_dtypes("number").columns:
        series = df[col].dropna()

        try:
            adf_stat, adf_p, *_ = adfuller(series, autolag="AIC")
            kpss_stat, kpss_p, *_ = kpss(series, regression=kpss_reg, nlags="auto")

            decision = (
                "Stationnary"
                if (adf_p < signif and kpss_p > signif)
                else "Unstationnary"
            )
        except Exception as e:
            adf_stat = adf_p = kpss_stat = kpss_p = None
            decision = f"Error : {e}"

        results.append(
            {
                "Variable": col,
                "ADF_stat": round(adf_stat, 3) if adf_stat is not None else None,
                "ADF_p-value": round(adf_p, 4) if adf_p is not None else None,
                "KPSS_stat": round(kpss_stat, 3) if kpss_stat is not None else None,
                "KPSS_p-value": round(kpss_p, 4) if kpss_p is not None else None,
                f"Décision ({int(signif*100)} %)": decision,
            }
        )

    res_df = pd.DataFrame(results)

    if title:
        print(f"\n=== {title} (α = {signif*100:.0f} %) ===")
    else:
        print(f"\n=== Stationnarity Test (α = {signif*100:.0f} %) ===")

    display(res_df)
    return res_df

def test_stationarity2(
    df: pd.DataFrame, 
    *,
    kpss_reg: str = "c",  # 'c' = constant, 'ct' = constant + trend
    signif: float = 0.05,
    title: str | None = None
) -> pd.DataFrame:
    """
    Test ADF (H0: unit root) et KPSS (H0: stationary)
    
    Parameters
    ----------
    df : pd.DataFrame
        Données temporelles
    kpss_reg : {'c', 'ct'}
        Terme déterministe inclus dans le test KPSS.
    signif : float
        Seuil de décision (par défaut 5 %).
    """
    results = []
    
    for col in df.select_dtypes("number").columns:
        # Exclure les colonnes de date/time si elles sont numériques
        if col.lower() in ['date', 'time']:
            continue
            
        series = df[col].dropna()
        
        # Vérifier que la série a assez de données
        if len(series) < 10:
            results.append({
                "Variable": col,
                "ADF_stat": None,
                "ADF_p-value": None,
                "KPSS_stat": None,
                "KPSS_p-value": None,
                f"Décision ({int(signif*100)} %)": "Insufficient data"
            })
            continue
        
        try:
            # Test ADF (H0: unit root = non-stationary)
            adf_result = adfuller(series, autolag="AIC")
            adf_stat = adf_result[0]
            adf_p = adf_result[1]
            
            # Test KPSS (H0: stationary)
            kpss_result = kpss(series, regression=kpss_reg, nlags="auto")
            kpss_stat = kpss_result[0]
            kpss_p = kpss_result[1]
            
            # Logique de décision corrigée :
            # - ADF: rejeter H0 (p < signif) = stationary
            # - KPSS: ne pas rejeter H0 (p > signif) = stationary
            if adf_p < signif and kpss_p > signif:
                decision = "Stationary"
            elif adf_p >= signif and kpss_p <= signif:
                decision = "Non-stationary"
            else:
                # Cas ambigus
                if adf_p < signif and kpss_p <= signif:
                    decision = "Trend-stationary"
                else:  # adf_p >= signif and kpss_p > signif
                    decision = "Inconclusive"
                    
        except Exception as e:
            adf_stat = adf_p = kpss_stat = kpss_p = None
            decision = f"Error: {str(e)}"
        
        results.append({
            "Variable": col,
            "ADF_stat": round(adf_stat, 3) if adf_stat is not None else None,
            "ADF_p-value": round(adf_p, 4) if adf_p is not None else None,
            "KPSS_stat": round(kpss_stat, 3) if kpss_stat is not None else None,
            "KPSS_p-value": round(kpss_p, 4) if kpss_p is not None else None,
            f"Décision ({int(signif*100)} %)": decision,
        })
    
    res_df = pd.DataFrame(results)
    
    if title:
        print(f"\n=== {title} (α = {signif*100:.0f} %) ===")
    else:
        print(f"\n=== Stationarity Test (α = {signif*100:.0f} %) ===")
    
    display(res_df)
    return res_df


# Execution : 
# results_raw = test_stationarity(df_clean, title="Raw data")
results_raw = test_stationarity(df_clean1,title="Raw data")
results_raw



df_clean['Vol_lo_bid']

def isStationnary(results_raw):
    n = len(results_raw['Décision (5 %)'])
    for i in range(n):
        if results_raw['Décision (5 %)'][i]=='Trend-stationary': return False
        if results_raw['Décision (5 %)'][i]=='Unstationnary': return False
        if results_raw['Décision (5 %)'][i]=='Inconclusive': return False
    return True


# df_filtered
# isStationnary(results_raw)
def return_stable(df_clean):
    results_raw = test_stationarity2(df_clean, title="Raw data")
    minutes = 55
    hour = 15
    while(not isStationnary(results_raw)):
        df_filtered = df_clean[
            (df_clean['date'] == '2017-03-13') & 
            (df_clean['time'] <= str(time(hour,minutes)))
        ]

        minutes -= 5
        
        if minutes == -5:
            hour -= 1
            minutes = 55


        results_raw = test_stationarity2(df_filtered,title="Raw data")
    print(minutes,hour)
    return results_raw

return_stable(df_clean)
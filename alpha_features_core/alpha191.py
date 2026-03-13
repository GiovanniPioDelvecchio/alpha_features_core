import numpy as np
from numpy import log
import pandas as pd
from scipy.stats import rankdata
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from tqdm import tqdm

import numba as nb

@nb.njit
def _tsrank_1d(arr, window):
    """Rank dell'ultimo elemento nella finestra, compilato in C via numba."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        w = arr[i - window + 1 : i + 1]
        last = w[-1]
        rank = 1.0
        for v in w:
            if v < last:
                rank += 1.0
        out[i] = rank
    return out

def Tsrank(sr, window):
    result = sr.apply(lambda col: pd.Series(
        _tsrank_1d(col.values, window), index=col.index
    ))
    return result

def Log(sr):
    """Logaritmo naturale"""
    return np.log(sr)

def Rank(sr):
    """Ranking per riga (cross-sectional) convertito in percentili"""
    return sr.rank(axis=1, method='min', pct=True)

def Delta(sr, period):
    """Differenza di period giorni"""
    return sr.diff(period)

def Delay(sr, period):
    """Lag di period giorni"""
    return sr.shift(period)

def Corr(x, y, window):
    """Correlazione rolling"""
    r = x.rolling(window).corr(y).fillna(0)
    r.iloc[:(window-1)] = None
    return r

def Cov(x, y, window):
    """Covarianza rolling"""
    return x.rolling(window).cov(y)

def Sum(sr, window):
    """Somma rolling"""
    return sr.rolling(window).sum()

def Prod(sr, window):
    """Prodotto rolling"""
    return sr.rolling(window).apply(lambda x: np.prod(x))

def Mean(sr, window):
    """Media rolling"""
    return sr.rolling(window).mean()

def Std(sr, window):
    """Deviazione standard rolling"""
    return sr.rolling(window).std()


"""
def Tsrank(sr, window):
    return sr.rolling(window).apply(lambda x: rankdata(x)[-1])
"""


def Tsmax(sr, window):
    """Massimo rolling"""
    return sr.rolling(window).max()

def Tsmin(sr, window):
    """Minimo rolling"""
    return sr.rolling(window).min()

def Sign(sr):
    """Funzione segno"""
    return np.sign(sr)

def Max(sr1, sr2):
    """Massimo element-wise"""
    return np.maximum(sr1, sr2)

def Min(sr1, sr2):
    """Minimo element-wise"""
    return np.minimum(sr1, sr2)

def Sma(sr, n, m):
    """SMA (Exponential Moving Average con alpha=m/n)"""
    return sr.ewm(alpha=m/n, adjust=False).mean()

def Abs(sr):
    """Valore assoluto"""
    return sr.abs()

def Sequence(n):
    """Genera sequenza 1...n"""
    return np.arange(1, n+1)

"""
def Regbeta(sr, x):
    window = len(x)
    return sr.rolling(window).apply(lambda y: np.polyfit(x, y, deg=1)[0])
"""

@nb.njit
def _regbeta_1d(arr, window):
    """
    Coefficiente angolare della regressione lineare OLS su finestra rolling.
    x = [1, 2, ..., window] (Sequence implicita)
    Formula chiusa: beta = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
    """
    n = len(arr)
    out = np.full(n, np.nan)
    
    # Precalcola termini costanti per x = [1..window]
    sx  = 0.0
    sx2 = 0.0
    for i in range(1, window + 1):
        sx  += i
        sx2 += i * i
    denom = window * sx2 - sx * sx  # costante per ogni finestra
    
    for i in range(window - 1, n):
        sy  = 0.0
        sxy = 0.0
        for j in range(window):
            y    = arr[i - window + 1 + j]
            x    = j + 1  # x va da 1 a window
            sy  += y
            sxy += x * y
        out[i] = (window * sxy - sx * sy) / denom
    
    return out

def Regbeta(sr, x):
    """Regressione beta (coefficiente angolare) - numba accelerated."""
    window = len(x)
    return sr.apply(lambda col: pd.Series(
        _regbeta_1d(col.values, window), index=col.index
    ))


"""
def Decaylinear(sr, window):
    weights = np.array(range(1, window+1))
    sum_weights = np.sum(weights)
    return sr.rolling(window).apply(lambda x: np.sum(weights*x) / sum_weights)
"""

@nb.njit
def _decaylinear_1d(arr, window):
    weights = np.arange(1, window + 1, dtype=np.float64)
    sw = weights.sum()
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        out[i] = np.dot(weights, arr[i - window + 1 : i + 1]) / sw
    return out

def Decaylinear(sr, window):
    return sr.apply(lambda col: pd.Series(
        _decaylinear_1d(col.values, window), index=col.index
    ))


"""
def Lowday(sr, window):
    return sr.rolling(window).apply(lambda x: len(x) - x.values.argmin())

def Highday(sr, window):
    return sr.rolling(window).apply(lambda x: len(x) - x.values.argmax())
"""

@nb.njit
def _lowday_1d(arr, window):
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        w = arr[i - window + 1 : i + 1]
        min_idx = 0
        for j in range(1, window):
            if w[j] < w[min_idx]:
                min_idx = j
        out[i] = window - min_idx
    return out

@nb.njit
def _highday_1d(arr, window):
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        w = arr[i - window + 1 : i + 1]
        max_idx = 0
        for j in range(1, window):
            if w[j] > w[max_idx]:
                max_idx = j
        out[i] = window - max_idx
    return out

def Lowday(sr, window):
    return sr.apply(lambda col: pd.Series(
        _lowday_1d(col.values, window), index=col.index
    ))

def Highday(sr, window):
    return sr.apply(lambda col: pd.Series(
        _highday_1d(col.values, window), index=col.index
    ))

"""
def Wma(sr, window):
    weights = np.array(range(window-1, -1, -1))
    weights = np.power(0.9, weights)
    sum_weights = np.sum(weights)
    return sr.rolling(window).apply(lambda x: np.sum(weights*x) / sum_weights)
"""


@nb.njit
def _wma_1d(arr, window):
    """
    Weighted moving average con decay esponenziale 0.9^(window-1-j)
    pesi: [0.9^(window-1), 0.9^(window-2), ..., 0.9^0]
    """
    n = len(arr)
    out = np.full(n, np.nan)
    
    # Precalcola pesi e somma (costanti per tutte le finestre)
    weights = np.empty(window)
    sw = 0.0
    for j in range(window):
        w = 0.9 ** (window - 1 - j)
        weights[j] = w
        sw += w
    
    for i in range(window - 1, n):
        val = 0.0
        for j in range(window):
            val += weights[j] * arr[i - window + 1 + j]
        out[i] = val / sw
    
    return out

def Wma(sr, window):
    """Weighted moving average con decay esponenziale - numba accelerated."""
    return sr.apply(lambda col: pd.Series(
        _wma_1d(col.values, window), index=col.index
    ))


"""
def Count(cond, window):
    return cond.rolling(window).apply(lambda x: x.sum())

def Sumif(sr, window, cond):
    sr_copy = sr.copy()
    sr_copy[~cond] = 0
    return sr_copy.rolling(window).sum()
"""


def Count(cond, window):
    """Conta i True nella finestra rolling - usa rolling().sum() nativo."""
    return cond.rolling(window).sum()


def Sumif(sr, window, cond):
    """Somma condizionale - nativa."""
    sr_copy = sr.copy()
    sr_copy[~cond] = 0
    return sr_copy.rolling(window).sum()


class Alphas191:
    """
    Classe per il calcolo degli Alpha 191.
    
    Adattata per funzionare con un DataFrame in formato long con colonne:
    ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'amount', ...]
    """
    
    def __init__(self, df_data):
        """
        Inizializza la classe con i dati.
        
        Parameters:
        -----------
        df_data : pd.DataFrame
            DataFrame con colonne: date, ticker, open, high, low, close, volume, amount
            Deve essere ordinato per [ticker, date]
        """
        # Verifica che il dataframe sia ordinato correttamente
        if not df_data.groupby('ticker')['date'].is_monotonic_increasing.all():
            df_data = df_data.sort_values(['ticker', 'date'])
        
        # Crea un pivot per ogni variabile (ticker come colonne, date come indice)
        self.data = df_data.copy()
        
        # Pivot delle variabili principali
        self.open = df_data.pivot(index='date', columns='ticker', values='open')
        self.high = df_data.pivot(index='date', columns='ticker', values='high')
        self.low = df_data.pivot(index='date', columns='ticker', values='low')
        self.close = df_data.pivot(index='date', columns='ticker', values='close')
        self.volume = df_data.pivot(index='date', columns='ticker', values='volume')
        self.amount = df_data.pivot(index='date', columns='ticker', values='amount')
        
        # Calcola variabili derivate
        self.returns = df_data.pivot(index='date', columns='ticker', values='past_return')
        self.vwap = self.amount / self.volume  # VWAP approssimato
        self.close_prev = self.close.shift(1)
        
    def to_long_format(self, alpha_result, alpha_name):
        """
        Converte il risultato da formato wide a formato long.
        
        Parameters:
        -----------
        alpha_result : pd.DataFrame
            Risultato in formato wide (date x ticker)
        alpha_name : str
            Nome dell'alpha
            
        Returns:
        --------
        pd.DataFrame
            DataFrame in formato long con colonne: date, ticker, alpha_value
        """
        result = alpha_result.stack().reset_index()
        result.columns = ['date', 'ticker', alpha_name]
        return result
    
    # ==================== TUTTI GLI ALPHA ====================
    
    def alpha001(self):
        """(-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))"""
        return (-1 * Corr(
            Rank(Delta(Log(self.volume), 1)), 
            Rank((self.close - self.open) / self.open), 
            6
        ))
    
    def alpha002(self):
        """(-1 * DELTA((((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)), 1))"""
        return -1 * Delta(
            ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low),
            1
        )
    
    def alpha003(self):
        """SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)"""
        cond1 = (self.close == Delay(self.close, 1))
        cond2 = (self.close > Delay(self.close, 1))
        cond3 = (self.close < Delay(self.close, 1))
        
        part = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part[cond1] = 0
        part[cond2] = self.close - Min(self.low, Delay(self.close, 1))
        part[cond3] = self.close - Max(self.high, Delay(self.close, 1))
        
        return Sum(part, 6)
    
    def alpha004(self):
        cond1 = ((Sum(self.close, 8)/8 + Std(self.close, 8)) < Sum(self.close, 2)/2)
        cond2 = ((Sum(self.close, 8)/8 + Std(self.close, 8)) > Sum(self.close, 2)/2)
        cond3 = ((Sum(self.close, 8)/8 + Std(self.close, 8)) == Sum(self.close, 2)/2)
        cond4 = (self.volume/Mean(self.volume, 20) >= 1)
        part = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part[cond1] = -1
        part[cond2] = 1
        part[cond3] = -1
        part[cond3 & cond4] = 1
        return part
    
    def alpha005(self):
        """(-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))"""
        return -1 * Tsmax(Corr(Tsrank(self.volume, 5), Tsrank(self.high, 5), 5), 3)
    
    def alpha006(self):
        """(RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4))) * -1)"""
        return -1 * Rank(Sign(Delta((self.open * 0.85 + self.high * 0.15), 4)))
    
    def alpha007(self):
        """((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))"""
        return (
            (Rank(Tsmax(self.vwap - self.close, 3)) + 
             Rank(Tsmin(self.vwap - self.close, 3))) * 
            Rank(Delta(self.volume, 3))
        )
    
    def alpha008(self):
        """RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)"""
        return Rank(Delta(((self.high + self.low) / 2 * 0.2 + self.vwap * 0.8), 4) * -1)
    
    def alpha009(self):
        """SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)"""
        return Sma(
            ((self.high + self.low) / 2 - 
             (Delay(self.high, 1) + Delay(self.low, 1)) / 2) *
            (self.high - self.low) / self.volume,
            7, 2
        )
    
    def alpha010(self):
        cond = (self.returns < 0)
        part = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part[cond] = Std(self.returns, 20)
        part[~cond] = self.close
        part = part**2
        return Rank(Tsmax(part, 5))
    
    def alpha011(self):
        """SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,6)"""
        return Sum(((self.close-self.low)-(self.high-self.close))/(self.high-self.low)*self.volume, 6)
    
    def alpha012(self):
        """(RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))"""
        return (
            Rank(self.open - Sum(self.vwap, 10) / 10) * 
            (-1 * Rank(Abs(self.close - self.vwap)))
        )
    
    def alpha013(self):
        """(((HIGH * LOW)^0.5) - VWAP)"""
        return (self.high * self.low) ** 0.5 - self.vwap
    
    def alpha014(self):
        """CLOSE - DELAY(CLOSE, 5)"""
        return self.close - Delay(self.close, 5)
    
    def alpha015(self):
        """OPEN / DELAY(CLOSE, 1) - 1"""
        return self.open / Delay(self.close, 1) - 1
    
    def alpha016(self):
        """(-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))"""
        return -1 * Tsmax(Rank(Corr(Rank(self.volume), Rank(self.vwap), 5)), 5)
    
    def alpha017(self):
        """RANK((VWAP - MAX(VWAP, 15)))^DELTA(CLOSE, 5)"""
        return Rank(self.vwap - Tsmax(self.vwap, 15)) ** Delta(self.close, 5)
    
    def alpha018(self):
        """CLOSE/DELAY(CLOSE,5)"""
        return self.close / Delay(self.close, 5)
    
    def alpha019(self):
        cond1 = (self.close < Delay(self.close, 5))
        cond2 = (self.close == Delay(self.close, 5))
        cond3 = (self.close > Delay(self.close, 5))
        part = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part[cond1] = (self.close - Delay(self.close, 5)) / Delay(self.close, 5)
        part[cond2] = 0
        part[cond3] = (self.close - Delay(self.close, 5)) / self.close
        return part
    
    def alpha020(self):
        """(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100"""
        return (self.close - Delay(self.close, 6)) / Delay(self.close, 6) * 100
    
    def alpha021(self):
        """REGBETA(MEAN(CLOSE,6),SEQUENCE(6))"""
        return Regbeta(Mean(self.close, 6), Sequence(6))


    def alpha022(self):
        """SMA(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)"""
        return Sma(((self.close-Mean(self.close,6))/Mean(self.close,6)-Delay((self.close-Mean(self.close,6))/Mean(self.close,6),3)),12,1)

    def alpha023(self):
        cond = (self.close > Delay(self.close,1))
        part1 = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part1[cond] = Std(self.close,20)
        part1[~cond] = 0
        part2 = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part2[~cond] = Std(self.close,20)
        part2[cond] = 0
        return 100*Sma(part1,20,1)/(Sma(part1,20,1) + Sma(part2,20,1))

    def alpha024(self):
        """SMA(CLOSE-DELAY(CLOSE,5),5,1)"""
        return Sma(self.close-Delay(self.close,5),5,1)

    def alpha025(self):
        """((-1 * RANK((DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR((VOLUME / MEAN(VOLUME,20)), 9)))))) * (1 + RANK(SUM(RET, 250))))"""
        return ((-1 * Rank((Delta(self.close, 7) * (1 - Rank(Decaylinear((self.volume / Mean(self.volume,20)), 9)))))) * (1 + Rank(Sum(self.returns, 250))))

    def alpha026(self):
        """((((SUM(CLOSE, 7) / 7) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE, 5), 230))))"""
        return ((((Sum(self.close, 7) / 7) - self.close)) + ((Corr(self.vwap, Delay(self.close, 5), 230))))

    def alpha027(self):
        """WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)"""
        A = (self.close-Delay(self.close,3))/Delay(self.close,3)*100+(self.close-Delay(self.close,6))/Delay(self.close,6)*100
        return Wma(A, 12)

    def alpha028(self):
        """3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(MAX(HIGH,9)-TSMAX(LOW,9))*100,3,1),3,1)"""
        #return 3*Sma((self.close-Tsmin(self.low,9))/(Tsmax(self.high,9)-Tsmin(self.low,9))*100,3,1)-2*Sma(Sma((self.close-Tsmin(self.low,9))/(Tsmax(self.high,9)-Tsmax(self.low,9))*100,3,1),3,1)
        return 3*Sma((self.close-Tsmin(self.low,9))/(Tsmax(self.high,9)-Tsmin(self.low,9))*100,3,1)-2*Sma(Sma((self.close-Tsmin(self.low,9))/(Tsmax(self.high,9)-Tsmin(self.low,9))*100,3,1),3,1)


    def alpha029(self):
        """(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME"""
        return (self.close-Delay(self.close,6))/Delay(self.close,6)*self.volume

    def alpha030(self):
        """Non implementato - richiede dati di mercato esterni"""
        return pd.DataFrame(0, index=self.close.index, columns=self.close.columns)

    def alpha031(self):
        """(CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100"""
        return (self.close-Mean(self.close,12))/Mean(self.close,12)*100

    def alpha032(self):
        """(-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))"""
        return (-1 * Sum(Rank(Corr(Rank(self.high), Rank(self.volume), 3)), 3))

    def alpha033(self):
        """((((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) * RANK(((SUM(RET, 240) - SUM(RET, 20)) / 220))) *TSRANK(VOLUME, 5))"""
        return ((((-1 * Tsmin(self.low, 5)) + Delay(Tsmin(self.low, 5), 5)) * Rank(((Sum(self.returns, 240) - Sum(self.returns, 20)) / 220))) *Tsrank(self.volume, 5))

    def alpha034(self):
        """MEAN(CLOSE,12)/CLOSE"""
        return Mean(self.close,12)/self.close

    def alpha035(self):
        """(MIN(RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)), RANK(DECAYLINEAR(CORR((VOLUME), ((OPEN * 0.65) +(OPEN *0.35)), 17),7))) * -1)"""
        return (Min(Rank(Decaylinear(Delta(self.open, 1), 15)), Rank(Decaylinear(Corr((self.volume), ((self.open * 0.65) +(self.open *0.35)), 17),7))) * -1)

    def alpha036(self):
        """RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP),6), 2))"""
        return Rank(Sum(Corr(Rank(self.volume), Rank(self.vwap),6 ), 2))

    def alpha037(self):
        """(-1 * RANK(((SUM(OPEN, 5) * SUM(RET, 5)) - DELAY((SUM(OPEN, 5) * SUM(RET, 5)), 10))))"""
        return (-1 * Rank(((Sum(self.open, 5) * Sum(self.returns, 5)) - Delay((Sum(self.open, 5) * Sum(self.returns, 5)), 10))))

    def alpha038(self):
        cond = ((Sum(self.high, 20) / 20) < self.high)
        part = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part[cond] = -1 * Delta(self.high, 2)
        part[~cond] = 0
        return part

    def alpha039(self):
        """((RANK(DECAYLINEAR(DELTA((CLOSE), 2),8)) - RANK(DECAYLINEAR(CORR(((VWAP * 0.3) + (OPEN * 0.7)),SUM(MEAN(VOLUME,180), 37), 14), 12))) * -1)"""
        return ((Rank(Decaylinear(Delta((self.close), 2),8)) - Rank(Decaylinear(Corr(((self.vwap * 0.3) + (self.open * 0.7)),Sum(Mean(self.volume,180), 37), 14), 12))) * -1)

    def alpha040(self):
        cond = (self.close > Delay(self.close,1))
        part1 = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part1[cond] = self.volume
        part1[~cond] = 0
        part2 = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part2[~cond] = self.volume
        part2[cond] = 0
        return Sum(part1,26)/Sum(part2,26)*100

    def alpha041(self):
        """(RANK(MAX(DELTA((VWAP), 3), 5))* -1)"""
        return (Rank(Tsmax(Delta((self.vwap), 3), 5))* -1)

    def alpha042(self):
        """((-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10))"""
        return ((-1 * Rank(Std(self.high, 10))) * Corr(self.high, self.volume, 10))

    def alpha043(self):
        cond1 = (self.close > Delay(self.close,1))
        cond2 = (self.close < Delay(self.close,1))
        cond3 = (self.close == Delay(self.close,1))
        part = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part[cond1] = self.volume
        part[cond2] = -self.volume
        part[cond3] = 0
        return Sum(part,6)

    def alpha044(self):
        """(TSRANK(DECAYLINEAR(CORR(((LOW )), MEAN(VOLUME,10), 7), 6),4) + TSRANK(DECAYLINEAR(DELTA((VWAP),3), 10), 15))"""
        return (Tsrank(Decaylinear(Corr(((self.low)), Mean(self.volume,10), 7), 6),4) + Tsrank(Decaylinear(Delta((self.vwap),3), 10), 15))

    def alpha045(self):
        """(RANK(DELTA((((CLOSE * 0.6) + (OPEN *0.4))), 1)) * RANK(CORR(VWAP, MEAN(VOLUME,150), 15)))"""
        return (Rank(Delta((((self.close * 0.6) + (self.open *0.4))), 1)) * Rank(Corr(self.vwap, Mean(self.volume,150), 15)))

    def alpha046(self):
        """(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)"""
        return (Mean(self.close,3)+Mean(self.close,6)+Mean(self.close,12)+Mean(self.close,24))/(4*self.close)

    def alpha047(self):
        """SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)"""
        return Sma((Tsmax(self.high,6)-self.close)/(Tsmax(self.high,6)-Tsmin(self.low,6))*100,9,1)

    def alpha048(self):
        """(-1*((RANK(((SIGN((CLOSE - DELAY(CLOSE, 1))) + SIGN((DELAY(CLOSE, 1) - DELAY(CLOSE, 2)))) + SIGN((DELAY(CLOSE, 2) - DELAY(CLOSE, 3)))))) * SUM(VOLUME, 5)) / SUM(VOLUME, 20))"""
        return (-1*((Rank(((Sign((self.close - Delay(self.close, 1))) + Sign((Delay(self.close, 1) - Delay(self.close, 2)))) + Sign((Delay(self.close, 2) - Delay(self.close, 3)))))) * Sum(self.volume, 5)) / Sum(self.volume, 20))

    def alpha049(self):
        cond = ((self.high + self.low) > (Delay(self.high,1) + Delay(self.low,1)))
        part1 = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part1[cond] = 0
        part1[~cond] = Max(Abs(self.high - Delay(self.high,1)), Abs(self.low - Delay(self.low,1)))
        part2 = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part2[~cond] = 0
        part2[cond] = Max(Abs(self.high - Delay(self.high,1)), Abs(self.low - Delay(self.low,1)))
        return Sum(part1, 12) / (Sum(part1, 12) + Sum(part2, 12))

    def alpha050(self):
        cond = ((self.high + self.low) <= (Delay(self.high,1) + Delay(self.low,1)))
        part1 = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part1[cond] = 0
        part1[~cond] = Max(Abs(self.high - Delay(self.high,1)), Abs(self.low - Delay(self.low,1)))
        part2 = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part2[~cond] = 0
        part2[cond] = Max(Abs(self.high - Delay(self.high,1)), Abs(self.low - Delay(self.low,1)))
        return (Sum(part1, 12) - Sum(part2, 12)) / (Sum(part1, 12) + Sum(part2, 12))

    def alpha051(self):
        """Stesso di alpha049"""
        return self.alpha049()

    def alpha052(self):
        """SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-L),26)*100"""
        return Sum(Max(self.high-Delay((self.high+self.low+self.close)/3,1),0),26)/Sum(Max(Delay((self.high+self.low+self.close)/3,1)-self.low, 0),26)*100

    def alpha053(self):
        """COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100"""
        cond = (self.close > Delay(self.close,1))
        return Count(cond, 12) / 12 * 100

    def alpha054(self):
        """(-1 * RANK((STD(ABS(CLOSE - OPEN)) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN,10)))"""
        return (-1 * Rank(((Abs(self.close - self.open)).std() + (self.close - self.open)) + Corr(self.close, self.open,10)))

    def alpha055(self):
        """Formula complessa con condizioni multiple"""
        A = Abs(self.high - Delay(self.close, 1))
        B = Abs(self.low - Delay(self.close, 1))
        C = Abs(self.high - Delay(self.low, 1))
        cond1 = ((A > B) & (A > C))
        cond2 = ((B > C) & (B > A))
        cond3 = ((C >= A) & (C >= B))
        
        part0 = 16*(self.close + (self.close - self.open)/2 - Delay(self.open,1))
        
        # FIX: Inizializza con float, non con 0
        part1 = pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns, dtype=float)
        part1[cond1] = (Abs(self.high - Delay(self.close, 1)) + 
                        Abs(self.low - Delay(self.close, 1))/2 + 
                        Abs(Delay(self.close, 1)-Delay(self.open, 1))/4)
        part1[cond2] = (Abs(self.low - Delay(self.close, 1)) + 
                        Abs(self.high - Delay(self.close, 1))/2 + 
                        Abs(Delay(self.close, 1)-Delay(self.open, 1))/4)
        part1[cond3] = (Abs(self.high - Delay(self.low, 1)) + 
                        Abs(Delay(self.close, 1)-Delay(self.open, 1))/4)
        
        part2 = Max(Abs(self.high-Delay(self.close,1)), Abs(self.low-Delay(self.close,1)))
        
        return Sum(part0/part1*part2, 20)

    def alpha056(self):
        A = Rank((self.open - Tsmin(self.open, 12)))
        B = Rank((Rank(Corr(Sum(((self.high + self.low) / 2), 19),Sum(Mean(self.volume,40), 19), 13))**5))
        cond = (A < B)
        part = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part[cond] = 1
        part[~cond] = 0
        return part

    def alpha057(self):
        """SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)"""
        return Sma((self.close-Tsmin(self.low,9))/(Tsmax(self.high,9)-Tsmin(self.low,9))*100,3,1)

    def alpha058(self):
        """COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100"""
        cond = (self.close > Delay(self.close,1))
        return Count(cond,20)/20*100

    def alpha059(self):
        cond1 = (self.close == Delay(self.close,1))
        cond2 = (self.close > Delay(self.close,1))
        cond3 = (self.close < Delay(self.close,1))
        part = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part[cond1] = 0
        part[cond2] = self.close - Min(self.low,Delay(self.close,1))
        part[cond3] = self.close - Max(self.low,Delay(self.close,1))
        return Sum(part, 20)


    def alpha060(self):
        """SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,20)"""
        return Sum(((self.close-self.low)-(self.high-self.close))/(self.high-self.low)*self.volume,20)


    def alpha061(self):
        """(MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12)),RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(VOLUME,80), 8)), 17))) * -1)"""
        return (Max(Rank(Decaylinear(Delta(self.vwap, 1), 12)),
                    Rank(Decaylinear(Rank(Corr((self.low),Mean(self.volume,80), 8)), 17))) * -1)

    def alpha062(self):
        """(-1 * CORR(HIGH, RANK(VOLUME), 5))"""
        return (-1 * Corr(self.high, Rank(self.volume), 5))

    def alpha063(self):
        """SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100"""
        return Sma(Max(self.close-Delay(self.close,1),0),6,1)/Sma(Abs(self.close-Delay(self.close,1)),6,1)*100

    def alpha064(self):
        """(MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 4), 4)),RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(VOLUME,60)), 4), 13), 14))) * -1)"""
        return (Max(Rank(Decaylinear(Corr(Rank(self.vwap), Rank(self.volume), 4), 4)),
                    Rank(Decaylinear(Tsmax(Corr(Rank(self.close), Rank(Mean(self.volume,60)), 4), 13), 14))) * -1)

    def alpha065(self):
        """MEAN(CLOSE,6)/CLOSE"""
        return Mean(self.close,6)/self.close

    def alpha066(self):
        """(CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100"""
        return (self.close-Mean(self.close,6))/Mean(self.close,6)*100

    def alpha067(self):
        """SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100"""
        return Sma(Max(self.close-Delay(self.close,1),0),24,1)/Sma(Abs(self.close-Delay(self.close,1)),24,1)*100

    def alpha068(self):
        """SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)"""
        return Sma(((self.high+self.low)/2-(Delay(self.high,1)+Delay(self.low,1))/2)*(self.high-self.low)/self.volume,15,2)

    def alpha069(self):
        cond1 = (self.open <= Delay(self.open,1))
        cond2 = (self.open >= Delay(self.open,1))
        
        DTM = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        DTM[cond1] = 0
        DTM[~cond1] = Max((self.high-self.open),(self.open-Delay(self.open,1)))
        
        DBM = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        DBM[cond2] = 0
        DBM[~cond2] = Max((self.open-self.low),(self.open-Delay(self.open,1)))
        
        cond3 = (Sum(DTM,20) > Sum(DBM,20))
        cond4 = (Sum(DTM,20)== Sum(DBM,20))
        cond5 = (Sum(DTM,20) < Sum(DBM,20))
        
        part = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part[cond3] = (Sum(DTM,20)-Sum(DBM,20))/Sum(DTM,20)
        part[cond4] = 0
        part[cond5] = (Sum(DTM,20)-Sum(DBM,20))/Sum(DBM,20)
        return part

    def alpha070(self):
        """STD(AMOUNT,6)"""
        return Std(self.amount,6)

    def alpha071(self):
        """(CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100"""
        return (self.close-Mean(self.close,24))/Mean(self.close,24)*100

    def alpha072(self):
        """SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)"""
        return Sma((Tsmax(self.high,6)-self.close)/(Tsmax(self.high,6)-Tsmin(self.low,6))*100,15,1)

    def alpha073(self):
        """((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE), VOLUME, 10), 16), 4), 5) - RANK(DECAYLINEAR(CORR(VWAP, MEAN(VOLUME,30), 4),3))) * -1)"""
        return ((Tsrank(Decaylinear(Decaylinear(Corr((self.close), self.volume, 10), 16), 4), 5) - 
                 Rank(Decaylinear(Corr(self.vwap, Mean(self.volume,30), 4),3))) * -1)

    def alpha074(self):
        """(RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20), SUM(MEAN(VOLUME,40), 20), 7)) + RANK(CORR(RANK(VWAP), RANK(VOLUME), 6)))"""
        return (Rank(Corr(Sum(((self.low * 0.35) + (self.vwap * 0.65)), 20), Sum(Mean(self.volume,40), 20), 7)) + 
                Rank(Corr(Rank(self.vwap), Rank(self.volume), 6)))

    def alpha075(self):
        """Richiede benchmark - ritorna 0"""
        return pd.DataFrame(0, index=self.close.index, columns=self.close.columns)

    def alpha076(self):
        """STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)"""
        return Std(Abs((self.close/Delay(self.close,1)-1))/self.volume,20)/Mean(Abs((self.close/Delay(self.close,1)-1))/self.volume,20)

    def alpha077(self):
        """MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH) - (VWAP + HIGH)), 20)),RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 3), 6)))"""
        return Min(Rank(Decaylinear(((((self.high + self.low) / 2) + self.high) - (self.vwap + self.high)), 20)),
                   Rank(Decaylinear(Corr(((self.high + self.low) / 2), Mean(self.volume,40), 3), 6)))

    def alpha078(self):
        """((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))"""
        return ((self.high+self.low+self.close)/3-Mean((self.high+self.low+self.close)/3,12))/(0.015*Mean(Abs(self.close-Mean((self.high+self.low+self.close)/3,12)),12))

    def alpha079(self):
        """SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100"""
        return Sma(Max(self.close-Delay(self.close,1),0),12,1)/Sma(Abs(self.close-Delay(self.close,1)),12,1)*100

    def alpha080(self):
        """(VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100"""
        return (self.volume-Delay(self.volume,5))/Delay(self.volume,5)*100

    def alpha081(self):
        """SMA(VOLUME,21,2)"""
        return Sma(self.volume,21,2)

    def alpha082(self):
        """SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)"""
        return Sma((Tsmax(self.high,6)-self.close)/(Tsmax(self.high,6)-Tsmin(self.low,6))*100,20,1)

    def alpha083(self):
        """(-1 * RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), 5)))"""
        return (-1 * Rank(Cov(Rank(self.high), Rank(self.volume), 5)))

    def alpha084(self):
        cond1 = (self.close > Delay(self.close,1))
        cond2 = (self.close < Delay(self.close,1))
        cond3 = (self.close == Delay(self.close,1))
        part = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part[cond1] = self.volume
        part[cond2] = 0
        part[cond3] = -self.volume
        return Sum(part, 20)

    def alpha085(self):
        """(TSRANK((VOLUME / MEAN(VOLUME,20)), 20) * TSRANK((-1 * DELTA(CLOSE, 7)), 8))"""
        return (Tsrank((self.volume / Mean(self.volume,20)), 20) * Tsrank((-1 * Delta(self.close, 7)), 8))

    def alpha086(self):
        A = (((Delay(self.close, 20) - Delay(self.close, 10)) / 10) - ((Delay(self.close, 10) - self.close) / 10))
        cond1 = (A > 0.25)
        cond2 = (A < 0.0)
        cond3 = ((0 <= A) & (A <= 0.25))
        part = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part[cond1] = -1
        part[cond2] = 1
        part[cond3] = -1*(self.close - Delay(self.close, 1))
        return part

    def alpha087(self):
        """((RANK(DECAYLINEAR(DELTA(VWAP, 4), 7)) + TSRANK(DECAYLINEAR(((((LOW * 0.9) + (LOW * 0.1)) - VWAP) /(OPEN - ((HIGH + LOW) / 2))), 11), 7)) * -1)"""
        return ((Rank(Decaylinear(Delta(self.vwap, 4), 7)) + 
                 Tsrank(Decaylinear(((((self.low * 0.9) + (self.low * 0.1)) - self.vwap) /
                                     (self.open - ((self.high + self.low) / 2))), 11), 7)) * -1)

    def alpha088(self):
        """(CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100"""
        return (self.close-Delay(self.close,20))/Delay(self.close,20)*100

    def alpha089(self):
        """2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))"""
        return 2*(Sma(self.close,13,2)-Sma(self.close,27,2)-Sma(Sma(self.close,13,2)-Sma(self.close,27,2),10,2))

    def alpha090(self):
        """(RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1)"""
        return (Rank(Corr(Rank(self.vwap), Rank(self.volume), 5)) * -1)

    def alpha091(self):
        """((RANK((CLOSE - MAX(CLOSE, 5)))*RANK(CORR((MEAN(VOLUME,40)), LOW, 5))) * -1)"""
        return ((Rank((self.close - Tsmax(self.close, 5)))*Rank(Corr((Mean(self.volume,40)), self.low, 5))) * -1)

    def alpha092(self):
        """(MAX(RANK(DECAYLINEAR(DELTA(((CLOSE * 0.35) + (VWAP *0.65)), 2), 3)),TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)), CLOSE, 13)), 5), 15)) * -1)"""
        return (Max(Rank(Decaylinear(Delta(((self.close * 0.35) + (self.vwap *0.65)), 2), 3)),
                    Tsrank(Decaylinear(Abs(Corr((Mean(self.volume,180)), self.close, 13)), 5), 15)) * -1)

    def alpha093(self):
        cond = (self.open >= Delay(self.open,1))
        part = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part[cond] = 0
        part[~cond] = Max((self.open-self.low),(self.open-Delay(self.open,1)))
        return Sum(part, 20)

    def alpha094(self):
        cond1 = (self.close > Delay(self.close,1))
        cond2 = (self.close < Delay(self.close,1))
        cond3 = (self.close == Delay(self.close,1))
        part = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part[cond1] = self.volume
        part[cond2] = -1*self.volume
        part[cond3] = 0
        return Sum(part, 30)

    def alpha095(self):
        """STD(AMOUNT,20)"""
        return Std(self.amount,20)

    def alpha096(self):
        """SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)"""
        return Sma(Sma((self.close-Tsmin(self.low,9))/(Tsmax(self.high,9)-Tsmin(self.low,9))*100,3,1),3,1)

    def alpha097(self):
        """STD(VOLUME,10)"""
        return Std(self.volume,10)

    def alpha098(self):
        cond = (Delta(Sum(self.close,100)/100, 100)/Delay(self.close, 100) <= 0.05)
        part = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part[cond] = -1 * (self.close - Tsmin(self.close, 100))
        part[~cond] = -1 * Delta(self.close, 3)
        return part

    def alpha099(self):
        """(-1 * Rank(Cov(Rank(self.close), Rank(self.volume), 5)))"""
        return (-1 * Rank(Cov(Rank(self.close), Rank(self.volume), 5)))

    def alpha100(self):
        """Std(self.volume,20)"""
        return Std(self.volume,20)

    def alpha101(self):
        rank1 = Rank(Corr(self.close, Sum(Mean(self.volume,30), 37), 15))
        rank2 = Rank(Corr(Rank(((self.high * 0.1) + (self.vwap * 0.9))),Rank(self.volume), 11))
        cond = (rank1<rank2)
        part = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part[cond] = 1
        part[~cond] = 0
        return part

    def alpha102(self):
        """SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100"""
        return Sma(Max(self.volume-Delay(self.volume,1),0),6,1)/Sma(Abs(self.volume-Delay(self.volume,1)),6,1)*100

    def alpha103(self):
        """((20-LOWDAY(LOW,20))/20)*100"""
        return ((20-Lowday(self.low,20))/20)*100

    def alpha104(self):
        """(-1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20))))"""
        return (-1 * (Delta(Corr(self.high, self.volume, 5), 5) * Rank(Std(self.close, 20))))

    def alpha105(self):
        """(-1 * CORR(RANK(OPEN), RANK(VOLUME), 10))"""
        return (-1 * Corr(Rank(self.open), Rank(self.volume), 10))

    def alpha106(self):
        """CLOSE-DELAY(CLOSE,20)"""
        return self.close-Delay(self.close,20)

    def alpha107(self):
        """(((-1 * RANK((OPEN - DELAY(HIGH, 1)))) * RANK((OPEN - DELAY(CLOSE, 1)))) * RANK((OPEN - DELAY(LOW, 1))))"""
        return (((-1 * Rank((self.open - Delay(self.high, 1)))) * Rank((self.open - Delay(self.close, 1)))) * Rank((self.open - Delay(self.low, 1))))

    def alpha108(self):
        """((RANK((HIGH - MIN(HIGH, 2)))^RANK(CORR((VWAP), (MEAN(VOLUME,120)), 6))) * -1)"""
        return ((Rank((self.high - Tsmin(self.high, 2)))**Rank(Corr((self.vwap), (Mean(self.volume,120)), 6))) * -1)

    def alpha109(self):
        """SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)"""
        return Sma(self.high-self.low,10,2)/Sma(Sma(self.high-self.low,10,2),10,2)

    def alpha110(self):
        """SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100"""
        return Sum(Max(self.high-Delay(self.close,1),0),20)/Sum(Max(Delay(self.close,1)-self.low,0),20)*100

    def alpha111(self):
        """SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)"""
        return Sma(self.volume*((self.close-self.low)-(self.high-self.close))/(self.high-self.low),11,2)-Sma(self.volume*((self.close-self.low)-(self.high-self.close))/(self.high-self.low),4,2)

    def alpha112(self):
        cond = (self.close-Delay(self.close,1) > 0)
        part1 = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part1[cond] = self.close-Delay(self.close,1)
        part1[~cond] = 0
        part2 = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part2[~cond] = Abs(self.close-Delay(self.close,1))
        part2[cond] = 0
        return (Sum(part1,12) - Sum(part2,12))/(Sum(part1,12) + Sum(part2,12))*100

    def alpha113(self):
        """(-1 * ((RANK((SUM(DELAY(CLOSE, 5), 20) / 20)) * CORR(CLOSE, VOLUME, 2)) * RANK(CORR(SUM(CLOSE, 5),SUM(CLOSE, 20), 2))))"""
        return (-1 * ((Rank((Sum(Delay(self.close, 5), 20) / 20)) * Corr(self.close, self.volume, 2)) * Rank(Corr(Sum(self.close, 5),Sum(self.close, 20), 2))))

    def alpha114(self):
        """((RANK(DELAY(((HIGH - LOW) / (SUM(CLOSE, 5) / 5)), 2)) * RANK(RANK(VOLUME))) / (((HIGH - LOW) /(SUM(CLOSE, 5) / 5)) / (VWAP - CLOSE)))"""
        return ((Rank(Delay(((self.high - self.low) / (Sum(self.close, 5) / 5)), 2)) * Rank(Rank(self.volume))) / (((self.high - self.low) /(Sum(self.close, 5) / 5)) / (self.vwap - self.close)))

    def alpha115(self):
        """(RANK(CORR(((HIGH * 0.9) + (CLOSE * 0.1)), MEAN(VOLUME,30), 10))^RANK(CORR(TSRANK(((HIGH + LOW) /2), 4), TSRANK(VOLUME, 10), 7)))"""
        return (Rank(Corr(((self.high * 0.9) + (self.close * 0.1)), Mean(self.volume,30), 10))**Rank(Corr(Tsrank(((self.high + self.low) /2), 4), Tsrank(self.volume, 10), 7)))

    def alpha116(self):
        """REGBETA(CLOSE,SEQUENCE,20)"""
        return Regbeta(self.close, Sequence(20))

    def alpha117(self):
        """((TSRANK(VOLUME, 32) * (1 - TSRANK(((CLOSE + HIGH) - LOW), 16))) * (1 - TSRANK(RET, 32)))"""
        return ((Tsrank(self.volume, 32) * (1 - Tsrank(((self.close + self.high) - self.low), 16))) * (1 - Tsrank(self.returns, 32)))

    def alpha118(self):
        """SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100"""
        return Sum(self.high-self.open,20)/Sum(self.open-self.low,20)*100

    def alpha119(self):
        """(RANK(DECAYLINEAR(CORR(VWAP, SUM(MEAN(VOLUME,5), 26), 5), 7)) - RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN), RANK(MEAN(VOLUME,15)), 21), 9), 7), 8)))"""
        return (Rank(Decaylinear(Corr(self.vwap, Sum(Mean(self.volume,5), 26), 5), 7)) - Rank(Decaylinear(Tsrank(Tsmin(Corr(Rank(self.open), Rank(Mean(self.volume,15)), 21), 9), 7), 8)))

    def alpha120(self):
        """(RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))"""
        return (Rank((self.vwap - self.close)) / Rank((self.vwap + self.close)))

    def alpha121(self):
        """((RANK((VWAP - MIN(VWAP, 12)))^TSRANK(CORR(TSRANK(VWAP, 20), TSRANK(MEAN(VOLUME,60), 2), 18), 3)) *-1)"""
        return ((Rank((self.vwap - Tsmin(self.vwap, 12)))**Tsrank(Corr(Tsrank(self.vwap, 20), Tsrank(Mean(self.volume,60), 2), 18), 3)) *-1)

    def alpha122(self):
        """(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))/DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)"""
        return (Sma(Sma(Sma(Log(self.close),13,2),13,2),13,2)-Delay(Sma(Sma(Sma(Log(self.close),13,2),13,2),13,2),1))/Delay(Sma(Sma(Sma(Log(self.close),13,2),13,2),13,2),1)

    def alpha123(self):
        A = Rank(Corr(Sum(((self.high + self.low) / 2), 20), Sum(Mean(self.volume,60), 20), 9))
        B = Rank(Corr(self.low, self.volume,6))
        cond = (A < B)
        part = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part[cond] = -1
        part[~cond] = 0
        return part

    def alpha124(self):
        """(CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)),2)"""
        return (self.close - self.vwap) / Decaylinear(Rank(Tsmax(self.close, 30)),2)

    def alpha125(self):
        """(RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME,80),17), 20)) / RANK(DECAYLINEAR(DELTA(((CLOSE * 0.5) + (VWAP * 0.5)), 3), 16)))"""
        return (Rank(Decaylinear(Corr((self.vwap), Mean(self.volume,80),17), 20)) / Rank(Decaylinear(Delta(((self.close * 0.5) + (self.vwap * 0.5)), 3), 16)))

    def alpha126(self):
        """(CLOSE+HIGH+LOW)/3"""
        return (self.close+self.high+self.low)/3

    def alpha127(self):
        """(MEAN((100*(CLOSE-MAX(CLOSE,12))/(MAX(CLOSE,12)))^2),12)^(1/2)"""
        return (Mean((100*(self.close-Tsmax(self.close,12))/(Tsmax(self.close,12)))**2,12))**(1/2)

    def alpha128(self):
        A = (self.high+self.low+self.close)/3
        cond = (A > Delay(A,1))
        part1 = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part1[cond] = A*self.volume
        part1[~cond] = 0
        part2 = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part2[~cond] = A*self.volume
        part2[cond] = 0
        return 100-(100/(1+Sum(part1,14)/Sum(part2,14)))

    def alpha129(self):
        cond = ((self.close-Delay(self.close,1)) < 0)
        part = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part[cond] = Abs(self.close-Delay(self.close,1))
        part[~cond] = 0
        return Sum(part, 12)

    def alpha130(self):
        """(RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 9), 10)) / RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 7),3)))"""
        return (Rank(Decaylinear(Corr(((self.high + self.low) / 2), Mean(self.volume,40), 9), 10)) / Rank(Decaylinear(Corr(Rank(self.vwap), Rank(self.volume), 7),3)))

    def alpha131(self):
        """(RANK(DELAT(VWAP, 1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50), 18), 18))"""
        return (Rank(Delta(self.vwap, 1))**Tsrank(Corr(self.close,Mean(self.volume,50), 18), 18))

    def alpha132(self):
        """MEAN(AMOUNT,20)"""
        return Mean(self.amount,20)

    def alpha133(self):
        """((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100"""
        return ((20-Highday(self.high,20))/20)*100-((20-Lowday(self.low,20))/20)*100

    def alpha134(self):
        """(CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME"""
        return (self.close-Delay(self.close,12))/Delay(self.close,12)*self.volume

    def alpha135(self):
        """SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)"""
        return Sma(Delay(self.close/Delay(self.close,20),1),20,1)

    def alpha136(self):
        """((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10))"""
        return ((-1 * Rank(Delta(self.returns, 3))) * Corr(self.open, self.volume, 10))

    def alpha137(self):
        A = Abs(self.high- Delay(self.close,1))
        B = Abs(self.low - Delay(self.close,1))
        C = Abs(self.high- Delay(self.low,1))
        D = Abs(Delay(self.close,1)-Delay(self.open,1))
        
        cond1 = ((A>B) & (A>C))
        cond2 = ((B>C) & (B>A))
        cond3 = ~cond1 & ~cond2
        
        part0 = 16*(self.close + (self.close - self.open)/2 - Delay(self.open,1))
        part1 = pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns, dtype=float)
        part1[cond1] = A + B/2 + D/4
        part1[cond2] = B + A/2 + D/4
        part1[cond3] = C + D/4
        part1.replace({0: np.nan}, inplace=True)
        
        return part0/part1*Max(A,B)

    def alpha138(self):
        """((RANK(DECAYLINEAR(DELTA((((LOW * 0.7) + (VWAP *0.3))), 3), 20)) - TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8), TSRANK(MEAN(VOLUME,60), 17), 5), 19), 16), 7)) * -1)"""
        return ((Rank(Decaylinear(Delta((((self.low * 0.7) + (self.vwap *0.3))), 3), 20)) - Tsrank(Decaylinear(Tsrank(Corr(Tsrank(self.low, 8), Tsrank(Mean(self.volume,60), 17), 5), 19), 16), 7)) * -1)

    def alpha139(self):
        """(-1 * CORR(OPEN, VOLUME, 10))"""
        return (-1 * Corr(self.open, self.volume, 10))

    def alpha140(self):
        """MIN(RANK(DECAYLINEAR(((RANK(OPEN) + RANK(LOW)) - (RANK(HIGH) + RANK(CLOSE))), 8)), TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE, 8), TSRANK(MEAN(VOLUME,60), 20), 8), 7), 3))"""
        return Min(Rank(Decaylinear(((Rank(self.open) + Rank(self.low)) - (Rank(self.high) + Rank(self.close))), 8)), Tsrank(Decaylinear(Corr(Tsrank(self.close, 8), Tsrank(Mean(self.volume,60), 20), 8), 7), 3))

    def alpha141(self):
        """(RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME,15)), 9))* -1)"""
        return (Rank(Corr(Rank(self.high), Rank(Mean(self.volume,15)), 9))* -1)

    def alpha142(self):
        """(((-1 * RANK(TSRANK(CLOSE, 10))) * RANK(DELTA(DELTA(CLOSE, 1), 1))) * RANK(TSRANK((VOLUME/MEAN(VOLUME,20)), 5)))"""
        return (((-1 * Rank(Tsrank(self.close, 10))) * Rank(Delta(Delta(self.close, 1), 1))) * Rank(Tsrank((self.volume/Mean(self.volume,20)), 5)))

    def alpha143(self):
        """Non implementabile"""
        return pd.DataFrame(0, index=self.close.index, columns=self.close.columns)

    def alpha144(self):
        cond = (self.close<Delay(self.close,1))
        part1 = Abs(self.close/Delay(self.close,1)-1)/self.amount
        return Sumif(part1,20,cond)/Count(cond,20)

    def alpha145(self):
        """(MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100"""
        return (Mean(self.volume,9)-Mean(self.volume,26))/Mean(self.volume,12)*100

    def alpha146(self):
        """MEAN((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2),20)*((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2))/SMA(((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2)))^2,61,2)"""
        return Mean((self.close-Delay(self.close,1))/Delay(self.close,1)-Sma((self.close-Delay(self.close,1))/Delay(self.close,1),61,2),20)*((self.close-Delay(self.close,1))/Delay(self.close,1)-Sma((self.close-Delay(self.close,1))/Delay(self.close,1),61,2))/Sma(((self.close-Delay(self.close,1))/Delay(self.close,1)-((self.close-Delay(self.close,1))/Delay(self.close,1)-Sma((self.close-Delay(self.close,1))/Delay(self.close,1),61,2)))**2,61,2)

    def alpha147(self):
        """REGBETA(MEAN(CLOSE,12),SEQUENCE(12))"""
        return Regbeta(Mean(self.close, 12), Sequence(12))

    def alpha148(self):
        cond = (Rank(Corr((self.open), Sum(Mean(self.volume,60), 9), 6)) < Rank((self.open - Tsmin(self.open, 14))))
        part = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part[cond] = -1
        part[~cond] = 0
        return part

    def alpha149(self):
        """Richiede dati benchmark - ritorna 0"""
        return pd.DataFrame(0, index=self.close.index, columns=self.close.columns)

    def alpha150(self):
        """(CLOSE+HIGH+LOW)/3*VOLUME"""
        return (self.close+self.high+self.low)/3*self.volume

    def alpha151(self):
        """SMA(CLOSE-DELAY(CLOSE,20),20,1)"""
        return Sma(self.close-Delay(self.close,20),20,1)

    def alpha152(self):
        """SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26),9,1)"""
        return Sma(Mean(Delay(Sma(Delay(self.close/Delay(self.close,9),1),9,1),1),12)-Mean(Delay(Sma(Delay(self.close/Delay(self.close,9),1),9,1),1),26),9,1)

    def alpha153(self):
        """(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4"""
        return (Mean(self.close,3)+Mean(self.close,6)+Mean(self.close,12)+Mean(self.close,24))/4

    def alpha154(self):
        cond = (((self.vwap - Tsmin(self.vwap, 16))) < (Corr(self.vwap, Mean(self.volume,180), 18)))
        part = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part[cond] = 1
        part[~cond] = 0
        return part

    def alpha155(self):
        """SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)"""
        return Sma(self.volume,13,2)-Sma(self.volume,27,2)-Sma(Sma(self.volume,13,2)-Sma(self.volume,27,2),10,2)

    def alpha156(self):
        """(MAX(RANK(DECAYLINEAR(DELTA(VWAP, 5), 3)), RANK(DECAYLINEAR(((DELTA(((OPEN * 0.15) + (LOW *0.85)),2) / ((OPEN * 0.15) + (LOW * 0.85))) * -1), 3))) * -1)"""
        return (Max(Rank(Decaylinear(Delta(self.vwap, 5), 3)), Rank(Decaylinear(((Delta(((self.open * 0.15) + (self.low *0.85)),2) / ((self.open * 0.15) + (self.low * 0.85))) * -1), 3))) * -1)

    def alpha157(self):
        """(MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1 * RANK(DELTA((CLOSE - 1), 5))))), 2), 1)))), 1), 5) + TSRANK(DELAY((-1 * RET), 6), 5))"""
        return (Tsmin(Prod(Rank(Rank(Log(Sum(Tsmin(Rank(Rank((-1 * Rank(Delta((self.close - 1), 5))))), 2), 1)))), 1), 5) + Tsrank(Delay((-1 * self.returns), 6), 5))


    def alpha158(self):
        """((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE"""
        return ((self.high-Sma(self.close,15,2))-(self.low-Sma(self.close,15,2)))/self.close

    def alpha159(self):
        """((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6)*12*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),12)*6*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),24)*6*24)*100/(6*12+6*24+12*24)"""
        return ((self.close-Sum(Min(self.low,Delay(self.close,1)),6))/Sum(Max(self.high,Delay(self.close,1))-Min(self.low,Delay(self.close,1)),6)*12*24+(self.close-Sum(Min(self.low,Delay(self.close,1)),12))/Sum(Max(self.high,Delay(self.close,1))-Min(self.low,Delay(self.close,1)),12)*6*24+(self.close-Sum(Min(self.low,Delay(self.close,1)),24))/Sum(Max(self.high,Delay(self.close,1))-Min(self.low,Delay(self.close,1)),24)*6*24)*100/(6*12+6*24+12*24)

    def alpha160(self):
        cond = (self.close<=Delay(self.close,1))
        part = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part[cond] = Std(self.close,20)
        part[~cond] = 0
        return Sma(part, 20, 1)

    def alpha161(self):
        """MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)"""
        return Mean(Max(Max((self.high-self.low),Abs(Delay(self.close,1)-self.high)),Abs(Delay(self.close,1)-self.low)),12)

    def alpha162(self):
        """(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))/(MAX(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12)-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))"""
        return (Sma(Max(self.close-Delay(self.close,1),0),12,1)/Sma(Abs(self.close-Delay(self.close,1)),12,1)*100-Tsmin(Sma(Max(self.close-Delay(self.close,1),0),12,1)/Sma(Abs(self.close-Delay(self.close,1)),12,1)*100,12))/(Sma(Sma(Max(self.close-Delay(self.close,1),0),12,1)/Sma(Abs(self.close-Delay(self.close,1)),12,1)*100,12,1)-Tsmin(Sma(Max(self.close-Delay(self.close,1),0),12,1)/Sma(Abs(self.close-Delay(self.close,1)),12,1)*100,12))

    def alpha163(self):
        """RANK(((((-1 * RET) * MEAN(VOLUME,20)) * VWAP) * (HIGH - CLOSE)))"""
        return Rank(((((-1 * self.returns) * Mean(self.volume,20)) * self.vwap) * (self.high - self.close)))

    def alpha164(self):
        cond = (self.close>Delay(self.close,1))
        part = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part[cond] = 1/(self.close-Delay(self.close,1))
        part[~cond] = 1
        
        part2 = self.high-self.low
        part2.replace({0: np.nan}, inplace=True)
        
        return Sma((part - Tsmin(part,12))/(part2)*100, 13, 2)

    def alpha165(self):
        """Richiede rowmax/rowmin - implementazione semplificata"""
        return pd.DataFrame(0, index=self.close.index, columns=self.close.columns)

    def alpha166(self):
        """Formula problematica - implementazione semplificata"""
        p1 = -20* ( 20-1 )**1.5*Sum(self.close/Delay(self.close,1)-1-Mean(self.close/Delay(self.close,1)-1,20),20)
        p2 = ((20-1)*(20-2)*(Sum(Mean(self.close/Delay(self.close,1),20)**2,20))**1.5)
        return p1/p2

    def alpha167(self):
        cond = (self.close > Delay(self.close,1))
        part = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part[cond] = self.close-Delay(self.close,1)
        part[~cond] = 0
        return Sum(part,12)

    def alpha168(self):
        """(-1*VOLUME/MEAN(VOLUME,20))"""
        return (-1*self.volume/Mean(self.volume,20))

    def alpha169(self):
        """SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),26),10,1)"""
        return Sma(Mean(Delay(Sma(self.close-Delay(self.close,1),9,1),1),12)-Mean(Delay(Sma(self.close-Delay(self.close,1),9,1),1),26),10,1)

    def alpha170(self):
        """((((RANK((1 / CLOSE)) * VOLUME) / MEAN(VOLUME,20)) * ((HIGH * RANK((HIGH - CLOSE))) / (SUM(HIGH, 5) /5))) - RANK((VWAP - DELAY(VWAP, 5))))"""
        return ((((Rank((1 / self.close)) * self.volume) / Mean(self.volume,20)) * ((self.high * Rank((self.high - self.close))) / (Sum(self.high, 5) /5))) - Rank((self.vwap - Delay(self.vwap, 5))))

    def alpha171(self):
        """((-1 * ((LOW - CLOSE) * (OPEN^5))) / ((CLOSE - HIGH) * (CLOSE^5)))"""
        return ((-1 * ((self.low - self.close) * (self.open**5))) / ((self.close - self.high) * (self.close**5)))

    def alpha172(self):
        TR = Max(Max(self.high-self.low,Abs(self.high-Delay(self.close,1))),Abs(self.low-Delay(self.close,1)))
        HD = self.high-Delay(self.high,1)
        LD = Delay(self.low,1)-self.low
        cond1 = ((LD>0) & (LD>HD))
        cond2 = ((HD>0) & (HD>LD))
        part1 = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part1[cond1] = LD
        part1[~cond1] = 0
        part2 = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part2[cond2] = HD
        part2[~cond2] = 0
        return Mean(Abs(Sum(part1,14)*100/Sum(TR,14)-Sum(part2,14)*100/Sum(TR,14))/(Sum(part1,14)*100/Sum(TR,14)+Sum(part2,14)*100/Sum(TR,14))*100,6)

    def alpha173(self):
        """3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)"""
        return 3*Sma(self.close,13,2)-2*Sma(Sma(self.close,13,2),13,2)+Sma(Sma(Sma(Log(self.close),13,2),13,2),13,2)

    def alpha174(self):
        cond = (self.close>Delay(self.close,1))
        part = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part[cond] = Std(self.close,20)
        part[~cond] = 0
        return Sma(part,20,1)

    def alpha175(self):
        """MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)"""
        return Mean(Max(Max((self.high-self.low),Abs(Delay(self.close,1)-self.high)),Abs(Delay(self.close,1)-self.low)),6)

    def alpha176(self):
        """CORR(RANK(((CLOSE - TSMIN(LOW, 12)) / (TSMAX(HIGH, 12) - TSMIN(LOW,12)))), RANK(VOLUME), 6)"""
        return Corr(Rank(((self.close - Tsmin(self.low, 12)) / (Tsmax(self.high, 12) - Tsmin(self.low,12)))), Rank(self.volume), 6)

    def alpha177(self):
        """((20-HIGHDAY(HIGH,20))/20)*100"""
        return ((20-Highday(self.high,20))/20)*100

    def alpha178(self):
        """(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME"""
        return (self.close-Delay(self.close,1))/Delay(self.close,1)*self.volume

    def alpha179(self):
        """(RANK(CORR(VWAP, VOLUME, 4)) *RANK(CORR(RANK(LOW), RANK(MEAN(VOLUME,50)), 12)))"""
        return (Rank(Corr(self.vwap, self.volume, 4)) *Rank(Corr(Rank(self.low), Rank(Mean(self.volume,50)), 12)))

    def alpha180(self):
        cond = (Mean(self.volume,20) < self.volume)
        part = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part[cond] = (-1 * Tsrank(Abs(Delta(self.close, 7)), 60)) * Sign(Delta(self.close, 7))
        part[~cond] = -1 * self.volume
        return part

    def alpha181(self):
        """Richiede benchmark - ritorna 0"""
        return pd.DataFrame(0, index=self.close.index, columns=self.close.columns)

    def alpha182(self):
        """Richiede benchmark - ritorna 0"""
        return pd.DataFrame(0, index=self.close.index, columns=self.close.columns)

    def alpha183(self):
        """Richiede rowmax/rowmin - implementazione semplificata"""
        return pd.DataFrame(0, index=self.close.index, columns=self.close.columns)

    def alpha184(self):
        """(RANK(CORR(DELAY((OPEN - CLOSE), 1), CLOSE, 200)) + RANK((OPEN - CLOSE)))"""
        return (Rank(Corr(Delay((self.open - self.close), 1), self.close, 200)) + Rank((self.open - self.close)))

    def alpha185(self):
        """RANK((-1 * ((1 - (OPEN / CLOSE))^2)))"""
        return Rank((-1 * ((1 - (self.open / self.close))**2)))

    def alpha186(self):
        TR = Max(Max(self.high-self.low,Abs(self.high-Delay(self.close,1))),Abs(self.low-Delay(self.close,1)))
        HD = self.high-Delay(self.high,1)
        LD = Delay(self.low,1)-self.low
        cond1 = ((LD>0) & (LD>HD))
        cond2 = ((HD>0) & (HD>LD))
        part1 = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part1[cond1] = LD
        part1[~cond1] = 0
        part2 = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part2[cond2] = HD
        part2[~cond2] = 0
        return (Mean(Abs(Sum(part1,14)*100/Sum(TR,14)-Sum(part2,14)*100/Sum(TR,14))/(Sum(part1,14)*100/Sum(TR,14)+Sum(part2,14)*100/Sum(TR,14))*100,6)+Delay(Mean(Abs(Sum(part1,14)*100/Sum(TR,14)-Sum(part2,14)*100/Sum(TR,14))/(Sum(part1,14)*100/Sum(TR,14)+Sum(part2,14)*100/Sum(TR,14))*100,6),6))/2

    def alpha187(self):
        cond = (self.open<=Delay(self.open,1))
        part = pd.DataFrame(None, index=self.close.index, columns=self.close.columns)
        part[cond] = 0
        part[~cond] = Max((self.high-self.open),(self.open-Delay(self.open,1)))
        return Sum(part,20)

    def alpha188(self):
        """((HIGH-LOW–SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100"""
        return ((self.high-self.low-Sma(self.high-self.low,11,2))/Sma(self.high-self.low,11,2))*100

    def alpha189(self):
        """MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)"""
        return Mean(Abs(self.close-Mean(self.close,6)),6)

    def alpha190(self):
        """Non implementabile - formula complessa"""
        return pd.DataFrame(0, index=self.close.index, columns=self.close.columns)

    def alpha191(self):
        """((CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE)"""
        return ((Corr(Mean(self.volume,20), self.low, 5) + ((self.high + self.low) / 2)) - self.close)

    
    def calculate_alpha(self, alpha_number, return_long=True):
        """
        Calcola un singolo alpha.
        
        Parameters:
        -----------
        alpha_number : int
            Numero dell'alpha (1-191)
        return_long : bool
            Se True, ritorna formato long, altrimenti wide
            
        Returns:
        --------
        pd.DataFrame
            Alpha calcolato
        """
        alpha_method = getattr(self, f'alpha{alpha_number:03d}')
        result = alpha_method()
        
        if return_long:
            return self.to_long_format(result, f'alpha{alpha_number:03d}')
        return result
    
    def calculate_all_alphas(self, return_long=True):
        """
        Calcola tutti gli alpha implementati.
        
        Parameters:
        -----------
        return_long : bool
            Se True, ritorna formato long unito, altrimenti dict di wide
            
        Returns:
        --------
        pd.DataFrame o dict
            Tutti gli alpha calcolati
        """
        # Trova tutti i metodi alpha
        alpha_methods = sorted([m for m in dir(self) if m.startswith('alpha') and m[5:].isdigit()])
        
        if return_long:
            # Per formato long, dobbiamo fare merge su date e ticker
            results_dict = {}
            
            for method_name in alpha_methods:
                try:
                    alpha_num = int(method_name[5:])
                    alpha_name = f'alpha{alpha_num:03d}'
                    
                    # Calcola l'alpha
                    result = self.calculate_alpha(alpha_num, return_long=True)
                    results_dict[alpha_name] = result.set_index(['date', 'ticker'])[alpha_name]
                    
                    print(f"✓ Alpha {alpha_num:03d} calcolato")
                except Exception as e:
                    print(f"✗ Errore in Alpha {alpha_num:03d}: {str(e)}")
            
            # Concatena tutti gli alpha in un unico DataFrame
            all_alphas_df = pd.DataFrame(results_dict).reset_index()
            return all_alphas_df
        
        else:
            # Per formato wide, ritorna un dizionario
            results = {}
            for method_name in alpha_methods:
                try:
                    alpha_num = int(method_name[5:])
                    alpha_name = f'alpha{alpha_num:03d}'
                    result = self.calculate_alpha(alpha_num, return_long=False)
                    results[alpha_name] = result
                    print(f"✓ Alpha {alpha_num:03d} calcolato")
                except Exception as e:
                    print(f"✗ Errore in Alpha {alpha_num:03d}: {str(e)}")
            return results

    

def _compute_single_alpha(alpha_num, df_data):
    """Funzione standalone per multiprocessing."""
    #from alpha191 import Alphas191
    calc = Alphas191(df_data)
    result = calc.calculate_alpha(alpha_num, return_long=True)
    return alpha_num, result


def calculate_all_alphas_parallel(df_data, skip=None, n_workers=8):
    from tqdm import tqdm
    skip = skip or set()
    alpha_nums = [i for i in range(1, 192) if i not in skip]
    
    results_dict = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_compute_single_alpha, n, df_data): n
            for n in alpha_nums
        }
        for future in tqdm(as_completed(futures), total=len(alpha_nums)):
            try:
                alpha_num, result = future.result()
                name = f'alpha{alpha_num:03d}'
                results_dict[name] = result.set_index(['date', 'ticker'])[name]
            except Exception as e:
                print(f"✗ Alpha {futures[future]:03d}: {e}")
    
    return pd.DataFrame(results_dict).reset_index()


def calculate_alphas_streaming(df_data, output_dir, skip=None):
    """
    Calcola gli alpha uno alla volta e li salva su disco.
    Picco RAM = un pivot wide (2400×2400 float64) ≈ 45MB per alpha.
    """
    skip = skip or {} #or {5, 21, 25, 27, 33, 35}  # non implementati
    
    alpha_dir = os.path.join(output_dir, "alphas")
    os.makedirs(alpha_dir, exist_ok=True)

    calc = Alphas191(df_data)  # pivot costruito una volta sola
    alpha_nums = [i for i in range(1, 192) if i not in skip]
    
    computed = []
    for alpha_num in tqdm(alpha_nums, desc="Computing alphas"):
        print(f"Computing alpha {alpha_num}")
        out_path = os.path.join(alpha_dir, f"alpha{alpha_num:03d}.parquet")
        if os.path.exists(out_path):
            computed.append(alpha_num)
            continue  # riprendi da dove eri se crasha
        try:
            result = calc.calculate_alpha(alpha_num, return_long=True)
            result.to_parquet(out_path, index=False)
            computed.append(alpha_num)
            del result
        except Exception as e:
            print(f"✗ Alpha {alpha_num:03d}: {e}")
    
    return computed, alpha_dir


def merge_alphas_into_df(base_df, alpha_dir, computed_nums):
    """
    Legge gli alpha dal disco e li mergia nel df base uno alla volta.
    """
    print(f"Merging {len(computed_nums)} alpha files...")
    result = base_df.copy()
    
    for alpha_num in tqdm(computed_nums, desc="Merging alphas"):
        path = os.path.join(alpha_dir, f"alpha{alpha_num:03d}.parquet")
        col = f"alpha{alpha_num:03d}"
        alpha_df = pd.read_parquet(path)[["date", "ticker", col]]
        result = result.merge(alpha_df, on=["date", "ticker"], how="left")
        del alpha_df
    
    return result


# ==================== ESEMPIO D'USO CORRETTO ====================

if __name__ == '__main__':
    # 1. Inizializza la classe
    alphas = Alphas191(loaded_df)
    
    # 2. Calcola tutti gli alpha (formato long)
    all_alphas = alphas.calculate_all_alphas(return_long=True)
    
    # 3. Unisci con i dati originali (SENZA duplicare date e ticker)
    result_df = loaded_df.merge(
        all_alphas, 
        on=['date', 'ticker'], 
        how='left'
    )
    
    print(f"Shape originale: {loaded_df.shape}")
    print(f"Shape con alpha: {result_df.shape}")
    print(f"Colonne alpha aggiunte: {result_df.shape[1] - loaded_df.shape[1]}")
    print("\nPrime righe:")
    print(result_df.head())
    
    # 4. Oppure calcola un singolo alpha
    alpha001 = alphas.calculate_alpha(1, return_long=True)
    print("\nAlpha001:")
    print(alpha001.head())

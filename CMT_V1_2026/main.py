from AlgorithmImports import *
import numpy as np
import pandas as pd

class MultiCommodityTermStructureMomentum(QCAlgorithm):
    """Multi-commodity futures: 63d momentum + term structure
    Uses QC continuous futures — no manual contract management
    IS: 2020-01-01 to 2023-12-31
    OOS: 2024-01-01 to 2025-12-31
    """

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(1_000_000)

        # Continuous futures — QC handles roll automatically
        self.future_tickers = ['CL', 'GC', 'SI', 'NG']
        self.future_symbols = {}

        for t in self.future_tickers:
            future = self.AddFuture(t, Resolution.DAILY)
            future.SetFilter(timedelta(0), timedelta(days=60))
            self.future_symbols[t] = future.Symbol

        self.SetBenchmark('SPY')

        # Parameters
        self.mom_lookback = 63        # 3-month momentum
        self.vol_lookback = 21        # 1-month vol
        self.term_short   = 5
        self.term_long    = 20
        self.num_positions = 3
        self.rebalance_days = 5
        self.target_vol   = 0.18
        self.max_leverage = 2.0

        self._last_rebalance_time = self.Time - timedelta(days=10)
        self.SetWarmup(self.mom_lookback + self.term_long + 2)

    def OnData(self, data):
        if self.IsWarmingUp:
            return

        if (self.Time - self._last_rebalance_time).days < self.rebalance_days:
            return
        self._last_rebalance_time = self.Time

        signals = {}

        for ticker, sym in self.future_symbols.items():
            # Use the canonical continuous symbol for history
            if sym not in data.Bars:
                continue

            price = data.Bars[sym].Close
            if price <= 0:
                continue

            # History on the continuous future symbol
            lookback = self.mom_lookback + self.term_long + 5
            try:
                hist = self.History(sym, lookback, Resolution.DAILY)
            except Exception:
                continue

            if hist.empty or len(hist) < self.mom_lookback + self.term_long:
                continue

            # Extract close series
            close = None
            if isinstance(hist, pd.DataFrame):
                if 'close' in hist.columns:
                    if 'symbol' in hist.columns.names:
                        close = hist['close'].xs(sym.Value, level='symbol')
                    else:
                        close = hist['close']
            elif isinstance(hist, pd.Series):
                close = hist

            if close is None or len(close) < self.mom_lookback + self.term_long:
                continue

            close = close.dropna().astype(float)

            # 3-month momentum return
            mom_ret = (float(close.iloc[-1]) / float(close.iloc[0])) - 1

            # 1-month realized vol
            rets = close.pct_change().dropna()
            if len(rets) < self.vol_lookback:
                continue
            real_vol = float(rets.tail(self.vol_lookback).std() * np.sqrt(252))

            # Term structure: short vs long MA (proxy for roll yield)
            avg_short = float(close.tail(self.term_short).mean())
            avg_long  = float(close.tail(self.term_long).mean())
            term_signal = (avg_short / avg_long) - 1 if avg_long > 0 else 0

            # Composite score
            term_norm = term_signal / 0.02
            score = mom_ret + 0.3 * term_norm

            signals[ticker] = {
                'score':    score,
                'momentum': mom_ret,
                'vol':      max(real_vol, 0.05),
                'symbol':   sym,
            }

        if len(signals) < 2:
            return

        # Rank top N
        ranked   = sorted(signals.items(), key=lambda x: x[1]['score'], reverse=True)
        selected = ranked[:self.num_positions]

        # Liquidate unselected
        selected_tickers = {r[0] for r in selected}
        for t, info in signals.items():
            if t not in selected_tickers and self.Portfolio[info['symbol']].Invested:
                self.Liquidate(info['symbol'])

        # Inverse-vol weighted + vol-scaled
        inv_vols  = [1.0 / r[1]['vol'] for r in selected]
        total_inv = sum(inv_vols)
        avg_vol   = np.mean([r[1]['vol'] for r in selected])

        for ticker, info in selected:
            raw_weight = (1.0 / info['vol']) / total_inv
            vol_adj    = self.target_vol / avg_vol if avg_vol > 0 else 1.0
            target     = raw_weight * vol_adj
            target     = max(-self.max_leverage, min(self.max_leverage, target))
            self.SetHoldings(info['symbol'], target)

    def OnEndOfAlgorithm(self):
        self.Log(f'Final Value: ${self.Portfolio.TotalPortfolioValue:,.2f}')
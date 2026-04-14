from AlgorithmImports import *
import numpy as np
import pandas as pd

class MultiCommodityMomentum(QCAlgorithm):
    """Commodity futures momentum using History() as primary data source.
    Uses scheduled events to rebalance every 5 days.
    Data: CL GC SI NG via front-month futures (QC handles roll automatically).
    IS: 2020-01-01 to 2023-12-31 | OOS: 2024-2025
    """
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(1_000_000)

        # Add futures as data subscriptions
        self.future_tickers = ['CL', 'GC', 'SI', 'NG']
        self.future_symbols = {}
        for t in self.future_tickers:
            f = self.AddFuture(t, Resolution.DAILY)
            # Store canonical symbol for History() calls
            self.future_symbols[t] = f.Symbol

        self.SetBenchmark('SPY')

        # Parameters
        self.mom_lookback     = 63       # 3-month
        self.vol_lookback     = 21       # 1-month
        self.rebalance_days   = 5
        self.target_vol       = 0.18
        self.max_leverage     = 2.0
        self.num_positions    = 3

        # Scheduled rebalance — runs every 5 trading days
        self.rebalance_count = 0
        self.schedule.On(
            self.date_rules.Every(DayType.WEEK_DAY),
            self.time_rules.AfterMarketOpen('SPY', 30),
            self.Rebalance
        )

        self.SetWarmup(self.mom_lookback + 5)
        self.Debug(f'Initialized {self.Time}')

    def Rebalance(self):
        if self.IsWarmingUp:
            return

        signals = {}
        for ticker in self.future_tickers:
            # Use stored canonical symbol for History() calls
            sym = self.future_symbols[ticker]

            # History on the canonical future symbol
            lookback = self.mom_lookback + self.vol_lookback + 2
            try:
                hist = self.History(sym, lookback, Resolution.DAILY)
            except Exception:
                continue

            if hist.empty or len(hist) < self.mom_lookback:
                continue

            # Extract close
            if isinstance(hist, pd.DataFrame):
                cols = [c for c in hist.columns if c in ('close', 'Close', 'adjclose')]
                if cols:
                    close = hist[cols[0]]
                elif 'close' in hist.index.names:
                    close = hist.xs('close', level='close')
                else:
                    close = hist.iloc[:, 3]  # fallback: last column
            else:
                close = hist

            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = pd.Series(close).dropna().astype(float)

            if len(close) < self.mom_lookback:
                continue

            # Momentum
            mom = (float(close.iloc[-1]) / float(close.iloc[0])) - 1

            # Vol
            rets = close.pct_change().dropna()
            if len(rets) < self.vol_lookback:
                continue
            vol = float(rets.tail(self.vol_lookback).std() * np.sqrt(252))

            signals[ticker] = {
                'score':    mom,
                'momentum': mom,
                'vol':      max(vol, 0.05),
                'symbol':   sym,
            }

        if len(signals) < 2:
            return

        ranked   = sorted(signals.items(), key=lambda x: x[1]['score'], reverse=True)[:self.num_positions]
        selected = {r[0] for r in ranked}

        # Liquidate unselected
        for t, info in signals.items():
            if t not in selected and self.Portfolio[info['symbol']].Invested:
                self.Liquidate(info['symbol'])

        # Inverse-vol weights
        inv_vols  = [1.0 / r[1]['vol'] for r in ranked]
        total_inv = sum(inv_vols)
        avg_vol   = np.mean([r[1]['vol'] for r in ranked])

        for ticker, info in ranked:
            raw = (1.0 / info['vol']) / total_inv
            adj = self.target_vol / avg_vol if avg_vol > 0 else 1.0
            tgt = raw * adj
            tgt = max(-self.max_leverage, min(self.max_leverage, tgt))
            self.SetHoldings(info['symbol'], tgt)

        self.rebalance_count += 1
        self.Debug(f'Rebalance #{self.rebalance_count} @ {self.Time}')

    def OnEndOfAlgorithm(self):
        self.Log(f'Final: ${self.Portfolio.TotalPortfolioValue:,.2f}, Rebalances: {self.rebalance_count}')
from AlgorithmImports import *
import numpy as np

class MultiCommodityTermStructureMomentum(QCAlgorithm):
    """Multi-commodity futures: term structure carry + momentum
    Instruments: CL (WTI Crude), GC (Gold), SI (Silver), NG (Nat Gas)
    via QC AddFuture (handles roll automatically)
    Alpha: 63d momentum + term structure signal
    IS: 2020-01-01 to 2023-12-31
    OOS: 2024-01-01 to 2025-12-31
    """

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(1_000_000)

        # Futures universe — liquid energy + metals
        self.future_symbols = ['CL', 'GC', 'SI', 'NG']

        for ticker in self.future_symbols:
            # Add as perpetual front-month futures (QC handles roll)
            future = self.AddFuture(ticker, Resolution.DAILY)
            future.SetFilter(timedelta(0), timedelta(days=60))

        # Benchmark
        self.SetBenchmark('SPY')

        # Parameters
        self.mom_lookback = 63        # 3-month momentum
        self.vol_lookback = 21        # 1-month vol
        self.term_short = 5           # short-term avg for term structure
        self.term_long = 20           # long-term avg for term structure
        self.num_positions = 3        # top N commodities
        self.rebalance_days = 5       # rebalance every 5 days
        self.target_vol = 0.18        # 18% annual target vol
        self.max_leverage = 2.0

        # State
        self._last_rebalance = {s: self.Time - timedelta(days=10)
                                 for s in self.future_symbols}
        self._last_rebalance_time = self.Time - timedelta(days=10)

        # Warmup
        self.SetWarmup(self.mom_lookback + 5)

    def OnData(self, data):
        if self.IsWarmingUp:
            return

        # Time-based rebalance
        if (self.Time - self._last_rebalance_time).days < self.rebalance_days:
            return
        self._last_rebalance_time = self.Time

        # Build signals for all available futures chains
        signals = {}

        for ticker in self.future_symbols:
            # Get the front-month contract
            chain = self.FutureChainProvider.GetFutureContractList(
                getattr(Futures, ticker), self.Time)
            if not chain or len(chain) == 0:
                continue

            # Front-month contract
            contract = sorted(chain, key=lambda x: x.Expiry)[0]
            symbol = contract.Symbol

            if not data.ContainsKey(symbol) or data[symbol] is None:
                continue

            price = data[symbol].Close
            if price <= 0:
                continue

            # History for momentum + term structure
            hist = self.History(symbol, self.mom_lookback + self.term_long + 2, Resolution.DAILY)
            if hist.empty or len(hist) < self.mom_lookback + self.term_long:
                continue

            # Drop canonical/duplicates, keep last column
            if 'symbol' in hist.columns.names:
                hist = hist.xs(symbol.Value, level='symbol', drop_level=False)
            if len(hist.shape) > 1 and hist.shape[1] > 1:
                close = hist['close'].unstack(level='symbol')[symbol.Value].dropna()
            else:
                close = hist['close'].dropna() if 'close' in hist.columns else hist.iloc[:, 0]

            if len(close) < self.mom_lookback + self.term_long:
                continue

            # Momentum: 3-month return
            mom_start = float(close.iloc[0])
            mom_end   = float(close.iloc[-1])
            if mom_start <= 0:
                continue
            momentum = (mom_end / mom_start) - 1

            # Realized vol
            rets = close.pct_change().dropna()
            if len(rets) < self.vol_lookback:
                continue
            realized_vol = float(rets.tail(self.vol_lookback).std() * np.sqrt(252))

            # Term structure: short MA vs long MA (proxy for roll yield)
            avg_short = float(close.tail(self.term_short).mean())
            avg_long  = float(close.tail(self.term_long).mean())
            term_signal = (avg_short / avg_long) - 1 if avg_long > 0 else 0

            # Composite score: momentum + 0.3 * term_normalized
            term_norm = term_signal / 0.02   # normalize by ~2% typical range
            score = momentum + 0.3 * term_norm

            signals[ticker] = {
                'score':     score,
                'momentum':  momentum,
                'vol':       max(realized_vol, 0.05),
                'symbol':    symbol,
            }

        if len(signals) < 2:
            return

        # Rank and select top N
        ranked = sorted(signals.items(), key=lambda x: x[1]['score'], reverse=True)
        selected = ranked[:self.num_positions]

        # Liquidate unselected
        selected_tickers = {r[0] for r in selected}
        for ticker in self.future_symbols:
            if ticker not in selected_tickers:
                info = signals.get(ticker)
                if info and self.Portfolio[info['symbol']].Invested:
                    self.Liquidate(info['symbol'])

        # Compute inverse-vol weights
        inv_vols = [1.0 / r[1]['vol'] for r in selected]
        total_inv = sum(inv_vols)
        avg_vol   = np.mean([r[1]['vol'] for r in selected])

        for ticker, info in selected:
            raw_weight = (1.0 / info['vol']) / total_inv
            # Vol scaling
            vol_adj = self.target_vol / avg_vol if avg_vol > 0 else 1.0
            target  = raw_weight * vol_adj
            # Clamp leverage
            target = max(-self.max_leverage, min(self.max_leverage, target))
            self.SetHoldings(info['symbol'], target)

    def OnEndOfAlgorithm(self):
        self.Log(f'Rebalance count: {self._last_rebalance_time}')
        self.Log(f'Final Value: ${self.Portfolio.TotalPortfolioValue:,.2f}')
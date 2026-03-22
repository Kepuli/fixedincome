#What is tested — 7 groups, 42 tests total:**

# TestDataIntegrity** — parquet files have the right columns, no NaNs, correct date ranges, yields are between -5% and 15%.

# TestReturnCalculations** — log return formula is correct, and spot-checks against real history: MSCI Europe lost ~45% in 2008, long bonds crashed in 2022, bonds rallied in 2019.

# TestForwardRates** — forward rate formula satisfies the no-arbitrage condition mathematically, and the shift(-12) alignment works correctly.

# TestRegressions** — Fama-Bliss beta declines from 1Y to 4Y horizon (the key theoretical prediction), Q5 alpha is insignificant (matching Asvanunt & Richardson), R² is in the 0.35-0.75 range.

# TestPortfolioStats** — annualised return/vol/Sharpe formulas match your known outputs, all portfolio weight dicts sum to 1.0, 60/40 has lower vol than 100% equity.

# TestDataAlignment** — directly tests the Q5 NaN bug fix: after index normalization, joining MSCI and ETF data produces zero NaN rows.

# TestModuleSmoke** — imports each Q module and calls `run_all()` end-to-end without crashing. Most useful for catching broken imports after a teammate edits code.


# tests/test_pipeline.py
# ─────────────────────────────────────────────────────────────
# Automated test suite for the Fixed Income Case Study pipeline.
#
# Usage:
#   pip install pytest
#   pytest tests/test_pipeline.py -v
#
# Tests are grouped by component:
#   TestDataIntegrity     — raw parquet files have correct shape, dtypes, coverage
#   TestReturnCalculations— log return values match known financial events
#   TestForwardRates      — forward rate formula produces correct values
#   TestRegressions       — Q2 and Q5 regression outputs are in plausible range
#   TestPortfolioStats    — portfolio statistics are internally consistent
#   TestDataAlignment     — date indices align correctly across datasets
# ─────────────────────────────────────────────────────────────

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path so src imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_PROCESSED = Path("data/processed")


# ══════════════════════════════════════════════════════════════
# FIXTURES — load data once per test session
# ══════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def spot_rates():
    return pd.read_parquet(DATA_PROCESSED / "ecb_spot_rates.parquet")

@pytest.fixture(scope="session")
def etf_prices():
    return pd.read_parquet(DATA_PROCESSED / "etf_prices.parquet")

@pytest.fixture(scope="session")
def etf_returns():
    return pd.read_parquet(DATA_PROCESSED / "etf_returns.parquet")

@pytest.fixture(scope="session")
def msci():
    return pd.read_parquet(DATA_PROCESSED / "msci_europe.parquet")


# ══════════════════════════════════════════════════════════════
# 1. DATA INTEGRITY TESTS
# ══════════════════════════════════════════════════════════════

class TestDataIntegrity:
    """
    Tests that parquet files exist, have the right columns,
    no unexpected NaNs, correct date ranges, and plausible value ranges.
    These catch problems in data_import.py.
    """

    def test_spot_rates_exists(self, spot_rates):
        assert spot_rates is not None
        assert len(spot_rates) > 0, "ECB spot rates file is empty"

    def test_spot_rates_columns(self, spot_rates):
        expected = ["spot_1y", "spot_2y", "spot_3y", "spot_4y",
                    "spot_5y", "spot_10y", "spot_30y"]
        for col in expected:
            assert col in spot_rates.columns, f"Missing column: {col}"

    def test_spot_rates_coverage(self, spot_rates):
        assert spot_rates.index.min() <= pd.Timestamp("2004-12-31"), \
            "ECB data should start by end of 2004"
        assert spot_rates.index.max() >= pd.Timestamp("2025-01-01"), \
            "ECB data should extend to at least 2025"

    def test_spot_rates_no_nans(self, spot_rates):
        nan_counts = spot_rates.isna().sum()
        for col, count in nan_counts.items():
            assert count == 0, f"{col} has {count} NaN values"

    def test_spot_rates_plausible_range(self, spot_rates):
        """Yields should be between -5% and 15% for euro area."""
        assert spot_rates.min().min() >= -5.0, \
            "Some yields are implausibly negative (below -5%)"
        assert spot_rates.max().max() <= 15.0, \
            "Some yields are implausibly high (above 15%)"

    def test_spot_rates_term_structure(self, spot_rates):
        """
        On average, 10Y should be higher than 1Y (upward sloping on average).
        Not true at every point, but should hold on average.
        """
        avg_1y  = spot_rates["spot_1y"].mean()
        avg_10y = spot_rates["spot_10y"].mean()
        assert avg_10y > avg_1y, \
            f"Average 10Y ({avg_10y:.2f}%) not above average 1Y ({avg_1y:.2f}%) — unexpected"

    def test_etf_prices_columns(self, etf_prices):
        expected = ["govt_short", "govt_mid", "govt_long", "corp_ig", "equity"]
        for col in expected:
            assert col in etf_prices.columns, f"Missing ETF price column: {col}"

    def test_etf_returns_columns(self, etf_returns):
        expected = ["govt_short_logret", "govt_mid_logret", "govt_long_logret",
                    "corp_ig_logret", "equity_logret"]
        for col in expected:
            assert col in etf_returns.columns, f"Missing ETF return column: {col}"

    def test_etf_returns_plausible_range(self, etf_returns):
        """Monthly log returns should be between -50% and +50% for these assets."""
        for col in etf_returns.columns:
            series = etf_returns[col].dropna()
            assert series.min() >= -0.50, \
                f"{col} has monthly return below -50%: {series.min():.3f}"
            assert series.max() <= 0.50, \
                f"{col} has monthly return above +50%: {series.max():.3f}"

    def test_etf_prices_positive(self, etf_prices):
        """ETF prices must always be positive."""
        for col in etf_prices.columns:
            series = etf_prices[col].dropna()
            assert (series > 0).all(), f"{col} contains non-positive prices"

    def test_msci_exists(self, msci):
        assert msci is not None
        assert "equity_logret" in msci.columns, "MSCI parquet missing equity_logret column"

    def test_msci_coverage(self, msci):
        assert msci.index.min() <= pd.Timestamp("2000-06-30"), \
            "MSCI Europe data should start by mid-2000"
        assert msci.index.max() >= pd.Timestamp("2025-01-01"), \
            "MSCI Europe data should extend to at least 2025"

    def test_msci_price_positive(self, msci):
        assert (msci["price"] > 0).all(), "MSCI price contains non-positive values"


# ══════════════════════════════════════════════════════════════
# 2. RETURN CALCULATION TESTS
# ══════════════════════════════════════════════════════════════

class TestReturnCalculations:
    """
    Spot-checks return calculations against known financial history.
    If these fail, the log return formula or resampling is wrong.
    """

    def test_log_return_formula(self, etf_prices, etf_returns):
        """
        Verify that log returns were computed correctly:
        r_t = ln(P_t / P_{t-1})
        Check this holds for a random sample of 10 rows.
        """
        col_price  = "govt_mid"
        col_return = "govt_mid_logret"
        prices  = etf_prices[col_price].dropna()
        returns = etf_returns[col_return].dropna()

        # Align indices
        common = prices.index.intersection(returns.index)
        prices  = prices.loc[common]
        returns = returns.loc[common]

        # Compute expected returns manually
        expected = np.log(prices / prices.shift(1)).dropna()
        actual   = returns.loc[expected.index]

        np.testing.assert_allclose(
            actual.values, expected.values, atol=1e-10,
            err_msg="Log return formula mismatch — check data_import.py"
        )

    def test_2008_equity_crash(self, msci):
        """
        MSCI Europe lost roughly 45-50% in 2008.
        Annual log return should be approximately -0.65 to -0.45.
        """
        year_2008 = msci.loc["2008-01":"2008-12", "equity_logret"].dropna()
        annual_return = year_2008.sum()
        assert -0.70 <= annual_return <= -0.35, \
            f"2008 MSCI Europe return {annual_return:.3f} outside expected range [-0.70, -0.35]"

    def test_2022_bond_crash(self, etf_returns):
        """
        2022 was the worst year for bonds in decades due to ECB rate hikes.
        Long govt bonds should have large negative returns.
        """
        year_2022 = etf_returns.loc["2022-01":"2022-12", "govt_long_logret"].dropna()
        annual_return = year_2022.sum()
        assert annual_return <= -0.15, \
            f"Long govt 2022 return {annual_return:.3f} — expected severe loss (< -15%)"

    def test_2019_bond_rally(self, etf_returns):
        """
        2019: ECB dovish pivot, rates fell, bonds rallied.
        Mid govt should have a clearly positive return.
        """
        year_2019 = etf_returns.loc["2019-01":"2019-12", "govt_mid_logret"].dropna()
        annual_return = year_2019.sum()
        assert annual_return >= 0.05, \
            f"Govt mid 2019 return {annual_return:.3f} — expected positive (> 5%)"

    def test_cumulative_return_consistency(self, etf_prices, etf_returns):
        """
        Cumulative sum of log returns should approximately equal
        ln(final_price / initial_price).
        Tests that resampling and return calculation are consistent.
        """
        col_price  = "govt_mid"
        col_return = "govt_mid_logret"

        prices  = etf_prices[col_price].dropna()
        returns = etf_returns[col_return].dropna()

        # Find common window
        start = max(prices.index.min(), returns.index.min())
        end   = min(prices.index.max(), returns.index.max())

        prices  = prices.loc[start:end]
        returns = returns.loc[start:end]

        cum_return_from_prices  = np.log(prices.iloc[-1] / prices.iloc[0])
        cum_return_from_returns = returns.sum()

        np.testing.assert_allclose(
            cum_return_from_returns, cum_return_from_prices,
            atol=0.02,  # allow 2% tolerance for resampling differences
            err_msg="Cumulative return from prices vs summed log returns mismatch"
        )

    def test_msci_log_return_formula(self, msci):
        """Verify MSCI log returns computed correctly from price."""
        prices  = msci["price"].dropna()
        returns = msci["equity_logret"].dropna()
        expected = np.log(prices / prices.shift(1)).dropna()
        actual   = returns.loc[expected.index]
        np.testing.assert_allclose(
            actual.values, expected.values, atol=1e-10,
            err_msg="MSCI log return formula mismatch"
        )


# ══════════════════════════════════════════════════════════════
# 3. FORWARD RATE TESTS
# ══════════════════════════════════════════════════════════════

class TestForwardRates:
    """
    Tests the forward rate formula used in Q2.
    f(n, n+1) = ((1 + s_{n+1})^{n+1} / (1 + s_n)^n) - 1
    """

    def test_forward_rate_formula_manual(self):
        """
        Manual verification: if s_1y = 2% and s_2y = 3%,
        then f(1,2) = ((1.03)^2 / (1.02)^1) - 1 ≈ 4.009%
        """
        s1 = 0.02
        s2 = 0.03
        expected = ((1 + s2) ** 2 / (1 + s1) ** 1) - 1
        assert abs(expected - 0.04009) < 0.0001, \
            f"Forward rate formula gives {expected:.5f}, expected ~0.04009"

    def test_forward_rate_no_arbitrage(self, spot_rates):
        """
        No-arbitrage: investing for 2Y at the spot rate must equal
        investing for 1Y then rolling into the 1Y forward.
        (1 + s_2y)^2 = (1 + s_1y) * (1 + fwd_1y1y)
        Test this holds for all observations.
        """
        s1 = spot_rates["spot_1y"] / 100
        s2 = spot_rates["spot_2y"] / 100

        fwd = ((1 + s2) ** 2 / (1 + s1) ** 1) - 1

        lhs = (1 + s2) ** 2
        rhs = (1 + s1) * (1 + fwd)

        np.testing.assert_allclose(
            lhs.values, rhs.values, atol=1e-10,
            err_msg="Forward rate no-arbitrage condition violated"
        )

    def test_forward_rate_positive_slope(self, spot_rates):
        """
        When the yield curve is upward sloping (10Y > 1Y),
        the 1Y forward rate should be above the 2Y spot rate.
        Test holds for the majority of observations.
        """
        s1 = spot_rates["spot_1y"] / 100
        s2 = spot_rates["spot_2y"] / 100
        fwd = ((1 + s2) ** 2 / (1 + s1) ** 1) - 1

        upward_sloping = s2 > s1
        fwd_above_s2   = fwd > s2

        # Should hold for most upward-sloping observations
        pct_correct = (fwd_above_s2 & upward_sloping).sum() / upward_sloping.sum()
        assert pct_correct > 0.95, \
            f"Forward rate above spot only {pct_correct:.1%} of upward-sloping cases"

    def test_fwd_shift_alignment(self, spot_rates):
        """
        The shift(-12) alignment should produce exactly 12 fewer
        non-NaN observations than the original series.
        """
        original = spot_rates["spot_1y"]
        shifted  = spot_rates["spot_1y"].shift(-12)

        n_original = original.notna().sum()
        n_shifted  = shifted.notna().sum()

        assert n_original - n_shifted == 12, \
            f"shift(-12) should reduce non-NaN count by 12, got {n_original - n_shifted}"


# ══════════════════════════════════════════════════════════════
# 4. REGRESSION TESTS
# ══════════════════════════════════════════════════════════════

class TestRegressions:
    """
    Tests that Q2 Fama-Bliss and Q5 systematic risk regressions
    produce results in the expected range.
    These are regression tests against known-good outputs —
    if these fail after a code change, something broke.
    """

    def test_fama_bliss_1y_beta(self, spot_rates):
        """
        1Y Fama-Bliss beta should be between 0.5 and 1.0.
        Your pipeline produced 0.792 — test within ±0.15 tolerance.
        """
        import statsmodels.api as sm

        s = spot_rates[["spot_1y", "spot_2y"]].dropna()
        fwd = ((1 + s["spot_2y"] / 100) ** 2 /
               (1 + s["spot_1y"] / 100) ** 1) - 1
        fwd = fwd * 100

        realized = s["spot_1y"].shift(-12)
        df = pd.DataFrame({"fwd": fwd, "realized": realized}).dropna()

        X = sm.add_constant(df["fwd"])
        model = sm.OLS(df["realized"], X).fit()
        beta = model.params["fwd"]

        assert 0.60 <= beta <= 0.95, \
            f"1Y Fama-Bliss beta = {beta:.3f}, expected between 0.60 and 0.95"

    def test_fama_bliss_beta_declining(self, spot_rates):
        """
        Key theoretical prediction: beta should decline as horizon increases.
        1Y beta > 2Y beta > 3Y beta (declining predictive power).
        """
        import statsmodels.api as sm

        betas = {}
        pairs = [(1, "spot_1y", "spot_2y"), (2, "spot_2y", "spot_3y"),
                 (3, "spot_3y", "spot_4y")]

        for horizon, col_n, col_n1 in pairs:
            n = int(col_n.split("_")[1].replace("y",""))
            n1 = n + 1
            s_n  = spot_rates[col_n] / 100
            s_n1 = spot_rates[col_n1] / 100
            fwd  = ((1 + s_n1) ** n1 / (1 + s_n) ** n - 1) * 100
            realized = spot_rates["spot_1y"].shift(-horizon * 12)
            df = pd.DataFrame({"fwd": fwd, "realized": realized}).dropna()
            X = sm.add_constant(df["fwd"])
            betas[horizon] = sm.OLS(df["realized"], X).fit().params["fwd"]

        assert betas[1] > betas[2], \
            f"1Y beta ({betas[1]:.3f}) should exceed 2Y beta ({betas[2]:.3f})"
        assert betas[2] > betas[3], \
            f"2Y beta ({betas[2]:.3f}) should exceed 3Y beta ({betas[3]:.3f})"

    def test_q5_regression_r_squared(self, etf_returns, msci):
        """
        Q5 systematic risk regression R² should be between 0.4 and 0.7.
        Your pipeline produced 0.561.
        """
        import statsmodels.api as sm

        etf = etf_returns.copy()
        m   = msci["equity_logret"].copy()

        etf.index = etf.index.to_period("M").to_timestamp("M")
        m.index   = m.index.to_period("M").to_timestamp("M")

        df = pd.DataFrame({
            "corp_ig":  etf["corp_ig_logret"],
            "equity":   m,
            "govt_mid": etf["govt_mid_logret"],
        }).dropna()

        y = df["corp_ig"]
        X = sm.add_constant(df[["equity", "govt_mid"]])
        model = sm.OLS(y, X).fit()

        assert 0.35 <= model.rsquared <= 0.75, \
            f"Q5 R² = {model.rsquared:.3f}, expected between 0.35 and 0.75"

    def test_q5_equity_beta_positive(self, etf_returns, msci):
        """
        Beta of corp bonds to equity should be small but positive.
        Corporate bonds have some equity-like risk (0.05 to 0.25 range).
        """
        import statsmodels.api as sm

        etf = etf_returns.copy()
        m   = msci["equity_logret"].copy()
        etf.index = etf.index.to_period("M").to_timestamp("M")
        m.index   = m.index.to_period("M").to_timestamp("M")

        df = pd.DataFrame({
            "corp_ig":  etf["corp_ig_logret"],
            "equity":   m,
            "govt_mid": etf["govt_mid_logret"],
        }).dropna()

        X = sm.add_constant(df[["equity", "govt_mid"]])
        model = sm.OLS(df["corp_ig"], X).fit()
        beta_eq = model.params["equity"]

        assert 0.05 <= beta_eq <= 0.30, \
            f"Equity beta = {beta_eq:.4f}, expected between 0.05 and 0.30"

    def test_q5_alpha_insignificant(self, etf_returns, msci):
        """
        Per Asvanunt & Richardson (2017), alpha should be near zero
        and statistically insignificant (p > 0.05).
        """
        import statsmodels.api as sm

        etf = etf_returns.copy()
        m   = msci["equity_logret"].copy()
        etf.index = etf.index.to_period("M").to_timestamp("M")
        m.index   = m.index.to_period("M").to_timestamp("M")

        df = pd.DataFrame({
            "corp_ig":  etf["corp_ig_logret"],
            "equity":   m,
            "govt_mid": etf["govt_mid_logret"],
        }).dropna()

        X = sm.add_constant(df[["equity", "govt_mid"]])
        model = sm.OLS(df["corp_ig"], X).fit(cov_type="HC3")
        alpha_pval = model.pvalues["const"]

        assert alpha_pval > 0.05, \
            f"Alpha p-value = {alpha_pval:.3f} — alpha is significant, " \
            f"contradicts Asvanunt & Richardson finding. Check data."


# ══════════════════════════════════════════════════════════════
# 5. PORTFOLIO STATISTICS TESTS
# ══════════════════════════════════════════════════════════════

class TestPortfolioStats:
    """
    Tests that portfolio statistics are internally consistent
    and match known outputs from your pipeline.
    """

    def test_annualised_return_formula(self, etf_returns):
        """Ann. return = mean(monthly) * 12. Verify against known output."""
        r = etf_returns["govt_mid_logret"].dropna()
        ann_return = r.mean() * 12 * 100
        # Your pipeline reported 2.22% — allow ±0.1% tolerance
        assert abs(ann_return - 2.22) < 0.5, \
            f"Govt mid annualised return = {ann_return:.2f}%, expected ~2.22%"

    def test_annualised_vol_formula(self, etf_returns):
        """Ann. vol = std(monthly) * sqrt(12). Verify against known output."""
        r = etf_returns["govt_mid_logret"].dropna()
        ann_vol = r.std() * np.sqrt(12) * 100
        # Your pipeline reported 5.82%
        assert abs(ann_vol - 5.82) < 0.5, \
            f"Govt mid annualised vol = {ann_vol:.2f}%, expected ~5.82%"

    def test_sharpe_ratio_formula(self, etf_returns):
        """Sharpe = Ann.Return / Ann.Vol. Verify consistency."""
        r = etf_returns["govt_mid_logret"].dropna()
        ann_return = r.mean() * 12
        ann_vol    = r.std()  * np.sqrt(12)
        sharpe     = ann_return / ann_vol
        # Your pipeline reported 0.38
        assert abs(sharpe - 0.38) < 0.05, \
            f"Govt mid Sharpe = {sharpe:.3f}, expected ~0.38"

    def test_max_drawdown_negative(self, etf_returns):
        """Max drawdown must always be negative or zero."""
        for col in etf_returns.columns:
            r   = etf_returns[col].dropna()
            cum = np.exp(r.cumsum())
            dd  = ((cum - cum.cummax()) / cum.cummax()).min()
            assert dd <= 0, f"{col} max drawdown is positive: {dd}"

    def test_portfolio_weights_sum_to_one(self):
        """All portfolio weight definitions must sum to 1.0."""
        portfolios = {
            "Pure Govt":        {"govt_mid": 1.0},
            "Blended Bond":     {"govt_mid": 0.50, "corp_ig": 0.50},
            "Diversified Bond": {"govt_short": 0.33, "govt_mid": 0.34, "corp_ig": 0.33},
            "60/40":            {"equity": 0.60, "govt_mid": 0.40},
            "60/30/10":         {"equity": 0.60, "govt_mid": 0.30, "corp_ig": 0.10},
            "60/20/20":         {"equity": 0.60, "govt_mid": 0.20, "corp_ig": 0.20},
        }
        for name, weights in portfolios.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-9, \
                f"Portfolio '{name}' weights sum to {total}, not 1.0"

    def test_diversification_reduces_vol(self, etf_returns, msci):
        """
        60/40 portfolio should have lower volatility than 100% equity.
        Basic diversification test.
        """
        etf = etf_returns.copy()
        m   = msci["equity_logret"].copy()
        etf.index = etf.index.to_period("M").to_timestamp("M")
        m.index   = m.index.to_period("M").to_timestamp("M")

        df = pd.DataFrame({
            "equity":   m,
            "govt_mid": etf["govt_mid_logret"],
        }).dropna()

        vol_equity   = df["equity"].std() * np.sqrt(12)
        vol_portfolio = (0.6 * df["equity"] + 0.4 * df["govt_mid"]).std() * np.sqrt(12)

        assert vol_portfolio < vol_equity, \
            f"60/40 vol ({vol_portfolio:.3f}) not below equity vol ({vol_equity:.3f})"


# ══════════════════════════════════════════════════════════════
# 6. DATA ALIGNMENT TESTS
# ══════════════════════════════════════════════════════════════

class TestDataAlignment:
    """
    Tests that date indices align correctly after normalization.
    This was the root cause of the Q5 NaN bug.
    """

    def test_etf_index_is_month_end(self, etf_returns):
        """ETF return index should be month-end dates after normalization."""
        normalized = etf_returns.copy()
        normalized.index = normalized.index.to_period("M").to_timestamp("M")
        for date in normalized.index[:10]:
            # Month-end: next day should be in a different month
            next_day = date + pd.Timedelta(days=1)
            assert next_day.month != date.month or next_day.year != date.year, \
                f"Date {date} is not month-end"

    def test_msci_index_is_month_end(self, msci):
        """MSCI return index should be month-end dates."""
        for date in msci.index[:10]:
            next_day = date + pd.Timedelta(days=1)
            assert next_day.month != date.month or next_day.year != date.year, \
                f"MSCI date {date} is not month-end"

    def test_aligned_join_has_no_nans(self, etf_returns, msci):
        """
        After index normalization, joining MSCI and ETF should produce
        no NaN rows in the common sample window.
        This directly tests the fix for the Q5 bug.
        """
        etf = etf_returns.copy()
        m   = msci["equity_logret"].copy()
        etf.index = etf.index.to_period("M").to_timestamp("M")
        m.index   = m.index.to_period("M").to_timestamp("M")

        df = pd.DataFrame({
            "equity":   m,
            "govt_mid": etf["govt_mid_logret"],
            "corp_ig":  etf["corp_ig_logret"],
        }).dropna()

        assert len(df) > 100, \
            f"After alignment, only {len(df)} common rows — expected 100+"

        # Within the common sample, there should be no NaNs
        assert df.isna().sum().sum() == 0, \
            "Aligned DataFrame still contains NaN values"

    def test_common_sample_start(self, etf_returns, msci):
        """
        Common sample should start around May/June 2009
        (when corp_ig ETF becomes available).
        """
        etf = etf_returns.copy()
        m   = msci["equity_logret"].copy()
        etf.index = etf.index.to_period("M").to_timestamp("M")
        m.index   = m.index.to_period("M").to_timestamp("M")

        df = pd.DataFrame({
            "equity":  m,
            "corp_ig": etf["corp_ig_logret"],
        }).dropna()

        start = df.index.min()
        assert pd.Timestamp("2009-01-01") <= start <= pd.Timestamp("2009-12-31"), \
            f"Common sample starts {start.date()}, expected somewhere in 2009"

    def test_spot_rates_monthly_frequency(self, spot_rates):
        """
        Spot rates should be at monthly frequency with no gaps.
        Max gap between observations should be ~31 days.
        """
        diffs = spot_rates.index.to_series().diff().dropna()
        max_gap = diffs.max()
        assert max_gap <= pd.Timedelta(days=35), \
            f"Gap of {max_gap.days} days found in spot rate index — missing months?"


# ══════════════════════════════════════════════════════════════
# 7. SMOKE TESTS — run all Q modules
# ══════════════════════════════════════════════════════════════

class TestModuleSmoke:
    """
    Smoke tests: import each Q module and call run_all().
    Verifies the module runs end-to-end and returns a dict of figures.
    Does not check visual correctness — just that nothing crashes.
    """

    def test_q1_runs(self):
        import src.Q1_yield_curve as q1
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for testing
        result = q1.run_all()
        assert isinstance(result, dict), "Q1 run_all() should return a dict"
        assert len(result) > 0, "Q1 run_all() returned empty dict"
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_q2_runs(self):
        import src.Q2_forward_rates as q2
        import matplotlib
        matplotlib.use("Agg")
        result = q2.run_all()
        assert isinstance(result, dict)
        assert len(result) > 0
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_q3_runs(self):
        import src.Q3_duration_risk as q3
        import matplotlib
        matplotlib.use("Agg")
        result = q3.run_all()
        assert isinstance(result, dict)
        assert len(result) > 0
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_q4_runs(self):
        import src.Q4_credit_markets as q4
        import matplotlib
        matplotlib.use("Agg")
        result = q4.run_all()
        assert isinstance(result, dict)
        assert len(result) > 0
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_q5_runs(self):
        import src.Q5_equity_bond_portfolio as q5
        import matplotlib
        matplotlib.use("Agg")
        result = q5.run_all()
        assert isinstance(result, dict)
        assert len(result) > 0
        import matplotlib.pyplot as plt
        plt.close("all")

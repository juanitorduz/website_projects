"""User-level transaction data generator for cohort revenue retention analysis.

Uses a BG/NBD-inspired generative process for transaction timing and churn,
a Gamma-Gamma model for monetary values, configurable seasonality, and an
optional discount boost for late cohorts.
"""

from __future__ import annotations

import calendar
import logging
from datetime import UTC, datetime

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import BaseModel
from rich.logging import RichHandler

_UNSET = object()

# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class DateRangeConfig(BaseModel, frozen=True):
    """Temporal boundaries for the simulation.

    Defines the start and end of the observation window (monthly cohorts)
    and, optionally, the cohort from which the discount boost begins.

    Attributes
    ----------
    start_date : datetime
        First cohort month (inclusive), e.g. ``datetime(2020, 1, 1)``.
    end_date : datetime
        Last cohort month (inclusive), e.g. ``datetime(2022, 12, 1)``.
    discount_start_date : datetime or None
        Cohorts from this month onward receive the discount boost.
        Set to ``None`` to disable. Defaults to April 2022.
    """

    start_date: datetime
    end_date: datetime
    discount_start_date: datetime | None = datetime(2022, 4, 1, tzinfo=UTC)


class CohortConfig(BaseModel, frozen=True):
    """Cohort size parameters.

    Controls how many new users join in each monthly cohort.

    Attributes
    ----------
    base_cohort_size : int
        Baseline number of users per cohort before trend and noise
        adjustments.
    """

    base_cohort_size: int = 500


class BGNBDParams(BaseModel, frozen=True):
    """BG/NBD transaction process parameters.

    Governs how often users transact and when they churn.

    - Transaction rate per user: ``lambda_i ~ Gamma(r, 1/alpha)``
    - Dropout probability per user: ``p_i ~ Beta(a, b)``

    Attributes
    ----------
    r : float
        Shape parameter for the Gamma-distributed transaction rate.
    alpha : float
        Rate parameter for the Gamma-distributed transaction rate.
    a : float
        First shape parameter for the Beta-distributed dropout probability.
    b : float
        Second shape parameter for the Beta-distributed dropout probability.
    """

    r: float = 0.5
    alpha: float = 5.0
    a: float = 1.0
    b: float = 3.0


class GammaGammaParams(BaseModel, frozen=True):
    """Gamma-Gamma monetary value parameters.

    Governs per-transaction spending amounts.

    - Customer spending rate: ``v_i ~ Gamma(q_spend, 1/gamma_spend)``
    - Per-transaction amount: ``amount ~ Gamma(p_spend, v_i)``

    Attributes
    ----------
    p_spend : float
        Shape parameter for the per-transaction amount distribution.
    q_spend : float
        Shape parameter for the customer-level spending rate.
    gamma_spend : float
        Rate parameter for the customer-level spending rate.
    """

    p_spend: float = 5.0
    q_spend: float = 3.0
    gamma_spend: float = 5.0


class SeasonalityConfig(BaseModel, frozen=True):
    """Seasonality control.

    Modulates transaction rates with a harmonic seasonal pattern
    (peaks around Dec/Jan).

    Attributes
    ----------
    strength : float
        Amplitude multiplier. ``0.0`` produces no seasonality (flat),
        ``1.0`` gives the reference strength matching the observed pattern.
    """

    strength: float = 1.0


class DiscountBoostConfig(BaseModel, frozen=True):
    """Discount-driven purchase boost for late cohorts.

    For cohorts starting from the discount start date (set in
    ``DateRangeConfig``), the transaction rate is multiplied by a
    linearly-decaying boost during the first ``boost_months`` periods.

    Attributes
    ----------
    boost_factor : float
        Peak multiplier added to the base rate at ``cohort_age=0``.
    boost_months : int
        Number of months over which the boost decays to zero.
    """

    boost_factor: float = 3.0
    boost_months: int = 3


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class UserLevelCohortDataGenerator:
    """Generate user-level transaction data for cohort revenue retention.

    Uses a BG/NBD-inspired process for transaction timing and churn,
    a Gamma-Gamma model for monetary values, configurable seasonality,
    and an optional discount boost for late cohorts.

    Parameters
    ----------
    rng : np.random.Generator
        NumPy random generator for full determinism.
    date_range_config : DateRangeConfig
        Observation window and discount start date.
    cohort_config : CohortConfig
        Cohort sizing parameters.
    bgnbd_params : BGNBDParams
        BG/NBD transaction and churn parameters.
    gamma_gamma_params : GammaGammaParams
        Gamma-Gamma spending parameters.
    seasonality_config : SeasonalityConfig
        Seasonal pattern control.
    discount_boost_config : DiscountBoostConfig or None
        Discount boost parameters. Pass ``None`` to disable.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        date_range_config: DateRangeConfig,
        cohort_config: CohortConfig | None = None,
        bgnbd_params: BGNBDParams | None = None,
        gamma_gamma_params: GammaGammaParams | None = None,
        seasonality_config: SeasonalityConfig | None = None,
        discount_boost_config: DiscountBoostConfig | None | object = _UNSET,
    ) -> None:
        self.rng = rng
        self.date_range = date_range_config
        self.cohort_cfg = cohort_config or CohortConfig()
        self.bgnbd = bgnbd_params or BGNBDParams()
        self.gg = gamma_gamma_params or GammaGammaParams()
        self.seasonality = seasonality_config or SeasonalityConfig()
        self.discount: DiscountBoostConfig | None = (
            DiscountBoostConfig()
            if discount_boost_config is _UNSET
            else discount_boost_config  # type: ignore[assignment]
        )

    # --- Cohort setup ---

    def _generate_cohort_dates(self) -> npt.NDArray[np.datetime64]:
        """Return array of monthly cohort start dates from the date range.

        Returns
        -------
        ndarray of datetime64[D], shape (n_cohorts,)
            First day of each cohort month.
        """
        start = self.date_range.start_date
        end = self.date_range.end_date
        dates: list[np.datetime64] = []
        year, month = start.year, start.month
        while datetime(year, month, 1, tzinfo=end.tzinfo) <= end:
            dates.append(np.datetime64(f"{year:04d}-{month:02d}-01"))
            month += 1
            if month > 12:
                month = 1
                year += 1
        return np.array(dates, dtype="datetime64[D]")

    def _generate_cohort_sizes(self, n_cohorts: int) -> npt.NDArray[np.int_]:
        """Draw number of new users per cohort (trend + Gamma noise).

        Parameters
        ----------
        n_cohorts : int
            Number of cohorts.

        Returns
        -------
        ndarray of int, shape (n_cohorts,)
            Number of new users joining in each cohort.
        """
        ones = np.ones(n_cohorts)
        trend = ones.cumsum() / ones.sum()
        sizes = (
            self.cohort_cfg.base_cohort_size
            * trend
            * self.rng.gamma(shape=1.0, scale=1.0, size=n_cohorts)
        )
        return np.maximum(sizes.round().astype(int), 1)

    # --- User-level latent parameters (BG/NBD + Gamma-Gamma) ---

    def _assign_users_to_cohorts(
        self, cohort_sizes: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.int_]:
        """Map each user to a cohort index via ``np.repeat``.

        Parameters
        ----------
        cohort_sizes : ndarray of int, shape (n_cohorts,)
            Number of new users in each cohort.

        Returns
        -------
        ndarray of int, shape (n_total_users,)
            Cohort index for every user.
        """
        return np.repeat(np.arange(len(cohort_sizes)), cohort_sizes)

    def _draw_transaction_rates(self, n_users: int) -> npt.NDArray[np.float64]:
        """Draw per-user transaction rate lambda_i ~ Gamma(r, 1/alpha).

        Parameters
        ----------
        n_users : int
            Total number of users across all cohorts.

        Returns
        -------
        ndarray of float, shape (n_users,)
            Individual transaction rates.
        """
        return self.rng.gamma(
            shape=self.bgnbd.r,
            scale=1.0 / self.bgnbd.alpha,
            size=n_users,
        )

    def _draw_dropout_probabilities(self, n_users: int) -> npt.NDArray[np.float64]:
        """Draw per-user dropout probability p_i ~ Beta(a, b).

        Parameters
        ----------
        n_users : int
            Total number of users across all cohorts.

        Returns
        -------
        ndarray of float, shape (n_users,)
            Individual dropout probabilities in (0, 1).
        """
        return self.rng.beta(
            a=self.bgnbd.a,
            b=self.bgnbd.b,
            size=n_users,
        )

    def _draw_spending_rates(self, n_users: int) -> npt.NDArray[np.float64]:
        """Draw per-user spending rate v_i ~ Gamma(q_spend, 1/gamma_spend).

        Parameters
        ----------
        n_users : int
            Total number of users across all cohorts.

        Returns
        -------
        ndarray of float, shape (n_users,)
            Individual spending-rate parameters for the Gamma-Gamma model.
        """
        return self.rng.gamma(
            shape=self.gg.q_spend,
            scale=1.0 / self.gg.gamma_spend,
            size=n_users,
        )

    def _draw_death_months(
        self, dropout_probs: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.int_]:
        """Draw per-user churn time death_month_i ~ Geometric(p_i).

        The user is active for months 0, 1, ..., death_month_i - 1
        relative to their cohort start.

        Parameters
        ----------
        dropout_probs : ndarray of float, shape (n_users,)
            Per-user dropout probabilities.

        Returns
        -------
        ndarray of int, shape (n_users,)
            Number of active months for each user (>= 1).
        """
        return self.rng.geometric(p=dropout_probs)

    # --- Time-varying rate modifiers ---

    def _compute_seasonal_factors(
        self, cohort_dates: npt.NDArray[np.datetime64]
    ) -> npt.NDArray[np.float64]:
        """Compute the seasonal multiplier for each calendar month.

        Uses a harmonic pattern controlled by ``seasonality_config.strength``.
        The pattern peaks around December/January and has a secondary
        harmonic that creates the multi-modal shape visible in the reference
        retention plot.

        Parameters
        ----------
        cohort_dates : ndarray of datetime64[D], shape (n_periods,)
            Calendar dates for each period.

        Returns
        -------
        ndarray of float, shape (n_periods,)
            Multiplicative seasonal factor (clipped to stay positive).
        """
        months = cohort_dates.astype("datetime64[M]").astype(int) % 12 + 1
        month_frac = 2.0 * np.pi * months / 12.0
        c1 = 0.8
        c2 = 0.5
        raw = self.seasonality.strength * (
            c1 * np.cos(month_frac) + c2 * np.sin(3.0 * month_frac)
        )
        return np.clip(1.0 + raw, a_min=0.1, a_max=None)

    def _compute_discount_boost_matrix(
        self,
        cohort_indices: npt.NDArray[np.int_],
        cohort_dates: npt.NDArray[np.datetime64],
        n_periods: int,
    ) -> npt.NDArray[np.float64]:
        """Build the discount-boost multiplier for each (user, period) cell.

        Cohorts starting from the discount start date receive a linearly
        decaying uplift during the first ``boost_months`` periods.

        Parameters
        ----------
        cohort_indices : ndarray of int, shape (n_users,)
            Cohort index for every user.
        cohort_dates : ndarray of datetime64[D], shape (n_cohorts,)
            Calendar date for each cohort.
        n_periods : int
            Total number of monthly periods.

        Returns
        -------
        ndarray of float, shape (n_users, n_periods)
            Multiplicative boost factors (>= 1.0).
        """
        n_users = len(cohort_indices)
        boost_matrix = np.ones((n_users, n_periods), dtype=np.float64)

        if self.discount is None or self.date_range.discount_start_date is None:
            return boost_matrix

        discount_start_np = np.datetime64(
            f"{self.date_range.discount_start_date.year:04d}"
            f"-{self.date_range.discount_start_date.month:02d}-01"
        )
        discount_cohort_mask = cohort_dates >= discount_start_np

        period_idx = np.arange(n_periods)
        user_cohort_idx = cohort_indices

        cohort_age_matrix = period_idx[np.newaxis, :] - user_cohort_idx[:, np.newaxis]

        is_discount_user = discount_cohort_mask[user_cohort_idx]

        decay = np.clip(
            1.0 - cohort_age_matrix / self.discount.boost_months, a_min=0.0, a_max=1.0
        )
        boost_values = 1.0 + self.discount.boost_factor * decay

        return np.where(
            is_discount_user[:, np.newaxis] & (cohort_age_matrix >= 0),
            boost_values,
            1.0,
        )

    # --- Active-user matrix and transaction counts ---

    def _build_active_mask(
        self,
        cohort_indices: npt.NDArray[np.int_],
        death_months: npt.NDArray[np.int_],
        n_periods: int,
    ) -> npt.NDArray[np.bool_]:
        """Build a boolean mask indicating whether each user is alive.

        A user is active in period ``t`` if ``cohort_idx <= t`` and
        ``t - cohort_idx < death_month``.

        Parameters
        ----------
        cohort_indices : ndarray of int, shape (n_users,)
            Cohort index for every user.
        death_months : ndarray of int, shape (n_users,)
            Number of active months for each user.
        n_periods : int
            Total number of monthly periods.

        Returns
        -------
        ndarray of bool, shape (n_users, n_periods)
            True where the user is alive in that period.
        """
        period_idx = np.arange(n_periods)[np.newaxis, :]
        cohort_idx = cohort_indices[:, np.newaxis]
        cohort_age = period_idx - cohort_idx
        return (cohort_age >= 0) & (cohort_age < death_months[:, np.newaxis])

    def _draw_transaction_counts(
        self,
        transaction_rates: npt.NDArray[np.float64],
        seasonal_factors: npt.NDArray[np.float64],
        discount_boost: npt.NDArray[np.float64],
        active_mask: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.int_]:
        """Draw the number of transactions per (user, period) cell.

        Samples from Poisson(lambda_i * seasonal * boost) for active cells;
        inactive cells are set to zero.

        Parameters
        ----------
        transaction_rates : ndarray of float, shape (n_users,)
            Per-user base transaction rates.
        seasonal_factors : ndarray of float, shape (n_periods,)
            Seasonal multiplier per period.
        discount_boost : ndarray of float, shape (n_users, n_periods)
            Discount-boost multiplier per (user, period).
        active_mask : ndarray of bool, shape (n_users, n_periods)
            True where the user is alive.

        Returns
        -------
        ndarray of int, shape (n_users, n_periods)
            Transaction counts (zero for inactive cells).
        """
        rate_matrix = (
            transaction_rates[:, np.newaxis]
            * seasonal_factors[np.newaxis, :]
            * discount_boost
        )
        counts = self.rng.poisson(lam=rate_matrix)
        return np.where(active_mask, counts, 0)

    # --- Expand to transaction-level rows ---

    def _expand_to_transactions(
        self,
        transaction_counts: npt.NDArray[np.int_],
        cohort_indices: npt.NDArray[np.int_],
        cohort_dates: npt.NDArray[np.datetime64],
        spending_rates: npt.NDArray[np.float64],
    ) -> pl.DataFrame:
        """Expand (user, period) counts into individual transaction rows.

        For each transaction, assigns a uniformly random day within the
        calendar month and draws an amount ~ Gamma(p_spend, v_i).

        Parameters
        ----------
        transaction_counts : ndarray of int, shape (n_users, n_periods)
            Number of transactions per (user, period).
        cohort_indices : ndarray of int, shape (n_users,)
            Cohort index for every user.
        cohort_dates : ndarray of datetime64[D], shape (n_cohorts,)
            Calendar date for each cohort index.
        spending_rates : ndarray of float, shape (n_users,)
            Per-user spending-rate parameter v_i.

        Returns
        -------
        pl.DataFrame
            Columns: ``user_id``, ``cohort``, ``transaction_date``, ``amount``.
        """
        user_idx, period_idx = np.nonzero(transaction_counts)
        counts_nz = transaction_counts[user_idx, period_idx]

        if counts_nz.sum() == 0:
            return pl.DataFrame(
                schema={
                    "user_id": pl.Int64,
                    "cohort": pl.Date,
                    "transaction_date": pl.Date,
                    "amount": pl.Float64,
                }
            )

        txn_user_idx = np.repeat(user_idx, counts_nz)
        txn_period_idx = np.repeat(period_idx, counts_nz)

        n_txn = len(txn_user_idx)

        txn_period_dates = cohort_dates[txn_period_idx]

        days_in_month = np.array(
            [
                calendar.monthrange(
                    int(d.astype("datetime64[Y]").astype(int) + 1970),
                    int(d.astype("datetime64[M]").astype(int) % 12 + 1),
                )[1]
                for d in txn_period_dates
            ]
        )
        random_days = self.rng.integers(low=0, high=days_in_month)
        txn_dates = txn_period_dates + random_days.astype("timedelta64[D]")

        txn_spending_rates = spending_rates[txn_user_idx]
        txn_amounts = self.rng.gamma(
            shape=self.gg.p_spend,
            scale=txn_spending_rates,
            size=n_txn,
        )

        txn_cohort_dates = cohort_dates[cohort_indices[txn_user_idx]]

        return pl.DataFrame(
            {
                "user_id": txn_user_idx.astype(np.int64),
                "cohort": txn_cohort_dates,
                "transaction_date": txn_dates,
                "amount": txn_amounts,
            }
        ).sort("user_id", "transaction_date")

    # --- Orchestration ---

    def run(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Run the full simulation pipeline.

        Returns
        -------
        transactions_df : pl.DataFrame
            Transaction-level data with columns
            (``user_id``, ``cohort``, ``transaction_date``, ``amount``).
        users_df : pl.DataFrame
            Full user roster with columns (``user_id``, ``cohort``),
            including users with zero transactions after their first period.
        """
        cohort_dates = self._generate_cohort_dates()
        n_cohorts = len(cohort_dates)
        logging.info(
            f"Generated {n_cohorts} cohorts"
            f" from {cohort_dates[0]} to {cohort_dates[-1]}"
        )

        cohort_sizes = self._generate_cohort_sizes(n_cohorts)
        n_total_users = int(cohort_sizes.sum())
        logging.info(f"Total users: {n_total_users}")

        cohort_indices = self._assign_users_to_cohorts(cohort_sizes)
        transaction_rates = self._draw_transaction_rates(n_total_users)
        dropout_probs = self._draw_dropout_probabilities(n_total_users)
        spending_rates = self._draw_spending_rates(n_total_users)
        death_months = self._draw_death_months(dropout_probs)

        seasonal_factors = self._compute_seasonal_factors(cohort_dates)
        discount_boost = self._compute_discount_boost_matrix(
            cohort_indices, cohort_dates, n_cohorts
        )

        active_mask = self._build_active_mask(cohort_indices, death_months, n_cohorts)
        transaction_counts = self._draw_transaction_counts(
            transaction_rates, seasonal_factors, discount_boost, active_mask
        )

        logging.info(f"Total transactions: {int(transaction_counts.sum())}")

        transactions_df = self._expand_to_transactions(
            transaction_counts, cohort_indices, cohort_dates, spending_rates
        )

        users_df = pl.DataFrame(
            {
                "user_id": np.arange(n_total_users, dtype=np.int64),
                "cohort": cohort_dates[cohort_indices],
            }
        )

        return transactions_df, users_df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_transactions_to_cohort(
    transactions_df: pl.DataFrame,
    users_df: pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate user-level transactions to cohort-period summaries.

    Produces output comparable to ``CohortDataGenerator.run()`` from
    ``retention_data.py`` (without the latent ``retention_true_mu`` /
    ``retention_true`` columns, which do not apply to the BG/NBD
    generative process).

    Parameters
    ----------
    transactions_df : pl.DataFrame
        Transaction-level data with columns
        (``user_id``, ``cohort``, ``transaction_date``, ``amount``).
    users_df : pl.DataFrame
        Full user roster with columns (``user_id``, ``cohort``),
        including users with zero transactions after their first period.

    Returns
    -------
    pl.DataFrame
        Cohort-period summary with columns:

        - ``cohort`` (date) -- cohort month.
        - ``period`` (date) -- calendar month.
        - ``n_users`` (int) -- total users in the cohort.
        - ``n_active_users`` (int) -- users with >= 1 transaction in the period.
        - ``revenue`` (float) -- sum of transaction amounts in the period.
        - ``retention`` (float) -- ``n_active_users / n_users``.
        - ``cohort_age`` (int) -- months elapsed since cohort start.
    """
    cohort_user_counts = users_df.group_by("cohort").agg(
        pl.col("user_id").count().alias("n_users")
    )

    all_cohorts = users_df.select("cohort").unique().sort("cohort")
    all_periods = all_cohorts.select(pl.col("cohort").alias("period"))

    cohort_period_grid = all_cohorts.join(all_periods, how="cross").filter(
        pl.col("period") >= pl.col("cohort")
    )

    txn_with_period = transactions_df.with_columns(
        pl.col("transaction_date").dt.truncate("1mo").alias("period")
    )

    txn_agg = txn_with_period.group_by("cohort", "period").agg(
        pl.col("user_id").n_unique().alias("n_active_users"),
        pl.col("amount").sum().alias("revenue"),
    )

    result = (
        cohort_period_grid.join(txn_agg, on=["cohort", "period"], how="left")
        .join(cohort_user_counts, on="cohort", how="left")
        .with_columns(
            pl.col("n_active_users").fill_null(0),
            pl.col("revenue").fill_null(0.0),
        )
    )

    result = result.with_columns(
        (
            (pl.col("period").dt.year() - pl.col("cohort").dt.year()) * 12
            + (pl.col("period").dt.month() - pl.col("cohort").dt.month())
        ).alias("cohort_age")
    )

    result = result.with_columns(
        pl.when(pl.col("cohort_age") == 0)
        .then(pl.col("n_users"))
        .otherwise(pl.col("n_active_users"))
        .alias("n_active_users")
    )

    result = result.with_columns(
        (pl.col("n_active_users") / pl.col("n_users")).alias("retention")
    )

    return result.select(
        "cohort",
        "period",
        "n_users",
        "n_active_users",
        "revenue",
        "retention",
        "cohort_age",
    ).sort("cohort", "period")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    FORMAT: str = "%(message)s"
    logging.basicConfig(
        level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )

    seed: int = sum(map(ord, "retention_v2"))
    rng: np.random.Generator = np.random.default_rng(seed=seed)

    date_range = DateRangeConfig(
        start_date=datetime(2020, 1, 1, tzinfo=UTC),
        end_date=datetime(2022, 12, 1, tzinfo=UTC),
        discount_start_date=datetime(2022, 4, 1, tzinfo=UTC),
    )

    generator = UserLevelCohortDataGenerator(
        rng=rng,
        date_range_config=date_range,
    )

    logging.info("Generating user-level transaction data...")
    transactions_df, users_df = generator.run()
    logging.info(
        f"Transactions shape: {transactions_df.shape}, Users shape: {users_df.shape}"
    )

    logging.info("Aggregating to cohort level...")
    cohort_df = aggregate_transactions_to_cohort(transactions_df, users_df)
    logging.info(f"Cohort summary shape: {cohort_df.shape}")

    transactions_df.write_csv("data/retention_user_transactions.csv")
    logging.info("Saved transactions to data/retention_user_transactions.csv")

    cohort_df.write_csv("data/retention_user_cohort.csv")
    logging.info("Saved cohort summary to data/retention_user_cohort.csv")

    logging.info("Done!")
    logging.info(f"\n{cohort_df.head(10)}")

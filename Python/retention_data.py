import logging
from datetime import UTC, datetime

import numpy as np
import numpy.typing as npt
import polars as pl
from rich.logging import RichHandler
from scipy.special import expit


class CohortDataGenerator:
    def __init__(
        self,
        rng: np.random.Generator,
        start_cohort: datetime,
        n_cohorts,
        user_base: int = 10_000,
    ) -> None:
        self.rng = rng
        self.start_cohort = start_cohort
        self.n_cohorts = n_cohorts
        self.user_base = user_base

    def _generate_cohort_labels(self) -> pl.Series:
        # Calculate end date by adding n_cohorts months to start date
        start_date = self.start_cohort
        # Add (n_cohorts - 1) months to get exactly n_cohorts periods
        total_months = start_date.month + self.n_cohorts - 1
        year = start_date.year + (total_months - 1) // 12
        month = (total_months - 1) % 12 + 1
        end_date = datetime(year, month, 1, tzinfo=start_date.tzinfo)

        logging.info(
            f"""
        Generating cohort labels from
        {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}
            """
        )
        return pl.date_range(start=start_date, end=end_date, interval="1mo", eager=True)

    def _generate_cohort_sizes(self) -> npt.NDArray[np.int_]:
        ones = np.ones(shape=self.n_cohorts)
        trend = ones.cumsum() / ones.sum()
        return (
            (
                self.user_base
                * trend
                * self.rng.gamma(shape=1, scale=1, size=self.n_cohorts)
            )
            .round()
            .astype(int)
        )

    def _generate_dataset_base(self) -> pl.DataFrame:
        cohorts = self._generate_cohort_labels()
        n_users = self._generate_cohort_sizes()

        # Create base dataframes
        cohort_df = pl.DataFrame({"cohort": cohorts, "n_users": n_users})

        period_df = pl.DataFrame({"period": cohorts})

        # Cross join
        data_df = cohort_df.join(period_df, how="cross")

        # Calculate age and cohort_age
        max_period = data_df["period"].max()
        data_df = data_df.with_columns(
            [
                ((max_period - pl.col("cohort")).dt.total_days()).alias("age"),
                ((pl.col("period") - pl.col("cohort")).dt.total_days()).alias(
                    "cohort_age"
                ),
            ]
        )

        return data_df.filter(pl.col("cohort_age") >= 0)

    def _generate_retention_rates(self, data_df: pl.DataFrame) -> pl.DataFrame:
        data_df = data_df.with_columns(
            [
                (
                    -pl.col("cohort_age") / (pl.col("age") + 1)
                    + 0.8 * np.cos(2 * np.pi * pl.col("period").dt.ordinal_day() / 365)
                    + 0.5
                    * np.sin(2 * 3 * np.pi * pl.col("period").dt.ordinal_day() / 365)
                    - 0.5 * np.log1p(pl.col("age"))
                    + 1.0
                ).alias("retention_true_mu")
            ]
        )

        return data_df.with_columns(
            [expit(data_df["retention_true_mu"]).alias("retention_true")]
        )

    def _generate_user_history(self, data_df: pl.DataFrame) -> pl.DataFrame:
        # Convert to numpy for binomial sampling, then back to polars
        n_active_users = self.rng.binomial(
            n=data_df["n_users"].to_numpy(), p=data_df["retention_true"].to_numpy()
        )

        # Set cohort_age == 0 to n_users
        n_active_users = np.where(
            data_df["cohort_age"].to_numpy() == 0,
            data_df["n_users"].to_numpy(),
            n_active_users,
        )

        data_df = data_df.with_columns([pl.Series("n_active_users", n_active_users)])

        # Calculate lambda for revenue generation
        lam = 1e-2 * np.exp(
            data_df["cohort_age"].to_numpy() / (data_df["age"].to_numpy() + 1)
            + (data_df["cohort_age"].to_numpy() / data_df["cohort_age"].max())
        )

        revenue = self.rng.gamma(
            shape=data_df["n_active_users"].to_numpy(), scale=1 / lam
        )

        return data_df.with_columns([pl.Series("revenue", revenue)])

    def run(
        self,
    ) -> pl.DataFrame:
        return (
            self._generate_dataset_base()
            .pipe(self._generate_retention_rates)
            .pipe(self._generate_user_history)
        )


if __name__ == "__main__":
    FORMAT: str = "%(message)s"
    logging.basicConfig(
        level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )

    logging.info("Generating data...")
    seed: int = sum(map(ord, "retention"))
    rng: np.random.Generator = np.random.default_rng(seed=seed)

    start_cohort: datetime = datetime(2020, 1, 1, tzinfo=UTC)
    n_cohorts: int = 48

    cohort_generator = CohortDataGenerator(
        rng=rng, start_cohort=start_cohort, n_cohorts=n_cohorts
    )
    data_df = cohort_generator.run()
    data_df = data_df.with_columns(
        [(pl.col("n_active_users") / pl.col("n_users")).alias("retention")]
    )
    logging.info("Data generation complete!")

    data_df.write_csv("data/retention_data.csv")
    logging.info("Data saved to data/retention_data.csv")

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
from rich.logging import RichHandler
from scipy.special import expit


class CohortDataGenerator:
    def __init__(
        self,
        rng: np.random.Generator,
        start_cohort: str,
        n_cohorts,
        user_base: int = 10_000,
    ) -> None:
        self.rng = rng
        self.start_cohort = start_cohort
        self.n_cohorts = n_cohorts
        self.user_base = user_base

    def _generate_cohort_labels(self) -> pd.DatetimeIndex:
        return pd.period_range(
            start="2020-01-01", periods=self.n_cohorts, freq="M"
        ).to_timestamp()

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

    def _generate_dataset_base(self) -> pd.DataFrame:
        cohorts = self._generate_cohort_labels()
        n_users = self._generate_cohort_sizes()
        data_df = pd.merge(
            left=pd.DataFrame(data={"cohort": cohorts, "n_users": n_users}),
            right=pd.DataFrame(data={"period": cohorts}),
            how="cross",
        )
        data_df["age"] = (data_df["period"].max() - data_df["cohort"]).dt.days
        data_df["cohort_age"] = (data_df["period"] - data_df["cohort"]).dt.days
        data_df = data_df.query("cohort_age >= 0")
        return data_df

    def _generate_retention_rates(self, data_df: pd.DataFrame) -> pd.DataFrame:
        data_df["retention_true_mu"] = (
            -data_df["cohort_age"] / (data_df["age"] + 1)
            + 0.8 * np.cos(2 * np.pi * data_df["period"].dt.dayofyear / 365)
            + 0.5 * np.sin(2 * 3 * np.pi * data_df["period"].dt.dayofyear / 365)
            - 0.5 * np.log1p(data_df["age"])
            + 1.0
        )
        data_df["retention_true"] = expit(data_df["retention_true_mu"])
        return data_df

    def _generate_user_history(self, data_df: pd.DataFrame) -> pd.DataFrame:
        data_df["n_active_users"] = self.rng.binomial(
            n=data_df["n_users"], p=data_df["retention_true"]
        )
        data_df["n_active_users"] = np.where(
            data_df["cohort_age"] == 0, data_df["n_users"], data_df["n_active_users"]
        )

        lam = 1e-2 * np.exp(
            data_df["cohort_age"] / (data_df["age"] + 1)
            + (data_df["cohort_age"] / data_df["cohort_age"].max())
        )

        data_df["revenue"] = self.rng.gamma(
            shape=data_df["n_active_users"], scale=1 / lam
        )

        return data_df

    def run(
        self,
    ) -> pd.DataFrame:
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

    start_cohort: str = "2020-01-01"
    n_cohorts: int = 48

    cohort_generator = CohortDataGenerator(
        rng=rng, start_cohort=start_cohort, n_cohorts=n_cohorts
    )
    data_df = cohort_generator.run()
    data_df["retention"] = data_df["n_active_users"] / data_df["n_users"]
    logging.info("Data generation complete!")

    data_df.to_csv("data/retention_data.csv", index=False)
    logging.info("Data saved to data/retention_data.csv")

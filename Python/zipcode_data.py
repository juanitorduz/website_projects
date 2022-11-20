from datetime import datetime
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.special import expit


class ZipCodeDataGenerator:
    """Class to generate data for a simulated geo-experiment of orders by zipcode.

    Orders are modeled as a binomial distribution over the zipcode population.
    The order rate is a function of a trend, seasonal components, the zipcode strength
    (unobserved variable) and the campaign effect for the treatment group.
    """

    def __init__(
        self,
        n_zipcodes: int,
        start_date: datetime,
        end_date: datetime,
        freq: str,
        start_campaign: datetime,
        rng: np.random.Generator,
    ) -> None:
        self.n_zipcodes = n_zipcodes
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.start_campaign = start_campaign
        self.rng = rng

    def generate_zipcodes_ids(self) -> npt.ArrayLike:
        return np.arange(start=0, stop=self.n_zipcodes, step=1)

    def generate_dates(self) -> npt.ArrayLike:
        return pd.date_range(start=self.start_date, end=self.end_date, freq=self.freq)

    def generate_base_df(
        self, zipcodes: npt.ArrayLike, date_range: npt.ArrayLike
    ) -> pd.DataFrame:
        data_df: pd.DataFrame = pd.merge(
            left=pd.DataFrame(data={"date": date_range}),
            right=pd.DataFrame(data={"zipcode": zipcodes}),
            how="cross",
        )
        data_df["is_campaign"] = data_df["date"] >= self.start_campaign
        return data_df

    def add_seasonality_features(self, data_df: pd.DataFrame) -> pd.DataFrame:
        data_df["is_weekend"] = data_df["date"].dt.weekday > 4
        data_df["is_weekday"] = ~data_df["is_weekend"]
        data_df["norm_trend"] = np.linspace(start=0, stop=1, num=data_df.shape[0])
        return data_df

    def generate_zipcodes_population(
        self, data_df: pd.DataFrame, zipcodes: npt.ArrayLike
    ) -> pd.DataFrame:
        zipcodes_population_map: dict[int, int] = {
            zipcode: int(10_000 * self.rng.gamma(shape=(zipcode % 3) + 1, scale=1))
            for zipcode in zipcodes
        }
        data_df["population"] = data_df["zipcode"].map(zipcodes_population_map)
        return data_df

    def generate_strength_feature(
        self, data_df: pd.DataFrame, zipcodes: npt.ArrayLike
    ) -> pd.DataFrame:
        zipcodes_strength_map: dict[int, int] = {
            zipcode: (zipcode < (self.n_zipcodes // 2)).astype(int)
            for zipcode in zipcodes
        }
        data_df["strength"] = data_df["zipcode"].map(zipcodes_strength_map)
        return data_df

    def generate_variant_tag(
        self, data_df: pd.DataFrame, zipcodes: npt.ArrayLike
    ) -> pd.DataFrame:
        zipcode_variant_map: dict[int, int] = {
            zipcode: (zipcode < (self.n_zipcodes // 3)).astype(int)
            for zipcode in zipcodes
        }
        data_df["variant"] = (
            data_df["zipcode"]
            .map(zipcode_variant_map)
            .map({0: "control", 1: "treatment"})
        )
        mask = data_df["is_campaign"] & (data_df["variant"] == "treatment")
        data_df["is_campaign_treatment"] = mask
        return data_df

    def generate_order_rate(self, data_df: pd.DataFrame) -> pd.DataFrame:

        base_or: dict[int, float] = {
            0: 0.4,
            1: 0.6,
        }  # base conversion rate depends on the strength level
        treatment_effect: float = 7e-2

        data_df["order_rate_true_logit"] = (
            data_df["strength"].map(base_or)
            + (data_df["is_campaign_treatment"] * treatment_effect)
            - 5e-2 * data_df["is_weekday"]
            + 6e-2 * data_df["norm_trend"]
            - 2.3
        )

        data_df["order_rate_true_logit_no_treatment"] = data_df[
            "order_rate_true_logit"
        ] - (data_df["is_campaign_treatment"] * treatment_effect)

        data_df["order_rate_true"] = expit(data_df["order_rate_true_logit"])
        data_df["order_rate_true_no_treatment"] = expit(
            data_df["order_rate_true_logit_no_treatment"]
        )
        return data_df

    def generate_orders(self, data_df: pd.DataFrame) -> pd.DataFrame:
        data_df["orders"] = rng.binomial(
            n=data_df["population"], p=data_df["order_rate_true"]
        )
        data_df["orders_no_treatment"] = rng.binomial(
            n=data_df["population"], p=data_df["order_rate_true_no_treatment"]
        )
        data_df["order_rate"] = data_df["orders"] / data_df["population"]
        data_df["order_rate_no_treatment"] = (
            data_df["orders_no_treatment"] / data_df["population"]
        )
        data_df["expected_orders"] = data_df["population"] * data_df["order_rate_true"]
        return data_df

    def run(self) -> pd.DataFrame:
        zipcodes = self.generate_zipcodes_ids()
        date_range = self.generate_dates()
        data_df = self.generate_base_df(zipcodes=zipcodes, date_range=date_range)
        data_df = self.add_seasonality_features(data_df=data_df)
        data_df = self.generate_zipcodes_population(data_df=data_df, zipcodes=zipcodes)
        data_df = self.generate_strength_feature(data_df=data_df, zipcodes=zipcodes)
        data_df = self.generate_variant_tag(data_df=data_df, zipcodes=zipcodes)
        data_df = self.generate_order_rate(data_df=data_df)
        data_df = self.generate_orders(data_df=data_df)
        return data_df


if __name__ == "__main__":
    seed: int = sum(map(ord, "wolt"))
    rng: np.random.Generator = np.random.default_rng(seed=seed)
    n_zipcodes: int = 100
    start_date: datetime = datetime(year=2022, month=4, day=1)
    end_date: datetime = datetime(year=2022, month=7, day=31)
    freq: str = "D"
    start_campaign: datetime = datetime(year=2022, month=7, day=1)

    zipcode_data_generator = ZipCodeDataGenerator(
        n_zipcodes=n_zipcodes,
        start_date=start_date,
        end_date=end_date,
        freq=freq,
        start_campaign=start_campaign,
        rng=rng,
    )

    data_df: pd.DataFrame = zipcode_data_generator.run()

    data_df.to_csv("data/zipcodes_data.csv", index=False)

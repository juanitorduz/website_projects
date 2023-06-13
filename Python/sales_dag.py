import logging

import numpy as np
import pandas as pd
import pymc as pm
from rich.logging import RichHandler


def generate_users_observational_data(
    rng: np.random.Generator,
    n_users: int,
    sales_discount_threshold: int,
    sales_loyalty_threshold: int,
    discount_probability: float,
    rate: float,
    discount_effect: float,
    visits_mu: float = 20,
    visits_sigma: float = 100,
    sales_sigma: float = 2,
) -> pd.DataFrame:
    """
    Generate a dataframe of users with visits, sales, and loyalty according to the DAG:

    >> import graphviz as gr
    >> g = gr.Digraph()
    >> g.node(name="sales", label="sales", color="deepskyblue", style="filled")
    >> g.node(name="discount", label="discount", color="deeppink", style="filled")

    >> g.edge(tail_name="discount", head_name="sales")
    >> g.edge(tail_name="visits", head_name="discount")
    >> g.edge(tail_name="visits", head_name="sales")
    >> g.edge(tail_name="discount", head_name="is_loyal")
    >> g.edge(tail_name="sales", head_name="is_loyal")
    """
    visits_dist = pm.NegativeBinomial.dist(mu=visits_mu, alpha=visits_sigma)
    visits_samples = pm.draw(vars=visits_dist, draws=n_users, random_seed=rng) + 1

    is_discount_candidate = visits_samples > sales_discount_threshold
    discount_distribution = pm.Bernoulli.dist(
        p=is_discount_candidate * discount_probability
    )
    discount_samples = pm.draw(vars=discount_distribution, random_seed=rng)

    sales_distribution = pm.Gamma.dist(
        mu=rate * visits_samples + discount_effect * discount_samples + 1,
        sigma=sales_sigma,
    )
    sales_samples = pm.draw(vars=sales_distribution, random_seed=rng)

    is_loyal = (discount_samples + (sales_samples > sales_loyalty_threshold)) > 0

    return pd.DataFrame(
        data={
            "visits": visits_samples,
            "discount": discount_samples,
            "is_loyal": is_loyal.astype(int),
            "sales": sales_samples,
            "sales_per_visit": sales_samples / visits_samples,
        }
    )


if __name__ == "__main__":
    FORMAT: str = "%(message)s"
    logging.basicConfig(
        level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )

    logging.info("Setting parameters...")
    seed: int = sum(map(ord, "causality"))
    rng: np.random.Generator = np.random.default_rng(seed=seed)

    sales_discount_threshold = 14
    sales_loyalty_threshold = sales_discount_threshold + 11
    discount_probability = 0.8
    rate = 0.8
    discount_effect = 2

    logging.info("Generating data...")
    data = generate_users_observational_data(
        rng=rng,
        n_users=700,
        sales_discount_threshold=sales_discount_threshold,
        sales_loyalty_threshold=sales_loyalty_threshold,
        discount_probability=discount_probability,
        rate=rate,
        discount_effect=discount_effect,
    )

    logging.info("Saving data...")
    data.to_csv("data/sales_dag.csv", index=False)
    logging.info("Data saved to data/sales_dag.csv")

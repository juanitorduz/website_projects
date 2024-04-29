eps = np.finfo(float).eps

coords = {"id": id_train}

with pm.Model(coords=coords) as model:
    # --- Data Containers ---

    t_train_max = t_train.max()

    model.add_coord(name="n", values=range(n_train))
    model.add_coord(name="t", values=t_train)

    idx_id_data = pm.Data(name="idx_id_data", value=idx_id_train, dims="n")

    t_data = pm.Data(name="t_data", value=t_train, dims="t")
    idx_t_data = pm.Data(name="idx_t_data", value=idx_t_train, dims="n")

    rec_data = pm.Data(name="rec_data", value=rec_train, dims="n")
    life_data = pm.Data(name="life_data", value=life_train, dims="n")
    pnum_data = pm.Data(name="pnum_data", value=pnum_train, dims="n")

    # --- Priors ---

    ## Weekly seasonality
    eta_week = pm.LogNormal(name="eta_week", mu=0, sigma=3)
    rho_week = pm.LogNormal(name="rho_week", mu=0, sigma=2)
    cov_week = pm.gp.cov.Periodic(input_dim=1, period=7, ls=rho_week)
    gp_alpha_week = pm.gp.HSGPPeriodic(m=20, scale=1, cov_func=cov_week)
    f_alpha_week = gp_alpha_week.prior(
        name="f_alpha_week", X=t_data[idx_t_data][:, None], dims="n"
    )

    ## Long term trend
    eta_long = pm.HalfNormal(name="eta_long", sigma=5)
    rho_long = pm.LogNormal(name="rho_long", mu=0, sigma=2)
    cov_long = eta_long**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=rho_long)
    gp_alpha_long = pm.gp.HSGP(m=[20], L=[120], cov_func=cov_long)
    f_alpha_long = gp_alpha_long.prior(
        name="f_alpha_long", X=t_data[idx_t_data][:, None], dims="n"
    )

    # ## Short term trend
    # eta_short = pm.HalfNormal(name="eta_short", sigma=5)
    # rho_short = pm.LogNormal(name="rho_short", mu=0, sigma=2)
    # cov_short = eta_short**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=rho_short)
    # gp_alpha_short = pm.gp.HSGP(m=[20], L=[120], cov_func=cov_short)
    # f_alpha_short = gp_alpha_short.prior(
    #     name="f_alpha_short", X=t_data[idx_t_data][:, None], dims="n"
    # )

    # ## Lifetime effect
    # eta_life = pm.HalfNormal(name="eta_life", sigma=5)
    # rho_life = pm.LogNormal(name="rho_life", mu=0, sigma=2)
    # cov_life = eta_life**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=rho_life)
    # gp_alpha_life = pm.gp.HSGP(m=[20], L=[120], cov_func=cov_life)
    # f_alpha_life = gp_alpha_life.prior(
    #     name="f_alpha_life", X=life_data[:, None], dims="n"
    # )

    # ## Recency effect
    # eta_rec = pm.HalfNormal(name="eta_rec", sigma=5)
    # rho_rec = pm.LogNormal(name="rho_rec", mu=0, sigma=2)
    # cov_rec = eta_rec**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=rho_rec)
    # gp_alpha_rec = pm.gp.HSGP(m=[20], L=[120], cov_func=cov_rec)
    # f_alpha_rec = gp_alpha_rec.prior(name="f_alpha_rec", X=rec_data[:, None], dims="n")

    # ## Purchase number effect
    # eta_pnum = pm.HalfNormal(name="eta_pnum", sigma=5)
    # rho_pnum = pm.LogNormal(name="rho_pnum", mu=0, sigma=2)
    # cov_pnum = eta_pnum**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=rho_pnum)
    # gp_alpha_pnum = pm.gp.HSGP(m=[20], L=[120], cov_func=cov_pnum)
    # f_alpha_pnum = gp_alpha_pnum.prior(
    #     name="f_alpha_pnum", X=pnum_data[:, None], dims="n"
    # )

    # sigma = pm.HalfNormal(name="sigma", sigma=3)

    # delta = pm.Normal(name="delta", mu=0, sigma=sigma, dims="id")

    # --- Parametrization

    logit_p = pm.Deterministic(
        name="logistic_p",
        var=(
            f_alpha_week + f_alpha_long
            # + f_alpha_short
            # + f_alpha_life
            # + f_alpha_rec
            # + f_alpha_pnum
            # + delta[idx_id_data]
        ),
        dims="n",
    )
    p = pm.Deterministic(name="p", var=pm.math.invlogit(logit_p), dims="n")
    # We add a small epsilon to avoid numerical issues.
    p = pt.switch(pt.eq(p, 0), eps, p)
    p = pt.switch(pt.eq(p, 1), 1 - eps, p)

    # --- Likelihood ---
    pm.Bernoulli(name="y_obs", p=p, observed=y_train, dims="n")


pm.model_to_graphviz(model)
# Berlin Beyesians MeetUp

## Introduction to BTYD (Buy Until You Die) Models

### [Dr. Juan Orduz](https://juanitorduz.github.io/)

**Abstract:** In this talk we give an introduction to a certain types of customer lifetime models for the non-contractual setting commonly known as *BTYD* (Buy Until You Die) models. We focus in two sub-model components: the frequency BG/NBD model and the monetary gamma-gamma model. We begin by introducing the models assumptions and parameterizations. Then we walk through the maximum-likelihood parameter estimation and describe how the models are used in practice. Next, we describe some limitations and how the bayesian framework can help us to overcome some of them, plus allowing more flexibility. Finally, we describe some ongoing efforts in the open source community to bring these new ideas and models to the public.

### References

- [Fader, Harder and Lee; “Counting Your Customers” the Easy Way:
An Alternative to the Pareto/NBD Model](http://brucehardie.com/papers/018/fader_et_al_mksc_05.pdf)
- [Fader and Harder; The Gamma-Gamma Model of Monetary Value](https://brucehardie.com/notes/025/gamma_gamma.pdf)
- [Fader and Harder; Computing P(alive) Using the BG/NBD Model](https://brucehardie.com/notes/021/palive_for_BGNBD.pdf)
- [Fader and Harder; Incorporating Time-Invariant Covariates into the Pareto/NBD and BG/NBD Models](https://brucehardie.com/notes/019/time_invariant_covariates.pdf)
- [Davidson-Pilon; `lifetimes` package](https://github.com/CamDavidsonPilon/lifetimes)
- [Orduz; BG/NBD Model in PyMC](https://juanitorduz.github.io/bg_nbd_pymc/)
- [Orduz; Gamma-Gamma Model of Monetary Value in PyMC](https://juanitorduz.github.io/gamma_gamma_pymc/)

---
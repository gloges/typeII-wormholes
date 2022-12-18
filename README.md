# Type II AdS Wormholes

Construction of numerical wormhole solutions of type II SUGRA for both massive type IIA on $S^3\times S^3$ and type IIB on $T^{1,1}$ using a shooting method. These solutions provide explicit uplifts of regular 4D/5D AdS wormholes to 10D.

### For more information see our preprint:

>Gregory J. Loges, Gary Shiu and Thomas Van Riet, _TITLE TBD_, [[arXiv:xxxx.xxxxx](https://arxiv.org/abs/xxxx.xxxxx)].

---
## Brief background
Consistent truncations are identified for each model in the reduction to 4D/5D. To find wormhole solutions these necessarily include massive scalars. The actions for the relevant modes are $\ldots$

- Massive type IIA on $S^3\times S^3$:

$$ \begin{align*}
    S_4 &= \frac{1}{2\kappa_4^2}\int\Big({\star R} - \frac{1}{2}\mathrm{d}\phi\wedge{\star\mathrm{d}\phi} - 6\mathrm{d}u\wedge{\star\mathrm{d}u} + \frac{1}{2}e^{-3u+\phi/2}\mathrm{d}\chi\wedge{\star\mathrm{d}\chi} - {\star\mathcal{V}}\Big)\\
    \mathcal{V} &= -12e^{-4u} + 5e^{-9u-\phi/2} + e^{-3u+5\phi/2}
\end{align*} $$

- Type IIB on $T^{1,1}$:

$$ \begin{align*}
    S_5 &= \frac{1}{2\kappa_5^2}\int\Big({\star R} - \frac{1}{2}\mathrm{d}\phi\wedge{\star\mathrm{d}\phi} + \frac{1}{2}e^{2\phi}\mathrm{d}\chi\wedge{\star\mathrm{d}\chi}\\
    &\qquad\qquad\qquad -\frac{1}{2}e^{-4u-\phi}\mathrm{d}b\wedge{\star\mathrm{d}b} + \frac{1}{2}e^{-4u+\phi}(\mathrm{d}c - \chi\,\mathrm{d}b)\wedge{\star(\mathrm{d}c-\chi\,\mathrm{d}b)}\\
    &\qquad\qquad\qquad - \frac{28}{3}\mathrm{d}u\wedge{\star\mathrm{d}u} - \frac{8}{3}\mathrm{d}u\wedge{\star\mathrm{d}v} - \frac{4}{3}\mathrm{d}v\wedge{\star\mathrm{d}v} - {\star\mathcal{V}} \Big)\\
    \mathcal{V} &= 2e^{-\frac{8}{3}(4u+v)}\left( 2e^{4u+4v} - 12e^{6u+2v} + 4 \right)
\end{align*} $$

In both cases the metric ansatz is taken to be

$$ \mathrm{d}s_d^2 = f^2\mathrm{d}{r^2} + q^2\mathrm{d}\Omega_{d-1}^2 \qquad q(r) = \sqrt{q_0^2 + r^2} $$

so that the coordinate $r$ covers the full wormhole. Numerical AdS wormhole solutions to the equations of motion are found using a shooting method; "initial values" for $f$ and the scalars at $r=0$ are chosen so that solutions are smooth across $r=0$ (care is needed because $q'(0)=0$) and are parity even/odd, then these initial values are adjusted so that integrating out to large $r$ the solutions may be matched onto AdS boundary conditions ($f\sim r^{-1}$ and $u,v,\phi\to0$ for $r\to\infty$).

---
## Contents overview
- [typeIIA_S3xS3.nb](https://github.com/gloges/typeII-wormholes/blob/main/typeIIA_S3xS3.nb) and [typeIIB_T11.nb](https://github.com/gloges/typeII-wormholes/blob/main/typeIIB_T11.nb): Mathematica notebooks with derivations of the 4D/5D equations of motion directly from 10D, including cross-checks with the 4D/5D actions above, as well as the construction of perturbative solutions used to set the BCs at $r=0$ in the shooting method.
- [wormholes.py](https://github.com/gloges/typeII-wormholes/blob/main/wormholes.py): Implementation of the equations of motion and shooting method, as well as some functionality for fitting the scalars for $r\to\infty$.
- [wormholes.ipynb](https://github.com/gloges/typeII-wormholes/blob/main/wormholes.ipynb): Analysis and plots.


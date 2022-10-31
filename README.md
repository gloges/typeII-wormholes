# Type II Wormholes

Construct numerical wormhole solutions of type II SUGRA for...
 - massive type IIA on $S^3\times S^3$
 - type IIB on $T^{1,1}$

using a shooting method. "Initial conditions" are chosen at the wormhole throat, leveraging the known perturbative solution at the $\mathbb{Z}_2$-symmetric point, and by integrating outward the solution should match to AdS boundary conditions for $r\to\infty$.

---
## Background
Consistent truncations are identified for each model in the reduction to 4D and 5D. To find wormhole solutions these necessarily include massive scalars. The actions for the relevant modes are...


- Massive type IIA on $S^3\times S^3$:
$$ \begin{align*}
    S_4 &= \frac{1}{2\kappa_4^2}\int\Big({\star R} - \frac{1}{2}\mathrm{d}\phi\wedge{\star\mathrm{d}\phi} - 6\mathrm{d}u\wedge{\star\mathrm{d}u} + \frac{1}{2}e^{-3u+\phi/2}\mathrm{d}\chi\wedge{\star\mathrm{d}\chi} - {\star\mathcal{V}}\Big)\\
    \mathcal{V} &= -12e^{-4u} + 5e^{-9u-\phi/2} + e^{-3u+5\phi/2}
\end{align*} $$
- Type IIB on $T^{1,1}$:
$$ \begin{align*}
    S_5 &= \frac{1}{2\kappa_5^2}\int\Big({\star R} - \frac{1}{2}\mathrm{d}\phi\wedge{\star\mathrm{d}\phi} + \frac{1}{2}e^{-4u+\phi}\mathrm{d}\chi\wedge{\star\mathrm{d}\chi}\\
    &\qquad\qquad\qquad - \frac{28}{3}\mathrm{d}u\wedge{\star\mathrm{d}u} - \frac{8}{3}\mathrm{d}u\wedge{\star\mathrm{d}v} - \frac{4}{3}\mathrm{d}v\wedge{\star\mathrm{d}v} - {\star\mathcal{V}} \Big)\\
    \mathcal{V} &= 2e^{-\frac{8}{3}(4u+v)}\left( 2e^{4u+4v} - 12e^{6u+2v} + 4 \right)
\end{align*} $$

In both cases the metric ansatz is taken to be
$$ \mathrm{d}s_d^2 = \frac{\mathrm{d}{r^2}}{q^2w} + q^2\mathrm{d}\Omega_{d-1}^2 \;, \qquad q(r) = \sqrt{q_0^2 + r^2} \;, $$
so that the coordinate $r$ covers the full wormhole. AdS boundary conditions are $w\to1$ and $u,v,\phi\to0$.


---
## Contents
- [typeIIA_S3xs3.nb]() and [typeIIB_T11.nb](): derivation of 4D/5D equations of motion directly from 10D and cross-check with 4D/5D actions above, as well as construction of perturbative solutions used to set BCs at $r=0$ in the shooting method
- [wormholes.py](): implementation of equations of motion and shooting method
- [wormholes.ipynb](): analysis and plots
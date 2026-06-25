# celestial-fourpoint

A calculator evaluating four-point integrals over GW source directions, with each external leg chosen as a PTA or astrometric antenna response.

For unit sky directions $p_i$, helicities $\epsilon_i=\pm1$, and observables $X_i\in\{P,A\}$, the evaluated quantity is

$$
\int_{S^2}\frac{d\Omega}{4\pi}\prod_{i=1}^4 R^{\epsilon_i}_{X_i}(p_i,\Omega).
$$

Here $\Omega$ is the source direction. The PTA response is

$$
R_P^\epsilon(p,\Omega)=\frac{1}{2}\frac{p^a p^b e^\epsilon_{ab}(-\Omega)}{1-\Omega\cdot p}.
$$

The astrometric response is

$$
R_A^\epsilon(p,\Omega)=\epsilon\frac{\Delta p(p,\Omega;e^\epsilon(-\Omega))\cdot m_{-\epsilon}(p)}{-2\sqrt{2}}.
$$

where

$$
\Delta p^a(n,q;e)=\frac{1}{2}\left[\frac{n^a-q^a}{1-q\cdot n}n^b n^c e_{bc}-e^a_{\ b}n^b\right],\qquad m_\sigma(p)=\frac{e_\theta(p)+i\sigma e_\phi(p)}{\sqrt{2}}.
$$

The astrometric external leg is a spin-1 tangent response. The normalization above is chosen from the single-leg descendant relation to the scalar PTA response.

A cached finite-spectral approximation with $L=\ell_{\max}$ is provided. The default cached kernel uses $L=15$.

## Use

```python
import numpy as np

from celestial_fourpoint import FourPointCalculator

vectors = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
])
vectors[3] /= np.linalg.norm(vectors[3])
epsilons = (-1, 1, -1, 1)

calc = FourPointCalculator.load_precomputed(lmax=15, build_if_missing=True)
value = calc.evaluate(("A", "A", "A", "A"), vectors, epsilons)
```

Optional direct check:

```python
direct = calc.direct(("A", "A", "A", "A"), vectors, epsilons, backend="auto")
```

Build the cached kernel if missing:

```python
from celestial_fourpoint import build_precomputed_spectral_kernel

build_precomputed_spectral_kernel(lmax=15)
```

## TODO

- Generic n-point PTA/astrometry antenna integrals.
- Closed-form expressions and special limits.

## References

- Adrien Kuntz, Clemente Smarra, Massimo Vaglio, [Looking for non-gaussianity in Pulsar Timing Arrays through the four point correlator](https://arxiv.org/abs/2603.12311), arXiv:2603.12311.
- [akuntz00/PTA-4PT-correlator](https://github.com/akuntz00/PTA-4PT-correlator).

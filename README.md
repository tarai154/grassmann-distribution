

# Grassmann Distribution
A Python implementation of probability distributions and statistical models for mixed continuous and binary variables based on the Grassmann formalism [1,2,3].



## Requirements

* Python 3
* NumPy
* SciPy
* JAX
* scikit-learn
* Matplotlib
* Seaborn



## Installation
`pip install git+https://github.com/tarai154/grassmann-distribution.git`



## Notes
This package uses JAX. The default floating-point precision in JAX is 32-bit.  
However, fitting the model in this package requires 64-bit precision for reliable parameter estimation.  
Before fitting the model, enable JAX 64-bit mode at the beginning of your program:

```python
import jax
jax.config.update("jax_enable_x64", True)
```


## Usage
`GrassmannDistribution` follows the scikit-learn estimator API conventions where applicable.

```python
import jax
jax.config.update("jax_enable_x64", True)

from grassmann_distribution import GrassmannDistribution

model = GrassmannDistribution()
model.fit(X)

print(model.score(X))
print(model.get_corr())
```




`FactorAnalysis` follows the scikit-learn estimator API conventions where applicable.

```python
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import matplotlib.pyplot as plt
from grassmann_distribution import FactorAnalysis, simulate_dichotomized_gaussian

level_list = np.array([0,0,1,1])

X = simulate_dichotomized_gaussian(level_list, n_samples=100)

model = FactorAnalysis(level_list=level_list, n_components=3)
model.fit(X)
model.biplot(X)
plt.show()
```




## References

[1] T. Arai, Multivariate binary probability distribution in the Grassmann formalism, Physical Review E, 103(6), 062104 (2021), [https://doi.org/10.1103/PhysRevE.103.062104](https://doi.org/10.1103/PhysRevE.103.062104), [arXiv:2009.08482v2 [stat.ME]](https://arxiv.org/abs/2009.08482).  

[2] T. Arai, A canonical probability distribution for a mixture of continuous and binary random variables, Physica Scripta, 100(11), 115219 (2025), [https://doi.org/10.1088/1402-4896/adfb07](https://doi.org/10.1088/1402-4896/adfb07), [arXiv:2209.12147v1 [stat.ME]](https://arxiv.org/abs/2209.12147v1).  

[3] T. Arai, Identifiable factor analysis for mixed continuous and binary variables based on the Gaussian-Grassmann distribution, [arXiv:2512.10804 [stat.ME]](https://arxiv.org/abs/2512.10804) (2025).  

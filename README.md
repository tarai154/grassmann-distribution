

# Grassmann Distribution
A Python implementation of the multivariate binary probability distribution based on the Grassmann formalism [1].  



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



## References
[1] T. Arai, Multivariate binary probability distribution in the Grassmann formalism, Physical Review E, 103(6), 062104 (2021).

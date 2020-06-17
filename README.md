## Description

This is a short implementation of the EP algorithm as described in [1] for a toy metabolic network. The code is heavily commented and all formulas are embedded for easy understanding.

**This implementation have only educative proposes, for a numerically efficient implementatio, see the reference.**

## Installation 

```
git clone https://github.com/josePereiro/MetabolicEP_ToyImplementation
```

`MetabolicEP-ToyImplementation.jl` is aim to be read as a jupyter notebook, even when it have not the `.ipynb` extension, see [jupytext](https://github.com/mwouts/jupytext).

I deliver a julia `Project.toml` that can be instantiated, but the dependency are really generic and should work for any `v1.x` julia version.

## References

[1] Braunstein, Alfredo, Anna Paola Muntoni, and Andrea Pagnani. “An Analytic Approximation of the Feasible Space of Metabolic Networks.” Nature Communications 8, no. 1 (April 6, 2017): 1–9. https://doi.org/10.1038/ncomms14915.
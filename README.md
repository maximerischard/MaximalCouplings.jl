# MaximalCouplings

[![Build Status](https://travis-ci.org/maximerischard/MaximalCouplings.jl.svg?branch=master)](https://travis-ci.org/maximerischard/MaximalCouplings.jl)

[![Coverage Status](https://coveralls.io/repos/maximerischard/MaximalCouplings.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/maximerischard/MaximalCouplings.jl?branch=master)

[![codecov.io](http://codecov.io/github/maximerischard/MaximalCouplings.jl/coverage.svg?branch=master)](http://codecov.io/github/maximerischard/MaximalCouplings.jl?branch=master)

Julia package for maximal couplings between two probability distributions. 
The package mainly implements an algorithm for obtaining random draws from a maximal coupling,
as described in Hermann Þórisson's book [1] and in Pierre Jacob's [blog post](https://statisfaction.wordpress.com/2017/09/06/sampling-from-a-maximal-coupling/).

# Example

```julia
using Distributions
using MaximalCouplings

# create the maximal coupling object
c = MaximalCoupling(Normal(1,2), 
                    Gamma(1, 2))

# obtain 100,000 random draws from the coupling
xy_pairs = rand(c, 100000)
println("Fraction of samples where X==Y: ", mean(xy_pairs[1,:] .== xy_pairs[2,:]))
```
```Fraction of samples where X==Y: 0.68141```
```julia
p = prob_couple(c)
println("Probability of coupling: ", p)
```
```Probability of coupling: 0.6812677887974173```


# References

1. Þórisson, H., 2000. *Coupling, stationarity, and regeneration* (Vol. 14). New York: Springer.

module MaximalCouplings

using Distributions
using Roots
using Random
import Random: rand, rand!
import Base: length, eltype
using StatsBase: Weights

include("types.jl")
include("rand.jl")
include("prob_couple.jl")

export MaximalCoupling
export prob_couple

end # module

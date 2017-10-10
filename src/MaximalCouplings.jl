module MaximalCouplings

using Distributions
using Roots
import Base: rand, rand!

include("types.jl")
include("rand.jl")
include("prob_couple.jl")

export MaximalCoupling
export prob_couple

end # module

using MaximalCouplings
using Distributions
using Test
using Random
using Random: rand, rand!

# For two distributions that are the same, the maximal coupling should
# be perfect.
@testset "Equal Distributions" begin
    c_equal = MaximalCoupling(Normal(1,5), Normal(1,5))
    @test prob_couple(c_equal) ≈ 1.0
    @testset "samples" begin
        for _ in 1:1000
            x, y = rand(c_equal)
            @test x == y
        end
    end # testset
end # testset

# Probabilistic test that the probability given by `prob_couple`
# matches the fraction of samples yielded by `rand` that are coupled.
@testset "Probability of Coupling" begin
    coup = MaximalCoupling(Normal(1,2), Gamma(1, 2))
    nsamples = 10^7
    X, Y = rand(coup, nsamples)
    pcouple = prob_couple(coup)
    frac_coupled = mean(X .== Y)
    se = sqrt(pcouple  * (1-pcouple) / nsamples)
    deviation = (pcouple - frac_coupled) / se
    @test abs(deviation) < 5 
    @test eltype(coup) == Float64
    @test eltype(coup) == eltype(X)
    @test eltype(coup) == eltype(Y)
    @test eltype(coup) == eltype(rand(coup)[1])
    @test eltype(coup) == eltype(rand(coup)[2])
end # testset

@testset "Multivariate Couplings" begin
    d1=MultivariateNormal([1., 1.], [[3., 1.]'; [1., 4.]'])
    d2=MvLogNormal([0., 0.], [[1., 0.1]'; [0.1, 1.]'])
    coup = MaximalCoupling(d1, d2)
    # Dimensionality checks
    @test length(coup) == 2
    @test length(rand(coup)[1]) == 2
    @test length(rand(coup)[2]) == 2
    @test size(rand(coup, 5)[1]) == size(rand(d1, 5))
    @test size(rand(coup, 5)[2]) == size(rand(d2, 5))
    @test size(rand(coup, 5)[1]) == (2, 5)
    @test size(rand(coup, 5)[2]) == (2, 5)

    X, Y = rand(coup, 10^4)
    equal13 = X[1, :] .== Y[1, :]
    equal24 = X[2, :] .== Y[2, :]
    @test all(equal13 .== equal24)
    # test that some samples are coupled
    @test sum(equal13) > 0
    @test eltype(coup) == Float64
    @test eltype(coup) == eltype(X)
    @test eltype(coup) == eltype(Y)
    @test eltype(coup) == eltype(rand(coup)[1])
    @test eltype(coup) == eltype(rand(coup)[2])
end

@testset "Discrete Couplings" begin
    d1 = Poisson(4.2)
    d2 = Categorical([0.1, 0.2, 0.3, 0.4])
    coup = MaximalCoupling(d1, d2)
    # Dimensionality checks
    @test length(coup) == 1
    @test length(rand(coup)[1]) == 1
    @test length(rand(coup)[2]) == 1
    @test size(rand(coup, 5)[1]) == size(rand(d1, 5))
    @test size(rand(coup, 5)[2]) == size(rand(d2, 5))
    @test size(rand(coup, 5)[1]) == (5,)
    @test size(rand(coup, 5)[2]) == (5,)
    pcouple = prob_couple(coup)

    nsamples = 10^7
    X, Y = rand(coup, nsamples)
    frac_coupled = mean(X .== Y)
    se = sqrt(pcouple  * (1-pcouple) / nsamples)
    deviation = (pcouple - frac_coupled) / se
    @test abs(deviation) < 5 
    @test eltype(coup) == Int64
    @test eltype(coup) == eltype(X)
    @test eltype(coup) == eltype(Y)
    @test eltype(coup) == eltype(rand(coup)[1])
    @test eltype(coup) == eltype(rand(coup)[2])
end

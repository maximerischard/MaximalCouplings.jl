using MaximalCouplings
using Distributions
using Base.Test

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
    c_normgamma = MaximalCoupling(Normal(1,2), Gamma(1, 2))
    nsamples = 10^7
    xy_pairs = rand(c_normgamma, nsamples);
    pcouple = prob_couple(c_normgamma)
    frac_coupled = mean(xy_pairs[1,:] .== xy_pairs[2,:])
    se = sqrt(pcouple  * (1-pcouple) / nsamples)
    deviation = (pcouple - frac_coupled) / se
    @test abs(deviation) < 5 
end # testset

@testset "Multivariate Couplings" begin
    n=MultivariateNormal([1., 1.], [[3., 1.]'; [1., 4.]'])
    ln=MvLogNormal([0., 0.], [[1., 0.1]'; [0.1, 1.]'])
    c = MaximalCoupling(n, ln)
    # Dimensionality checks
    @test length(c) == 4
    @test length(rand(c)) == 4
    @test size(rand(c, 5)) == (4, 5)
    sample = rand(c, 10^4)
    equal13 = sample[1,:] .== sample[3,:]
    equal24 = sample[2,:] .== sample[4,:]
    @test all(equal13 .== equal24)
    # test that some samples are coupled
    @test sum(equal13) > 0
end


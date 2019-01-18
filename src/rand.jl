function _rand_continuous(p::D1, q::D2) where {D1 <: Distribution, D2 <: Distribution}
    X = rand(p)
    log_pX = logpdf(p, X)
    log_qX = logpdf(q, X)
    if isnan(log_qX)
        # logpdf can return NaN if X is outside of
        # the support of q, in which case we should 
        # change it to -Inf
        log_qX = -Inf
    end
    logU = -randexp()
    local Y
    if logU <= ((log_qX - log_pX) + eps()) # eps gives a bit of margin
        Y = X
    else
        found_uncoupled = false
        for _ in 1:1_000_000 # give up after a while
            Ystar = rand(q)
            log_pYstar = logpdf(p, Ystar)
            if isnan(log_pYstar)
                # logpdf can return NaN if Ystar is outside of
                # the support of p, in which case we should 
                # change it to -Inf
                log_pYstar = -Inf
            end
            log_qYstar = logpdf(q, Ystar)
            if log_qYstar < log_pYstar
                continue
            end
            logUstar = -randexp()
            if logUstar >= ((log_pYstar - log_qYstar) - eps())
                Y = Ystar
                found_uncoupled = true
                break
            end
        end
        if !found_uncoupled
            @warn("failed to find uncoupled sample between ", p, "\n and \n", q)
            # just return a coupled sample after all
            Y = X
        end
    end
    return X, Y
end
function _rand_discrete(p::D1, q::D2) where {D1 <: DiscreteDistribution, D2 <: DiscreteDistribution}
    q_support = support(q)
    # overlap = p_support ∩ q_support
    # prob_min = [min(pdf(p, x), pdf(q, x)) for x in overlap]
    # prob_couple = sum(prob_min)

    X = rand(p)
    log_pX = logpdf(p, X)
    log_qX = logpdf(q, X)
    if isnan(log_qX)
        # logpdf can return NaN if X is outside of
        # the support of q, in which case we should 
        # change it to -Inf
        log_qX = -Inf
    end
    logU = -randexp()
    local Y
    if logU <= ((log_qX - log_pX) + eps()) # eps gives a bit of margin
        Y = X
    else
        q_minus_p = [max(pdf(q, x)-pdf(p,x), 0.0) for x in q_support]
        Y = sample(q_support, Weights(q_minus_p))

    end
    
    return X, Y
end

function rand(coup::MaximalCoupling{D1,D2}) where {D1, D2}
    _rand_continuous(coup.p, coup.q)
end
function rand(coup::MaximalCoupling{D1,D2}) where {D1<:DiscreteDistribution, D2<:DiscreteDistribution}
    p_finite_support = isfinite(minimum(coup.p)) && isfinite(maximum(coup.p))
    q_finite_support = isfinite(minimum(coup.q)) && isfinite(maximum(coup.q))
    if q_finite_support
        return _rand_discrete(coup.p, coup.q)
    elseif p_finite_support
        nx = length(coup.p)
        ny = length(coup.q)
        Y, X = _rand_discrete(coup.q, coup.p)
        return X, Y
    else
        return _rand_continuous(coup.p, coup.q)
    end
end

function rand(coup::MaximalCoupling{D1, D2}, n::Int64) where {D1, D2}
    X = Vector{eltype(coup)}(undef, n)
    Y = Vector{eltype(coup)}(undef, n)
    for i in 1:n
        xi, yi = rand(coup)
        X[i] = xi
        Y[i] = yi
    end
    return X, Y
end
function rand(coup::MaximalCoupling{D1, D2}, n::Int64) where {D1 <: MultivariateDistribution, D2 <: MultivariateDistribution}
    X = Matrix{eltype(coup)}(undef, length(coup), n)
    Y = Matrix{eltype(coup)}(undef, length(coup), n)
    for i in 1:n
        xi, yi = rand(coup)
        X[:, i] = xi
        Y[:, i] = yi
    end
    return X, Y
end

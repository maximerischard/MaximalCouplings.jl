using StatsBase: Weights

function _rand_continuous!(p::D1, q::D2, xy::V) where {D1 <: Distribution, D2 <: Distribution, V<:AbstractVector}
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
            warn("failed to find uncoupled sample between ", p, "\n and \n", q)
            # just return a coupled sample after all
            Y = X
        end
    end
    xy[1:length(p)] = X
    xy[length(p)+1:end] = Y
    return xy
end
function _rand_discrete!(p::D1, q::D2, xy::V) where {D1 <: DiscreteDistribution, D2 <: DiscreteDistribution, V<:AbstractVector}
    q_support = support(q)
    # overlap = p_support ∩ q_support
    # if !issorted(overlap)
        # sort!(overlap)
    # end
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
    
    xy[1:length(p)] = X
    xy[length(p)+1:end] = Y
    return xy
end

function rand!(coup::MaximalCoupling{D1,D2}, xy::AbstractVector) where {D1, D2}
    length(xy) == length(coup) ||
        throw(DimensionMismatch("Output size inconsistent with sample length."))
    _rand_continuous!(coup.p, coup.q, xy)
end
function rand!(coup::MaximalCoupling{D1,D2}, xy::AbstractVector) where {D1<:DiscreteDistribution, D2<:DiscreteDistribution}
    length(xy) == length(coup) ||
        throw(DimensionMismatch("Output size inconsistent with sample length."))
    p_finite_support = isfinite(minimum(coup.p)) && isfinite(maximum(coup.p))
    q_finite_support = isfinite(minimum(coup.q)) && isfinite(maximum(coup.q))
    if q_finite_support
        return _rand_discrete!(coup.p, coup.q, xy)
    elseif p_finite_support
        nx = length(coup.p)
        ny = length(coup.q)
        yx = view(xy, vcat(nx+1:nx+ny, 1:nx))
        return _rand_discrete!(coup.q, coup.p, yx)
    else
        return _rand_continuous!(coup.p, coup.q, xy)
    end
end



function rand(coup::MaximalCoupling, n::Int64)
    xy = Matrix{eltype(coup)}(length(coup), n)
    for i in 1:n
        rand!(coup, view(xy, :, i))
    end
    return xy
end

function rand(coup::MaximalCoupling)
    xy = Vector{eltype(coup)}(length(coup))
    rand!(coup, xy)
    return xy
end

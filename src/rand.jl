function rand!(coup::MaximalCoupling{D1,D2}, xy::AbstractVector) where {D1, D2}
    length(xy) == length(coup) ||
        throw(DimensionMismatch("Output size inconsistent with sample length."))

    X = rand(coup.p)
    log_pX = logpdf(coup.p, X)
    log_qX = logpdf(coup.q, X)
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
            Ystar = rand(coup.q)
            log_pYstar = logpdf(coup.p, Ystar)
            if isnan(log_pYstar)
                # logpdf can return NaN if Ystar is outside of
                # the support of p, in which case we should 
                # change it to -Inf
                log_pYstar = -Inf
            end
            log_qYstar = logpdf(coup.q, Ystar)
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
            warn("failed to find uncoupled sample between ", coup.p, "\n and \n", coup.q)
            # just return a coupled sample after all
            Y = X
        end
    end
    xy[1:length(coup.p)] = X
    xy[length(coup.p)+1:end] = Y
    return xy
end

function rand(coup::MaximalCoupling, n::Int64)
    xy = Matrix{Float64}(length(coup), n)
    for i in 1:n
        rand!(coup, view(xy, :, i))
    end
    return xy
end

function rand(coup::MaximalCoupling)
    xy = Vector{Float64}(length(coup))
    rand!(coup, xy)
    return xy
end

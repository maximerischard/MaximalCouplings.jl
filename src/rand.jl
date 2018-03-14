function rand!(coup::MaximalCoupling, xy::AbstractVector)
    length(xy) == length(coup) ||
        throw(DimensionMismatch("Output size inconsistent with sample length."))

    X = rand(coup.p)
    log_pX = logpdf(coup.p, X)
    log_qX = logpdf(coup.q, X)
    logU = log_pX - randexp()
    local Y
    if logU - log_qX <= eps()
        Y = X
    else
        found_uncoupled = false
        for _ in 1:1_000_000 # give up after a while
            Ystar = rand(coup.q)
            log_pYstar = logpdf(coup.p, Ystar)
            log_qYstar = logpdf(coup.q, Ystar)
            if log_qYstar < log_pYstar
                continue
            end
            logUstar = log_qYstar - randexp()
            if logUstar - log_pYstar >= -eps()
                Y = Ystar
                found_uncoupled = true
                break
            end
        end
        if !found_uncoupled
            println("failed to find uncoupled sample between ", coup.p, "\n and \n", coup.q)
        end
        if !found_uncoupled
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

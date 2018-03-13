function rand!(coup::MaximalCoupling, xy::AbstractVector)
    length(xy) == length(coup) ||
        throw(DimensionMismatch("Output size inconsistent with sample length."))

    X = rand(coup.p)
    pX = pdf(coup.p, X)
    qX = pdf(coup.q, X)
    local Y
    if (rand()*pX) < qX
        Y = X
    else
        while true
            Ystar = rand(coup.q)
            Ustar = rand() * pdf(coup.q, Ystar)
            if Ustar > pdf(coup.p, Ystar)
                Y = Ystar
                break
            end
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

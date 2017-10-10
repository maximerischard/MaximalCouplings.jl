function rand!(coup::MaximalCoupling, xy::AbstractVector)
    X = rand(coup.p)
    pX = pdf(coup.p, X)
    qX = pdf(coup.q, X)
    𝒰 = Uniform(0.0, pX)
    local Y
    if rand(𝒰) < qX
        Y = X
    else
        while true
            Ystar = rand(coup.q)
            𝒰star = Uniform(0.0, pdf(coup.q, Ystar))
            Ustar = rand(𝒰star)
            if Ustar > pdf(coup.p, Ystar)
                Y = Ystar
                break
            end
        end
    end
    xy[1] = X
    xy[2] = Y
    return xy
end

function rand(coup::MaximalCoupling, n::Int64)
    xy = Matrix{Float64}(2, n)
    for i in 1:n
        rand!(coup, view(xy, :, i))
    end
    return xy
end

function rand(coup::MaximalCoupling)
    xy = Vector{Float64}(2)
    rand!(coup, xy)
    return xy
end

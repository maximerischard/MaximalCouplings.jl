""" Given two sorted vectors x and y of length nx and ny, 
    return a sorted vector of length nx+ny containing all elements 
    from x and y.
    Equivalent to sort([x; y]) but hopefully a little bit more efficent.
"""
function merge_sorted(x, y)  
    nx = length(x)
    ny = length(y)
    merged = Vector{eltype(x)}(nx+ny)
    ix=1
    iy=1
    for ixy in 1:(nx+ny)
        if ix > nx
            # we've run out of x, just fill the rest of the vector with y
            merged[ixy] = y[iy]
            iy +=1
        elseif iy > ny
            # we've run out of y, just fill the rest of the vector with x
            merged[ixy] = x[ix]
            ix +=1
        elseif x[ix] < y[iy]
            merged[ixy] = x[ix]
            ix +=1
        else
            merged[ixy] = y[iy]
            iy +=1
        end
    end
    return merged
end

""" Find all the points where the PDFs of the two distributions in a coupling
    cross. Also return which PDF dominates in each segment.
"""
function get_crossings{U1<:ContinuousUnivariateDistribution, U2<:ContinuousUnivariateDistribution}(coup::MaximalCoupling{U1,U2}, nquantiles::Int)
    # Start by comparing the PDFs on a grid of points given by the
    # merged quantiles of the two distributions.
    pquant = quantile.(coup.p, linspace(0,1,nquantiles)[2:end-1])
    qquant = quantile.(coup.q, linspace(0,1,nquantiles)[2:end-1])
    pq_quantiles = merge_sorted(pquant, qquant)
    
    # function that returns the difference of the two PDFs
    pq_diff(x) = pdf(coup.p, x) - pdf(coup.q, x)
    
    # for each point, which marginal distribution
    # has the highest PDF?
    p_gt_q = pq_diff.(pq_quantiles) .> 0
    # find the indices of the crossing points of the two PDFs
    crossed_int = diff(p_gt_q)
    crossed_indx = findin(crossed_int, [-1,1])
    
    # For each crossing point, find the exact x at which the PDFs
    # cross, and also return which PDF is greater.
    ncrosses = length(crossed_indx)
    pgreater = Vector{Bool}(ncrosses+1)
    pgreater[1] = pq_diff(minimum(pq_quantiles)) > 0
    crossed_x = Vector{Float64}(ncrosses)
    for icross in 1:ncrosses
        xbefore = pq_quantiles[crossed_indx[icross]]
        xafter = pq_quantiles[crossed_indx[icross]+1]
        crossed_x[icross] = fzero(pq_diff, xbefore, xafter)
        pgreater[icross+1] = pq_diff(xafter) > 0
    end
    return crossed_x, pgreater
end

""" Probability that X=Y (coupling event) in a maximal coupling
    of two univariate distributions.
"""
function prob_couple(coup::MaximalCoupling{U1,U2}; nquantiles::Int=100) where {U1<:ContinuousUnivariateDistribution, U2<:ContinuousUnivariateDistribution}
    crossed_x, pgreater = get_crossings(coup, nquantiles)
    xbefore = -Inf # could use minimum support of p and q instead
    pcouple = 0.0
    for (ichange, xcross) in enumerate(crossed_x)
        if pgreater[ichange]
            pcouple += cdf(coup.q, xcross) - cdf(coup.q, xbefore)
        else
            pcouple += cdf(coup.p, xcross) - cdf(coup.p, xbefore)
        end
        xbefore = xcross
    end
    if pgreater[end]
        pcouple += cdf(coup.q, Inf) - cdf(coup.q, xbefore)
    else
        pcouple += cdf(coup.p, Inf) - cdf(coup.p, xbefore)
    end
    @assert 0.0 <= pcouple <= 1.0
    return pcouple
end
function prob_couple(coup::MaximalCoupling{U1,U2}; nquantiles::Int=100) where {U1<:DiscreteDistribution, U2<:DiscreteDistribution}
    p_finite_support = isfinite(minimum(coup.p)) && isfinite(maximum(coup.p))
    q_finite_support = isfinite(minimum(coup.q)) && isfinite(maximum(coup.q))
    local overlap
    if p_finite_support && q_finite_support
        p_support = support(coup.p)
        q_support = support(coup.q)
        overlap = p_support ∩ q_support
    elseif p_finite_support
        overlap = support(coup.p)
    elseif q_finite_support
        overlap = support(coup.q)
    else
        throw(DomainError()) # ToDo: implement
    end
    prob_min = [min(pdf(coup.p, x), pdf(q, x)) for x in overlap]
    prob_couple = sum(prob_min)
    return pcouple
end

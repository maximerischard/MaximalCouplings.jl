""" Given two sorted vectors x and y of length nx and ny, 
    return a sorted vector of length nx+ny containing all elements 
    from x and y.
    Equivalent to sort([x; y]) but hopefully a little bit more efficent.
"""
function merge_sorted(x, y)  
    nx = length(x)
    ny = length(y)
    merged = Vector{eltype(x)}(undef, nx+ny)
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
function get_crossings(coup::MaximalCoupling, nquantiles::Int)
    # Start by comparing the PDFs on a grid of points given by the
    # merged quantiles of the two distributions.
    pquant = quantile.(coup.p, range(0, stop=1, length=nquantiles)[2:end-1])
    qquant = quantile.(coup.q, range(0, stop=1, length=nquantiles)[2:end-1])
    pq_quantiles = merge_sorted(pquant, qquant)
    
    # function that returns the difference of the two PDFs
    pq_diff(x) = pdf(coup.p, x) - pdf(coup.q, x)
    
    # p(x) greater than q(x)
    p_gt_q = pq_diff(pq_quantiles[1]) >= 0
    xbefore = -Inf
    crossed_x = Float64[]
    pgreater = Bool[p_gt_q]
    for (idx, x) in enumerate(pq_quantiles)
        x = pq_quantiles[idx]
        if (
            (p_gt_q) # currently expect p(x) > q(x)
            && pq_diff(x) < 0 # but see p(x) < q(x)
           ) || ( # or
            !(p_gt_q)
            && pq_diff(x) > 0
           )
            # p&q have crossed!
            # push!(crossed_indx, idx)
            p_gt_q = !p_gt_q # switch record of dominant PDF

            push!(crossed_x, fzero(pq_diff, xbefore, x))
            push!(pgreater, p_gt_q)
        end
        xbefore = x
    end
    
    return crossed_x, pgreater
end

""" Probability that X=Y (coupling event) in a maximal coupling
    of two univariate distributions.
"""
function prob_couple(coup::MaximalCoupling; nquantiles::Int=100)
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
        approx_min = min(quantile(coup.p, 1e-10), quantile(coup.q, 1e-10))
        approx_max = max(quantile(coup.p, 1-1e-10), quantile(coup.q, 1-1e-10))
        overlap = approx_min:approx_max
        @assert length(overlap) < 10^5 # otherwise it'll take too long
        # if this is an issue, we should consider using an adaptation
        # of the crossings-detection method for continuous distributions
    end
    # prob_min = [min(pdf(coup.p, x), pdf(coup.q, x)) for x in overlap]
    # pcouple = sum(prob_min)
    pcouple = sum(x -> min(pdf(coup.p, x), pdf(coup.q, x)), overlap)
    return pcouple
end

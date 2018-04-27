import Base: length, eltype

type MaximalCoupling{U1<:Distribution, U2<:Distribution}
    p::U1
    q::U2
    function MaximalCoupling{U1, U2}(p::U1, q::U2) where {U1, U2}
        length(p) == length(q) || 
            throw(DimensionMismatch("Coupled distributions of different dimension."))
        new(p, q)
    end
end

MaximalCoupling{U1<:Distribution, U2<:Distribution}(p::U1, q::U2) = MaximalCoupling{U1, U2}(p, q)

length(coup::MaximalCoupling) = length(coup.p) + length(coup.q)
eltype(coup::MaximalCoupling) = Union{eltype(coup.p), eltype(coup.q)}

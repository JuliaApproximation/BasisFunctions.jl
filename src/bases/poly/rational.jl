# rational.jl

struct RationalBasis{T} <: Dictionary1d{T,T}
    roots   :: AbstractArray{T}
    support :: Domain

    RationalBasis{T}(roots) where {T} = new(roots)
end

RationalBasis(roots::AbstractArray{T}, d::Domain) where {T} = RationalBasis{T}(roots,d)

function unsafe_eval_element(b::RationalBasis, idx::Int, x)
    degree = 1
    for i = 1:(idx-1)
        if b.roots[i] == b.roots[idx]
            degree += 1
        end
    end
    return 1 ./ (x-b.roots[idx]).^(degree)
end

length(r::RationalBasis) = length(r.roots)
is_basis(r::RationalBasis) = false

support(r::RationalBasis) = r.support

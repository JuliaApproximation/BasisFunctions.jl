immutable RationalBasis{T} <: FunctionSet1d{T}
    roots :: AbstractArray{T}

    RationalBasis(roots) = new(roots)
end
RationalBasis{T}(roots::AbstractArray{T}) = RationalBasis{T}(roots)

function eval_element(b::RationalBasis, idx::Int, x)
    degree=1
    for i = 1:(idx-1)
        if b.roots[i]==b.roots[idx]
            degree+=1
        end
    end
    return 1./(x-b.roots[idx]).^(degree)
end

length(r::RationalBasis) = length(r.roots)
is_basis(r::RationalBasis) = false

left(r::RationalBasis, idx) = minimum(abs(r.roots))-1
right(r::RationalBasis, idx) = maximum(abs(r.roots))+1

left(r::RationalBasis) = minimum(abs(r.roots))-1
right(r::RationalBasis) = maximum(abs(r.roots))+1

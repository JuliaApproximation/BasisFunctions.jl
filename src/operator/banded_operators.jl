
"""
A banded operator of which every row contains equal elements.

The top row starts at index offset, the second row at step+offset.
"""
struct HorizontalBandedOperator{ELT} <: DictionaryOperator{ELT}
    src     ::  Dictionary
    dest    ::  Dictionary
    array   ::  Vector{ELT}
    step    ::  Int
    offset  ::  Int
    function HorizontalBandedOperator{ELT}(src::Dictionary, dest::Dictionary, array::Vector{ELT}, step::Int=1, offset::Int=0) where ELT
        @assert length(array) <= length(src)
        @assert step <= length(src) # apply! only works if step is smaller then L
        new{ELT}(src, dest, array, step, offset)
    end
end

HorizontalBandedOperator(src::Dictionary, dest::Dictionary, array::Vector{ELT}, step::Int=1, offset::Int=0) where ELT =
    HorizontalBandedOperator{ELT}(src, dest, array, step, offset)

function apply!(op::HorizontalBandedOperator, dest::Vector, src::Vector)
    dest[:] .= 0
    L = length(src)
    aL = length(op.array)
    dL = length(dest)
    @inbounds for a_i in 1:aL
        ind = mod(a_i+op.offset-1,L)+1
        for d_i in 1:dL
            dest[d_i] += op.array[a_i]*src[ind]
            ind += op.step
            if ind > L
                ind -= L
            end
        end
    end
    dest
end


struct VerticalBandedOperator{ELT} <: DictionaryOperator{ELT}
    src     ::  Dictionary
    dest    ::  Dictionary
    array   ::  Vector{ELT}
    step    ::  Int
    offset  ::  Int
    function VerticalBandedOperator{ELT}(src::Dictionary, dest::Dictionary, array::Vector{ELT}, step::Int, offset::Int) where ELT
        @assert length(array) <= length(dest)
        @assert step <= length(dest) # apply! only works if step is smaller then L
        new{ELT}(src, dest, array, step, offset)
    end
end

VerticalBandedOperator(src::Dictionary, dest::Dictionary, array::Vector{ELT}, step::Int=1, offset::Int=0) where ELT =
    VerticalBandedOperator{ELT}(src, dest, array, step, offset)

function apply!(op::VerticalBandedOperator, dest::Vector, src::Vector)
    dest[:] .= 0
    # assumes step is smaller then L
    L = length(dest)
    aL = length(op.array)
    sL = length(src)
    @inbounds for a_i in 1:aL
        ind = mod(a_i+op.offset-1,L)+1
        for s_i in 1:sL
            dest[ind] += op.array[a_i]*src[s_i]
            ind += op.step
            if ind > L
                ind -= L
            end
        end
    end

    dest
end

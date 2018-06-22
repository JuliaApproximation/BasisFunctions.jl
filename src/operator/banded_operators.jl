# banded_oprators.jl

"""
A banded operator of which every row contains equal elements.

The top row starts at index offset, the second row at step+offset.
"""
struct HorizontalBandedOperator{ELT} <: BasisFunctions.DictionaryOperator{ELT}
    src::Dictionary
    dest::Dictionary
    array::Vector{ELT}
    step::Int
    offset::Int
    function HorizontalBandedOperator{ELT}(src::Dictionary, dest::Dictionary, array::Vector{ELT}, step::Int=1, offset::Int=0) where ELT
        @assert length(array) <= length(src)
        @assert step <= length(src) # apply! only works if step is smaller then L
        new{ELT}(src, dest, array, step, offset)
    end
end

HorizontalBandedOperator(src::Dictionary, dest::Dictionary, array::Vector{ELT}, step::Int=1, offset::Int=0) where ELT =
    HorizontalBandedOperator{ELT}(src, dest, array, step, offset)

function BasisFunctions.apply!(op::HorizontalBandedOperator, dest::Vector, src::Vector)
    dest[:]=0
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

#
# struct VerticalBandedOperator{ELT} <: BasisFunctions.DictionaryOperator{ELT}
#     src::Dictionary
#     dest::Dictionary
#     array::Vector{ELT}
#     step::Int
#     offset::Int
#     function VerticalBandedOperator{ELT}(src::Dictionary, dest::Dictionary, array::Vector{ELT}, step::Int, offset::Int) where ELT
#         @assert length(array) <= length(dest)
#         new{ELT}(src, dest, array, step, offset)
#     end
# end
#
# VerticalBandedOperator(src::Dictionary, dest::Dictionary, array::Vector{ELT}, step::Int=1, offset::Int=0) where ELT =
#     VerticalBandedOperator{ELT}(src, dest, array, step, offset)
#
# function BasisFunctions.apply!(op::VerticalBandedOperator, dest::Vector, src::Vector)
#     dest[:]=0
#
#     # Crude, but works
#     # for a_i in 1:length(op.array)
#     #     for s_i in 1:length(src)
#     #         dest[mod(a_i+op.step*(s_i-1)+op.offset-1,length(dest))+1] += op.array[a_i]*src[s_i]
#     #     end
#     # end
#
#     # assumes step is smaller then L
#     L = length(dest)
#     for a_i in 1:length(op.array)
#         ind = mod(a_i+op.offset-1,L)+1
#         for s_i in 1:length(src)
#             dest[ind] += op.array[a_i]*src[s_i]
#             ind += op.step
#             if ind > L
#                 ind -= L
#             end
#         end
#     end
#
#     dest
# end

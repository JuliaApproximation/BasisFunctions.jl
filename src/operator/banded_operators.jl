
struct VerticalBandedOperator{ELT} <: BasisFunctions.AbstractOperator{ELT}
    src::Span
    dest::Span
    array::Vector{ELT}
    step::Int
    offset::Int
    function VerticalBandedOperator{ELT}(src::Span, dest::Span, array::Vector{ELT}, step::Int, offset::Int) where ELT
        @assert length(array) <= length(dest)
        new{ELT}(src, dest, array, step, offset)
    end
end

VerticalBandedOperator(src::Span, dest::Span, array::Vector{ELT}, step::Int=1, offset::Int=0) where ELT =
    VerticalBandedOperator{ELT}(src, dest, array, step, offset)

function BasisFunctions.apply!(op::VerticalBandedOperator, dest::Vector{ELT}, src::Vector{ELT}) where {ELT}
    dest[:]=0

    # Crude, but works
    # for a_i in 1:length(op.array)
    #     for s_i in 1:length(src)
    #         dest[mod(a_i+op.step*(s_i-1)+op.offset-1,length(dest))+1] += op.array[a_i]*src[s_i]
    #     end
    # end

    # assumes step is smaller then L
    L = length(dest)
    for a_i in 1:length(op.array)
        ind = mod(a_i+op.offset-1,L)+1
        for s_i in 1:length(src)
            dest[ind] += op.array[a_i]*src[s_i]
            ind += op.step
            if ind > L
                ind -= L
            end
        end
    end

    dest
end
struct HorizontalBandedOperator{ELT} <: BasisFunctions.AbstractOperator{ELT}
    src::Span
    dest::Span
    array::Vector{ELT}
    step::Int
    offset::Int
    function HorizontalBandedOperator{ELT}(src::Span, dest::Span, array::Vector{ELT}, step::Int=1, offset::Int=0) where ELT
        @assert length(array) <= length(src)
        new{ELT}(src, dest, array, step, offset)
    end
end


HorizontalBandedOperator(src::Span, dest::Span, array::Vector{ELT}, step::Int=1, offset::Int=0) where ELT =
    HorizontalBandedOperator{ELT}(src, dest, array, step, offset)

function BasisFunctions.apply!(op::HorizontalBandedOperator, dest::Vector{ELT}, src::Vector{ELT}) where {ELT}
    dest[:]=0

    # Crude, but works
    # for a_i in 1:length(op.array)
    #     for s_i in 1:length(src)
    #         dest[mod(a_i+op.step*(s_i-1)+op.offset-1,length(dest))+1] += op.array[a_i]*src[s_i]
    #     end
    # end

    # assumes step is smaller then L
    L = length(src)
    for a_i in 1:length(op.array)
        ind = mod(a_i+op.offset-1,L)+1
        for d_i in 1:length(dest)
            dest[d_i] += op.array[a_i]*src[ind]
            ind += op.step
            if ind > L
                ind -= L
            end
        end
    end

    dest
end

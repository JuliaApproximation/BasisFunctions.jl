
"""
A `ContextOperator` combines an `AbstractMatrix` with a source and destination
dictionary.
"""

abstract type ContextOperator{T} <: DictionaryOperator{T} end

verify_size(src, dest, A) = (size(A,2) == length(src)) && (size(A,1)== length(dest))

struct DiagonalContextOperator{T} <: ContextOperator{T}
    src     ::  Dictionary
    dest    ::  Dictionary
    A       ::  DiagonalMatrix{T}

    function DiagonalContextOperator{T}(src, dest, A)
        @assert verify_size(src, dest, A)
        new(src, dest, A)
    end
end

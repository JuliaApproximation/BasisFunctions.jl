# LazyArrays._copyto!(::LazyArrays.DiagonalLayout, dest::Diagonal, M::LazyArrays.MatMulMat{<:LazyArrays.DiagonalLayout}) =
#     copyto!(dest, M.factors[1]*M.factors[2])
# Base.similar(M::LazyArrays.MatMulMat{<:LazyArrays.DiagonalLayout}) = Diagonal(Vector{eltype(M)}(undef,size(M,1)))
# LazyArrays._materialize(M:: LazyArrays.ArrayMuls, ::Tuple{Base.OneTo}) = LazyArrays.rmaterialize(M)
# LazyArrays._materialize(M:: LazyArrays.Mul, ::Tuple{Base.OneTo}) = LazyArrays.rmaterialize(M)
# LazyArrays._materialize(M:: LazyArrays.ArrayMulArray, ::Tuple{Base.OneTo}) = copyto!(similar(M), M)

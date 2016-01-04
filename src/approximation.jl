# approximation.jl



sample{N,T}(g::AbstractGrid{N,T}, f::Function, ELT = T) = ELT[f(x...) for x in g]

(*)(op::AbstractOperator, f::Function) = op * sample(grid(src(op)), f, eltype(op))

approximate(s::FunctionSet, f::Function) = SetExpansion(s, approximation_operator(s) * f)


function interpolate{N}(s::FunctionSet{N}, xs::AbstractVector{AbstractVector}, f)
    A = interpolation_matrix(s, xs)
    B = [f(x...) for x in xs]
    SetExpansion(s, A\B)
end




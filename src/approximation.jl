# approximation.jl


rhs{N,T}(g::AbstractGrid{N,T}, f::Function) = T[f(x...) for x in g]


function approximate(s::FunctionSet, f::Function)
    A = approximation_operator(s)
    B = rhs(grid(s), f)
    SetExpansion(s, A*B)
end


function interpolate{N}(s::FunctionSet{N}, xs::AbstractVector{AbstractVector}, f)
    A = interpolation_matrix(s, xs)
    B = [f(x...) for x in xs]
    SetExpansion(s, A\B)
end

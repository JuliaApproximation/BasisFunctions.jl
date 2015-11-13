# approximation.jl


rhs{N,T}(g::AbstractGrid{N,T}, f::Function) = T[f(x...) for x in g]


function approximate(s::FunctionSet, f::Function)
    A = approximation_operator(s)
    B = rhs(grid(s), f)
    SetExpansion(s, A*B)
end

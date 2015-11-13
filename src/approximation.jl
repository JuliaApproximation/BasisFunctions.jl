# approximation.jl


rhs(g::AbstractGrid, f::Function) = [f(x...) for x in g]


function approximate(s::FunctionSet, f::Function)
    A = approximation_operator(s)
    B = rhs(grid(s), f)
    SetExpansion(s, A*B)
end

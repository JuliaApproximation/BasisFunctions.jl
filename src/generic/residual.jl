# residual.jl

residual_operator(b; options...) = inv(approximation_operator(b; options...))
residual(F::SetExpansion, f::Function; options...) = residual(f, F; options...)
residual(f::Function, F::SetExpansion; options...) = residual(f, set(F), coefficients(F); options...)

function residual(f::Function, set::FunctionSet, c; options...)
    R = residual_operator(set; options...)
    norm(R*c-project(dest(R), f; options...))
end


export 𝒟

abstract type SyntaxSymbol end

struct DerivativeSymbol <: SyntaxSymbol
end

(s::DerivativeSymbol)(Φ::Dictionary) = differentiation(Φ)

struct FixedDerivative <: SyntaxSymbol
    order

    FixedDerivative(order::Int) = new(order)
    FixedDerivative(order::NTuple{N,Int}) where {N} = new(order)
    function FixedDerivative(order)
        @warn "Expecting differentiation order to be an Int or a tuple of Int's."
        new(order)
    end
end

(s::FixedDerivative)(Φ::Dictionary) = differentiation(Φ, s.order)

getindex(s::DerivativeSymbol, order) = FixedDerivative(order)
getindex(s::DerivativeSymbol, I::Int...) = FixedDerivative(I)
^(s::DerivativeSymbol, order) = FixedDerivative(order)

^(s::FixedDerivative, order::Int) = FixedDerivative(order .* s.order)

"The differentiation operator"
const 𝒟 = DerivativeSymbol()

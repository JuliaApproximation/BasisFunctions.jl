
isorthogonal(::ChebyshevU, ::ChebyshevUWeight) = true
isorthogonal(::ChebyshevT, ::ChebyshevTWeight) = true

isorthogonal(dict::ChebyshevU, measure::GaussChebyshevU) = opsorthogonal(dict, measure)
isorthogonal(dict::ChebyshevT, measure::GaussChebyshevT) = opsorthogonal(dict, measure)

const UniformDiscreteChebyshevTWeight{T,G,W} = GenericGridWeight{T,G,W} where G <: ChebyshevNodes where W <:FillArrays.AbstractFill
isorthogonal(dict::ChebyshevT, measure::UniformDiscreteChebyshevTWeight) = BasisFunctions.opsorthogonal(dict, measure)

gauss_rule(dict::ChebyshevT{T}) where T = GaussChebyshevT{T}(length(dict))
gauss_rule(dict::ChebyshevU{T}) where T = GaussChebyshevU{T}(length(dict))

function gram(::Type{T}, dict::ChebyshevT, ::ChebyshevTWeight; options...) where {T}
	diag = zeros(T, length(dict))
	fill!(diag, convert(T, pi)/2)
	diag[1] = convert(T,pi)
	DiagonalOperator(dict, diag)
end

gram(::Type{T}, dict::ChebyshevU, ::ChebyshevUWeight; options...) where {T} =
	ScalingOperator(convert(T, pi)/2, dict)

function diagonal_gram(::Type{T}, dict::ChebyshevT, measure::GaussChebyshevT; options...) where {T}
    @assert isorthogonal(dict, measure)
    if length(dict) == length(points(measure))
        CoefficientScalingOperator{T}(dict, 1, convert(T,2))*ScalingOperator{T}(dict, convert(T,pi)/2)
    else
        default_diagonal_gram(T, dict, measure; options...)
    end
end


diagonal_gram(::Type{T}, dict::ChebyshevT, measure::UniformDiscreteChebyshevTWeight; options...) where {T} =
    CoefficientScalingOperator{T}(dict, 1, convert(T,2))*ScalingOperator(dict, convert(T,length(points(measure)))/2)

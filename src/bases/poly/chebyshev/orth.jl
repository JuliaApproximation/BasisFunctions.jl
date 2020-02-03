
isorthogonal(::ChebyshevU, ::ChebyshevUMeasure) = true
isorthogonal(::ChebyshevT, ::ChebyshevTMeasure) = true

isorthogonal(dict::ChebyshevU, measure::ChebyshevUGaussMeasure) = opsorthogonal(dict, measure)
isorthogonal(dict::ChebyshevT, measure::ChebyshevTGaussMeasure) = opsorthogonal(dict, measure)

const UniformDiscreteChebyshevTMeasure{T,G,W} = GenericDiscreteMeasure{T,G,W} where G <: ChebyshevNodes where W <:FillArrays.AbstractFill
isorthogonal(dict::ChebyshevT, measure::BasisFunctions.UniformDiscreteChebyshevTMeasure) = BasisFunctions.opsorthogonal(dict, measure)

function gram(::Type{T}, dict::ChebyshevT, ::ChebyshevMeasure; options...) where {T}
	diag = zeros(T, length(dict))
	fill!(diag, convert(T, pi)/2)
	diag[1] = convert(T,pi)
	DiagonalOperator(dict, diag)
end

gram(::Type{T}, dict::ChebyshevU, ::ChebyshevUMeasure; options...) where {T} =
	ScalingOperator(convert(T, pi)/2, dict)

function diagonal_gram(dict::ChebyshevT, measure::ChebyshevTGaussMeasure; options...) where {T}
    @assert isorthogonal(dict, measure)
    if length(dict) == length(grid(measure))
        CoefficientScalingOperator{T}(dict, 1, convert(T,2))*ScalingOperator{T}(dict, convert(T,pi)/2)
    else
        default_diagonal_gram(T, dict, measure; options...)
    end
end


diagonal_gram(::Type{T}, dict::ChebyshevT, measure::UniformDiscreteChebyshevTMeasure; options...) where {T} =
    CoefficientScalingOperator{T}(dict, 1, convert(T,2))*ScalingOperator(dict, convert(T,length(grid(measure)))/2)


isorthogonal(::ChebyshevU, ::ChebyshevUMeasure) = true
isorthogonal(::ChebyshevT, ::ChebyshevTMeasure) = true

isorthogonal(dict::ChebyshevU, measure::ChebyshevUGaussMeasure) = opsorthogonal(dict, measure)
isorthogonal(dict::ChebyshevT, measure::ChebyshevTGaussMeasure) = opsorthogonal(dict, measure)

const UniformDiscreteChebyshevTMeasure{T,G,W} = GenericDiscreteMeasure{T,G,W} where G <: ChebyshevNodes where W <:FillArrays.AbstractFill
isorthogonal(dict::ChebyshevT, measure::BasisFunctions.UniformDiscreteChebyshevTMeasure) = BasisFunctions.opsorthogonal(dict, measure)

function gramoperator(dict::ChebyshevT, ::ChebyshevMeasure; T = coefficienttype(dict), options...)
	diag = zeros(T, length(dict))
	fill!(diag, convert(T, pi)/2)
	diag[1] = convert(T,pi)
	DiagonalOperator{T}(dict, diag)
end

gramoperator(dict::ChebyshevU, ::ChebyshevUMeasure; T = coefficienttype(dict), options...) =
	ScalingOperator{T}(dict, convert(T, pi)/2)

function diagonal_gramoperator(dict::ChebyshevT, measure::ChebyshevTGaussMeasure{S}; T=promote_type(S,coefficienttype(dict)), options...) where S
    @assert isorthogonal(dict, measure)
    if length(dict) == length(grid(measure))
        CoefficientScalingOperator{T}(dict, 1, convert(T,2))*ScalingOperator{T}(dict, convert(T,pi)/2)
    else
        default_diagonal_gramoperator(dict, measure; T=T, options...)
    end
end


diagonal_gramoperator(dict::ChebyshevT, measure::UniformDiscreteChebyshevTMeasure; T=promote_type(subdomaintype(measure),coefficienttype(dict)), options...) =
    CoefficientScalingOperator{T}(dict, 1, convert(T,2))*ScalingOperator(dict, convert(T,length(grid(measure)))/2)

# gram.jl

#################
## Gram operators
#################
"""
The gram operator A of the given basisfunction, i.e., A_ij = <ϕ_i,ϕ_j>, if ϕ_i is the ith basisfunction
"""
Gram(s::Span; options...) = Gram(s, Val{is_orthonormal(set(s))}; options...)

Gram(s::Span, ::Type{Val{true}}; options...) = IdentityOperator(s, s)

function Gram(s::Span, ::Type{Val{false}}; options...)
    if is_orthogonal(set(s))
        d = zeros(s)
        gramdiagonal!(d, s; options...)
        DiagonalOperator(s, s, d)
    else
        A = zeros(coeftype(s), length(s), length(s))
        grammatrix!(A, s; options...)
        MatrixOperator(s, s, A)
    end
end

"""
The dual gram operator A of the given basisfunction, i.e., A_ij = <ϕ_i,ϕ_j>, if ϕ_i is the ith dual basisfunction
"""
DualGram(s::Span; options...) = DualGram(s, Val{is_biorthogonal(set(s))}; options...)

DualGram(s::Span, ::Type{Val{true}}; options...) = inv(Gram(s; options...))

"""
The mixed gram operator A of the given basisfunction, i.e., A_ij = <ϕ_i,ψ_j>, if ϕ_i is the ith dual basisfunction and ψ_j the jth basisfunction
"""
MixedGram(s::Span; options...) = MixedGram(s, Val{is_biorthogonal(set(s))}; options...)

MixedGram(s::Span, ::Type{Val{true}}; options...) = IdentityOperator(s, s)

function grammatrix!(result, s::Span; options...)
    for i in 1:size(result,1)
        for j in i:size(result,2)
            I = dot(s, i, j; options...)
            result[i,j] = I
            if i!= j
                result[j,i] = conj(I)
            end
        end
    end
    result
end

function gramdiagonal!(result, s::Span; options...)
    for i in 1:size(result,1)
        result[i] = dot(s, i, i; options...)
    end
    result
end

################################################################################################
## Take inner products between function and basisfunctions. Used in continous approximation case.
################################################################################################
"""
Project the function on the function space spanned by the functionset by taking innerproducts with the elements of the set.
"""
project(s, f::Function; options...) = project!(zeros(s), s, f; options...)

function project!(result, s, f::Function; options...)
    for i in eachindex(result)
        result[i] = dot(s, i, f; options...)
    end
    result
end

function dot(f::Function, nodes::Array{T,1}; atol=0, rtol=sqrt(eps(T)), verbose=false, options...) where {T}
    (I,e) = QuadGK.quadgk(x->f(x), nodes...; rtol=rtol, atol=atol)
    (e > sqrt(rtol) && verbose) && (warn("Dot product did not converge"))
    I
end

native_nodes(set::FunctionSet1d) = [left(set), right(set)]

dot(s::Span1d, f1::Function, f2::Function, nodes::Array=native_nodes(set(s)); options...)  =
    dot(x->conj(f1(x))*f2(x), nodes; options...)

dot(s::Span, f1::Int, f2::Function, nodes::Array=native_nodes(set(s)); options...) =
    dot(s, x->eval_element(set(s), f1, x), f2, nodes; options...)

dot(s::Span, f1::Int, f2::Int, nodes::Array=native_nodes(set(s)); options...) =
    dot(s, x->eval_element(set(s), f1, x), x->eval_element(set(s), f2, x), nodes; options...)

##########################
## Discrete Gram operators
##########################

oversampled_grid(b::FunctionSet, oversampling::Real) = grid(resize(b, length_oversampled_grid(b, oversampling)))

length_oversampled_grid(b::FunctionSet, oversampling::Real) = approx_length(b, basis_oversampling(b, oversampling)*length(b))

basis_oversampling(set::FunctionSet, sampling_factor::Real) =  sampling_factor

default_oversampling(b::FunctionSet) = 1
# E'E/N
DiscreteGram(s::Span; oversampling = default_oversampling(set(s))) =
  rangetype(s)(1)/discrete_gram_scaling(set(s), oversampling)*UnNormalizedGram(s, oversampling)

function UnNormalizedGram(s::Span, oversampling = 1)
    grid = oversampled_grid(set(s), oversampling)
    evaluation_operator(s, grid)'*evaluation_operator(s, grid)
end

# discrete_gram_scaling{N,T}(b::FunctionSet{N,T}, oversampling) = length_oversampled_grid(b, oversampling)
discrete_gram_scaling(b::FunctionSet, oversampling) = length(b)

# Ẽ'Ẽ/N and since Ẽ = NE^{-1}'
DiscreteDualGram(s::Span; oversampling = default_oversampling(set(s))) =
    inv(DiscreteGram(s; oversampling=oversampling))

# Ẽ'E/N
DiscreteMixedGram(s::Span; oversampling=default_oversampling(set(s))) = IdentityOperator(s,s)



#################
## Gram operators extended
#################

dual(span::Span; options...) = dual(span, Val{is_orthonormal(set(span))}; options...)
dual(span::Span, ::Type{Val{true}}; options...) = span
dual(set::FunctionSet; options...) = error("Dual of $(set) is not known.")
dual(span::Span, ::Type{Val{false}}; options...) = Span(dual(set(span); options...))

"""
The gram operator A of the given basisfunction, i.e., A_ij = <ϕ_i,ϕ_j>, if ϕ_i is the ith basisfunction
"""
function Gram(src::Span, dest::Span; options...)
    A = zeros(rangetype(src, dest),length(dest),length(src))*NaN
    grammatrix!(A, src, dest; options...)
    MatrixOperator(src, dest, A)
end

DualGram(span1::Span, span2::Span; options...) = inv(Gram(span1, span2; options...))

MixedGram(span1::Span, span2::Span; options...) = Gram(dual(span1; options...), span2; options...)

function grammatrix!(result, src::Span, dest::Span; options...)
  @assert size(result, 1) == length(dest)
  @assert size(result, 2) == length(src)
  for i in 1:size(result,1)
    for j in 1:size(result,2)
      result[i,j] = dot(dest, src, i, j; options...)
    end
  end
  result
end

dot(span1::Span1d, span2::Span1d, f1::Function, f2::Function, nodes::Array=native_nodes(set(span1), set(span2)); options...)  =
    dot(x->conj(f1(x))*f2(x), nodes; options...)

dot(span1::Span1d, span2::Span1d, f1::Int, f2::Int, nodes::Array=native_nodes(set(span1), set(span2)); options...) =
    dot(span1, span2, x->eval_element(set(span1), f1, x),x->eval_element(set(span2), f2, x), nodes; options...)

function native_nodes(set1::FunctionSet1d, set2::FunctionSet1d)
  @assert left(set1) ≈ left(set2)
  @assert right(set1) ≈ right(set2)
  [left(set1), right(set1)]
end

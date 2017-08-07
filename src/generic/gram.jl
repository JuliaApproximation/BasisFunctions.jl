# gram.jl

#################
## Gram operators
#################
"""
The gram operator A of the given basisfunction, i.e., A_ij = <ϕ_i,ϕ_j>, if ϕ_i is the ith basisfunction
"""
Gram(b::FunctionSet; options...) = Gram(b, is_orthonormal(b)? Val{true}: Val{false}; options...)

Gram(b::FunctionSet, ::Type{Val{true}}; options...) = IdentityOperator(b,b)

function Gram(set::FunctionSet, ::Type{Val{false}}; options...)
  if is_orthogonal(set)
    d = zeros(eltype(set), length(set))
    gramdiagonal!(d, set; options...)
    DiagonalOperator(set, set, d)
  else
    A = zeros(eltype(set),length(set),length(set))
    grammatrix!(A,set; options...)
    MatrixOperator(set, set, A)
  end
end

"""
The dual gram operator A of the given basisfunction, i.e., A_ij = <ϕ_i,ϕ_j>, if ϕ_i is the ith dual basisfunction
"""
DualGram(b::FunctionSet; options...) = DualGram(b, is_biorthogonal(b)? Val{true}: Val{false}; options...)

DualGram(b::FunctionSet, ::Type{Val{true}}; options...) = inv(Gram(b; options...))

"""
The mixed gram operator A of the given basisfunction, i.e., A_ij = <ϕ_i,ψ_j>, if ϕ_i is the ith dual basisfunction and ψ_j the jth basisfunction
"""
MixedGram(b::FunctionSet; options...) = MixedGram(b, is_biorthogonal(b)? Val{true}: Val{false}; options...)

MixedGram(b::FunctionSet, ::Type{Val{true}}; options...) = IdentityOperator(b,b)

function grammatrix!(result, b::FunctionSet; options...)
  for i in 1:size(result,1)
    for j in i:size(result,2)
      I = dot(b, i, j; options...)
      result[i,j] = I
      if i!= j
        result[j,i] = conj(I)
      end
    end
  end
  result
end

function gramdiagonal!(result, b::FunctionSet; options...)
  for i in 1:size(result,1)
    result[i] = dot(b, i, i; options...)
  end
  result
end

################################################################################################
## Take inner products between function and basisfunctions. Used in continous approximation case.
################################################################################################
"""
Project the function on the function space spanned by the functionset by taking innerproducts with the elements of the set.
"""
project(b, f::Function, ELT = eltype(b); options...) = project!(zeros(ELT,size(b)), b, f; options...)

function project!(result, b, f::Function; options...)
    for i in eachindex(result)
        result[i] = dot(b, i, f; options...)
    end
    result
end

function dot{T}(f::Function, nodes::Array{T,1}; abstol=0, reltol=sqrt(eps(T)), verbose=false, options...)
    (I,e) = QuadGK.quadgk(x->f(x), nodes...; reltol=reltol, abstol=abstol)
    (e > sqrt(reltol) && verbose) && (warn("Dot product did not converge"))
    I
end

native_nodes(set::FunctionSet1d) = [left(set), right(set)]

dot(set::FunctionSet1d, f1::Function, f2::Function, nodes::Array=native_nodes(set); options...)  =
    dot(x->conj(f1(x))*f2(x), nodes; options...)

dot(set::FunctionSet, f1::Int, f2::Function, nodes::Array=native_nodes(set); options...) =
    dot(set, x->eval_element(set, f1, x), f2, nodes; options...)

dot(set::FunctionSet, f1::Int, f2::Int, nodes::Array=native_nodes(set); options...) =
    dot(set, x->eval_element(set, f1, x),x->eval_element(set, f2, x), nodes; options...)

##########################
## Discrete Gram operators
##########################
oversampled_grid(b::FunctionSet, oversampling::Real) = grid(resize(b, length_oversampled_grid(b, oversampling)))

length_oversampled_grid(b::FunctionSet, oversampling::Real)::Int = approx_length(b, basis_oversampling(b, oversampling)*length(b))

basis_oversampling(set::FunctionSet, sampling_factor::Real) =  sampling_factor

default_oversampling(b::FunctionSet) = 1
# E'E/N
DiscreteGram{N,T}(b::FunctionSet{N,T}; oversampling = default_oversampling(b)) =
  1/real(T)(discrete_gram_scaling(b, oversampling))*UnNormalizedGram(b, oversampling)

function UnNormalizedGram{N,T}(b::FunctionSet{N,T}, oversampling = 1)
  grid = oversampled_grid(b, oversampling)
  evaluation_operator(b, grid)'*evaluation_operator(b, grid)
end

# discrete_gram_scaling{N,T}(b::FunctionSet{N,T}, oversampling) = length_oversampled_grid(b, oversampling)
discrete_gram_scaling{N,T}(b::FunctionSet{N,T}, oversampling) = length(b)

# Ẽ'Ẽ/N and since Ẽ = NE^{-1}'
DiscreteDualGram{N,T}(b::FunctionSet{N,T}; oversampling = default_oversampling(b)) = inv(DiscreteGram(b; oversampling=oversampling))

# Ẽ'E/N
DiscreteMixedGram(b::FunctionSet; oversampling=default_oversampling(b)) = IdentityOperator(b,b)



#################
## Gram operators extended
#################

dual(set::FunctionSet; options...) = dual(set, Val{is_orthonormal(set)}; options...)
dual(set::FunctionSet, ::Type{Val{true}}; options...) = set
dual(set::FunctionSet, ::Type{Val{false}}; options...) = error("Dual of $(set) is not known.")
"""
The gram operator A of the given basisfunction, i.e., A_ij = <ϕ_i,ϕ_j>, if ϕ_i is the ith basisfunction
"""
function Gram(src::FunctionSet, dest::FunctionSet; options...)
    A = zeros(eltype(src, dest),length(dest),length(src))*NaN
    grammatrix!(A, src, dest; options...)
    MatrixOperator(src, dest, A)
end

DualGram(set1::FunctionSet, set2::FunctionSet; options...) = inv(Gram(set1, set2; options...))

MixedGram(set1::FunctionSet, set2::FunctionSet; options...) = Gram(dual(set1), set2; options...)

function grammatrix!(result, src::FunctionSet, dest::FunctionSet; options...)
  @assert size(result, 1) == length(dest)
  @assert size(result, 2) == length(src)
  for i in 1:size(result,1)
    for j in 1:size(result,2)
      result[i,j] = dot(dest, src, i, j; options...)
    end
  end
  result
end

dot(set1::FunctionSet1d, set2::FunctionSet1d, f1::Function, f2::Function, nodes::Array=native_nodes(set1, set2); options...)  =
    dot(x->conj(f1(x))*f2(x), nodes; options...)

dot(set1::FunctionSet, set2::FunctionSet, f1::Int, f2::Int, nodes::Array=native_nodes(set1, set2); options...) =
    dot(set1, set2, x->eval_element(set1, f1, x),x->eval_element(set2, f2, x), nodes; options...)

function native_nodes(set1::FunctionSet, set2::FunctionSet)
  @assert left(set1) ≈ left(set2)
  @assert right(set1) ≈ right(set2)
  [left(set1), right(set1)]
end

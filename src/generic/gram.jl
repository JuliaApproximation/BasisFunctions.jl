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
    (I,e) = quadgk(x->f(x), nodes...; reltol=reltol, abstol=abstol)
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
# E'E/N
DiscreteGram{N,T}(b::FunctionSet{N,T}, g=grid(b)) = (1/real(T)(length(b)))*evaluation_operator(b, g)'*evaluation_operator(b, g)
# Ẽ'Ẽ/N and since Ẽ = NE^{-1}
DiscreteDualGram{N,T}(b::FunctionSet{N,T}, g=grid(b)) = inv(DiscreteGram(b, g))
# Ẽ'E/N
DiscreteMixedGram(b::FunctionSet, g=grid(b)) = IdentityOperator(b,b)


"""
A 'ComplexifiedDict' is a dictionary for which the coefficient type is the complex version of the original dictionary. It is obtained by calling dict_promote_coeftype() on a dictionary that does not implement this method.

"""
struct ComplexifiedDict{D,S,T} <: DerivedDict{S,T}
    superdict   :: D
end

const ComplexifiedDict1d{D,S<:Number,T<:Number} = ComplexifiedDict{D,S,T}

ComplexifiedDict(d::Dictionary{S,T}) where {S,T} = ComplexifiedDict{typeof(d),S,T}(d)

similar_dictionary(s::ComplexifiedDict, s2::Dictionary) = ComplexifiedDict(s2)

function dict_promote_coeftype(d::Dictionary{S,T},::Type{U}) where {S,T,U<:Complex}
    if coeftype(d)==real(U)
        return ComplexifiedDict(d)
    else
        dn = dict_promote_coeftype(d,real(U))
        return ComplexifiedDict(dn)
    end
end

apply_map(s::ComplexifiedDict, map) = mapped_dict(s, map)
dict_promote_coeftype(d::ComplexifiedDict{S,T}, ::Type{U}) where {S,T,U<:Complex} = similar_dictionary(d,dict_promote_coeftype(superdict(d),real(U)))
dict_promote_coeftype(d::Dictionary{S,T},::Type{U}) where {S,T,U<:Real} = warn("coefficient type promotion not supported by $(d)")

coefficient_type(dict::ComplexifiedDict) = complex(coefficient_type(superdict(dict)))

transform_dict(dict::ComplexifiedDict) = dict_promote_coeftype(transform_dict(superdict(dict)),coefficient_type(dict))

grid_evaluation_operator(s::ComplexifiedDict, dgs::GridBasis, grid::AbstractGrid; options...) = select_grid_evaluation_operator(s,dgs,grid;options...)
grid_evaluation_operator(s::ComplexifiedDict, dgs::GridBasis, grid::AbstractSubGrid; options...) = select_grid_evaluation_operator(s,dgs,grid;options...)

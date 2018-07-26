# vector_dict.jl

struct VectorvaluedDict{DICTS,S,N,T} <: CompositeDict{S,SVector{N,T}}
    dicts   ::  DICTS
    offsets ::  Vector{Int}

    function VectorvaluedDict{DICTS,S,N,T}(dicts) where {DICTS,S,N,T}
        offsets = compute_offsets(dicts)
        new(dicts, offsets)
    end
end

function VectorvaluedDict(dict::Dictionary1d, ::Val{N}) where {N}
    dicts = ntuple(d->dict, Val{N}())
    S = domaintype(dict)
    T = codomaintype(dict)
    VectorvaluedDict{typeof(dicts),S,N,T}(dicts)
end

unit_vector(k, ::Val{N}, T) where {N} = SVector{N,T}(ntuple( i->i==k, Val{N}()))

function unsafe_eval_element(dict::VectorvaluedDict{DICTS,S,N,T}, idx::MultilinearIndex, x) where {DICTS,S,N,T}
    idx1 = idx[1]
    idx2 = idx[2]
    z = unit_vector(idx1, Val{N}(), T) * unsafe_eval_element(element(dict, idx1), idx2, x)
end

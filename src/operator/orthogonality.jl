

isbiorthogonal(op::DictionaryOperator) = true
isorthogonal(op::DictionaryOperator) = isdiag(op) || isorthonormal(op)
isorthonormal(op::DictionaryOperator) = false

isorthonormal(::IdentityOperator) = true
isorthonormal(::IndexExtension) = true
isorthonormal(::IndexRestriction) = true
function isorthonormal(op::CompositeOperator)
    @warn "just a rough estimate on orthonormality"
    mapreduce(isorthonormal, &, components(op))
end


function isorthogonal(op::CompositeOperator)
    @warn "just a rough estimate on orthonormality"
    mapreduce(isorthogonal, &, components(op))
end

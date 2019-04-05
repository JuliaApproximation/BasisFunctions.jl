

isbiorthogonal(op::DictionaryOperator) = true
isorthogonal(op::DictionaryOperator) = isdiag(op) || isorthonormal(op)
isorthonormal(op::DictionaryOperator) = false

isorthonormal(::IdentityOperator) = true
isorthonormal(::IndexExtensionOperator) = true
isorthonormal(::IndexRestrictionOperator) = true
function isorthonormal(op::CompositeOperator)
    @warn "just a rough estimate on orthonormality"
    reduce(&, map(isorthonormal, elements(op)))
end


function isorthogonal(op::CompositeOperator)
    @warn "just a rough estimate on orthonormality"
    reduce(&, map(isorthogonal, elements(op)))
end

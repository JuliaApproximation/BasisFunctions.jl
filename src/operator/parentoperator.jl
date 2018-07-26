"""
A ParentOperator is an operator that has children, i.\e.\ combines the actions of sub-operators
"""
abstract type ParentOperator{T} <: DictionaryOperator{T}
end

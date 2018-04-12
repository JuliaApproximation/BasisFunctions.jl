# periodization.jl

"A periodization modifier results in the periodization of a dictionary."
struct Periodization{T} <: DictModifier
    period  ::  T
    support ::  T
end

periodize(dict::Dictionary, period) = TypedModifiedDictionary(dict, Periodization(period))

mod_unsafe_eval_element(mod::Periodization, dict, idx, x) =
    unsafe_eval_element(dict, idx, x) +
    unsafe_eval_element(dict, idx, x+mod.period) +
    unsafe_eval_element(dict, idx, x-mod.period)

mod_unsafe_eval_element_derivative(mod::Periodization, dict, idx, x) =
    unsafe_eval_element_derivative(dict, idx, x) +
    unsafe_eval_element_derivative(dict, idx, x+mod.period) +
    unsafe_eval_element_derivative(dict, idx, x-mod.period)

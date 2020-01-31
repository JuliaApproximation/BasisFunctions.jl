

(*)(op::ProjectionSampling, object::SynthesisOperator) = apply(op, object)

apply(op::SamplingOperator, dict::Dictionary; options...) =
	apply(op, SynthesisOperator(dict); options...)

(*)(op::SamplingOperator, object::SynthesisOperator) = apply(op, object)
apply(sampling::SamplingOperator, op::SynthesisOperator; T = op_eltype(dictionary(op), dest(sampling)), options...) =
    evaluation(T, dictionary(op), grid(sampling); options...)

function apply(analysis::ProjectionSampling, synthesis::SynthesisOperator; T=op_eltype(dictionary(synthesis), dest(analysis)), options...)
	synthesisdict = dictionary(synthesis)
	analysismeasure = measure(analysis)
	analysisdict = dictionary(analysis)
	# Check whether we can just compute the gram matrix instead.
	if isequal(analysisdict, synthesisdict) && size(analysisdict)==size(synthesisdict)
		gramoperator(synthesisdict, analysismeasure; T=T, options...)
	else
		mixedgramoperator(analysisdict, synthesisdict, analysismeasure; T=T, options...)
	end
end

function adjoint(op::SynthesisOperator)
    if op.measure == nothing
        @warn("Adjoint of $(typeof(op)) does not exist if measure is not defined. Returning inverse.")
        inv(op)
    else
        AnalysisOperator(dictionary(op), op.measure, dest_space(op))
    end
end


function adjoint(op::AnalysisOperator)
    if src_space(op) isa Span && iscompatible(dictionary(src_space(op)), dictionary(op))
        SynthesisOperator(dictionary(op), measure(op))
    else
        error("Adjoint of $(typeof(op)) does not exist if spaces do not match.")
    end
end

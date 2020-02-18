
using PGFPlotsX
import PGFPlotsX: Plot, PlotInc, Options, Axis

for F in (:Expansion,)
    @eval begin
        Axis(F::$F, trailing...; opts...) =
            @pgf Axis({ymode="log"}, Plot(F, trailing...; opts...))
    end
end

for plot in (:Plot, :PlotInc)
    _plot = Meta.parse("_"*string(plot))
    for EXP in (:Expansion,)
        @eval begin
            $(plot)(F::$EXP; opts...) =
                $(plot)(Options(), F; opts...)
            $(plot)(f::Function, F::$EXP; opts...) =
                $(plot)(Options(), f, F; opts...)
            $(plot)(F::$EXP, f::Function; opts...) =
                $(plot)(Options(), f, F)
            $(plot)(F::$EXP, f::$EXP; opts...) =
                $(plot)(Options(), f, F; opts...)
        end
    end
    @eval begin
        function $(plot)(options::Options, F::Expansion; n=200, plot_complex=false, opts...)
            grid = plotgrid(dictionary(F), n)
            if plot_complex
                vals = F(grid)
                options = @pgf {options..., no_markers}
                $(plot)(options, Table([grid, BasisFunctions.postprocess(dictionary(F), grid, real.(vals))])),
                    $(plot)(options, Table([grid, BasisFunctions.postprocess(dictionary(F), grid, imag.(vals))]))
            else
                vals = real.(F(grid))
                options = @pgf {options..., no_markers}
                $(plot)(options, Table([grid, BasisFunctions.postprocess(dictionary(F), grid, vals)]))
            end
        end

        $(plot)(options::Options, F::Expansion, f::Function; opts...) =
            $(_plot)(options, f, F; opts...)

        $(plot)(options::Options, f::Function, F::Expansion; opts...) =
            $(_plot)(options, f, F; opts...)
        $(plot)(options::Options, f::Expansion, F::Expansion; opts...) =
            $(_plot)(options, f, F; opts...)


        function $(_plot)(options::Options, f, F::Expansion; plot_complex=false, n=200, opts...)
            grid = plotgrid(dictionary(F), n)
            vals = abs.(f.(grid) - F.(grid))
            options = @pgf {options..., no_markers}
            $(plot)(options, Table([grid, BasisFunctions.postprocess(dictionary(F), grid, vals)]))
        end
    end
end

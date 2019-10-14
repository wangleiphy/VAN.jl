using Zygote
using Zygote: @adjoint, @nograd

@adjoint function free_energy(K::Array, model::AutoRegressiveModel, samples)
    free_energy(K, model, samples), function (adjy)
        adjmodel = grad_model(K, model, samples) .* adjy
        return (nothing, adjmodel, nothing)
    end
end
@nograd gen_samples, createmasks

using Zygote
using Zygote: @adjoint, @nograd

@adjoint function free_energy(K::Matrix{T}, model::AbstractSampler, samples)
    free_energy(K, model, samples), function (adjy)
        adjmodel = grad_model(K, model, samples) .* adjy
        return (nothing, adjmodel, nothing)
    end
end

@adjoint function PSAModel(nbit::Int) 
    PSAmodel(nbit) , _ -> nothing 
end 

@adjoint function AutoRegressiveModel(nbit::Int, nhiddens) 
    AutoRegressivemodel(nbit) , _ -> nothing 
end 

@nograd gen_samples, createmasks

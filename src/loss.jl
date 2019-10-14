using Flux.Optimise:update!
export train, loss

function get_energy(K::Array, samples)
    energy = sum(samples .* (K*samples), dims=1)
end

function free_energy(K::Array, model::AutoRegressiveModel, samples)
    return mean(get_energy(K, samples) .+ get_logp(model, samples))
end

function loss(K::Array, model::AutoRegressiveModel, nbatch::Int)
    samples = gen_samples(model, nbatch)
    free_energy(K, model, samples)
end

function loss_reinforce(K::Array, model::AutoRegressiveModel, samples)
    e = get_energy(K, samples)
    logp = get_logp(model, samples)
    f = e .+ logp
    b = mean(f)
    return mean(logp.* (f .- b))
end

function grad_model(K::Array, model::AutoRegressiveModel, samples)
    model_grad = gradient(loss_reinforce, K, model, samples)[2]
    (model_grad.W..., model_grad.b...)
end

function train(K::Array, model::AutoRegressiveModel; optimizer=ADAM(0.1), nbatch::Int=100, niter::Int=100)
    θ = model_parameters(model)
    for i = 1:niter
        _, gθ, _ = gradient(loss, K, model, nbatch)
        update!.(Ref(optimizer), θ, gθ)
        model_dispatch!(model, θ)
        println("$i, Free Energy = ", loss(K, model, nbatch))
    end
end

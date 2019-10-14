using Flux.Optimise: ADAM, update!
using StatsBase
using Random
Random.seed!(42)

using VAN

nbits = 10
nsamples = 1000
nhiddens = [10, 20, 10]

K = randn(nbits, nbits)
K = (K+K')/2
@show exact_free_energy(K)

model = build_model(nbits, nhiddens)

loss(K, model, nsamples)

train(K, model; optimizer=ADAM(0.1), nbatch=nsamples, niter=100)

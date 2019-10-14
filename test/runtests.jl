using Test, Random, StatsBase
using Zygote
using VAN
using VAN: bitarray, get_logp, network

@testset "normalize" begin
    Random.seed!(2)
    nbits = 6
    nhiddens = [10,10]
    model = build_model(nbits, nhiddens)
    configs = bitarray(collect(0:(1<<nbits-1)), nbits)
    logp = get_logp(model, configs)
    norm = sum(exp.(logp))

    @test isapprox(norm, 1.0, rtol=1e-5)
end

@testset "autoregressive" begin
    Random.seed!(2)
    nbits = 6
    nbatchs = 4
    nhiddens = [10,20,30]
    model = build_model(nbits, nhiddens)
    samples = rand(0:1, nbits, nbatchs)

    f(model, samples, n, b) = network(model, samples)[n, b]

    for n in 1:nbits
        for b in 1:nbatchs
            g = gradient(f, model, samples, n, b)[2]
            dependency = (g .!= 0)
            correct = BitArray( x< n && y==b  for x = 1:nbits, y = 1:nbatchs)
            @test dependency == correct
        end
    end
end

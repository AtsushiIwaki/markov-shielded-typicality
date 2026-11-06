using Test

include("../src/mps_auxiliary.jl")

@testset "Markov entropy of correlated physical sites" begin
    sites = siteinds("S=1/2", 4)
    all_up = MPS(sites, fill("Up", 4))
    all_down = MPS(sites, fill("Dn", 4))
    psi = LinearAlgebra.normalize(all_up + all_down)

    # The middle two spins are physical. Each has entropy log(2), and
    # together they also have entropy log(2).
    @test markov_entropy_aux(psi, 1) ≈ log(2)
    @test markov_entropy_aux(psi, 0) ≈ 2 * log(2)
end

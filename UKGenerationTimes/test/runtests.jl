using Test

@testset "UKGenerationTimes" begin
    include("test_data_import.jl")
    include("test_likelihood.jl")
    include("test_infectiousness.jl")
end

# This file is adapted from test/abstractarray.jl from Julia.

# This file is a part of Julia. License is MIT: https://julialang.org/license

A_abs = rand(5,4,3)
A = OpenCV.Mat(A_abs)
@testset "Bounds checking" begin
    @test checkbounds(Bool, A, 1, 1, 1) == true
    @test checkbounds(Bool, A, 5, 4, 3) == true
    @test checkbounds(Bool, A, 0, 1, 1) == false
    @test checkbounds(Bool, A, 1, 0, 1) == false
    @test checkbounds(Bool, A, 1, 1, 0) == false
    @test checkbounds(Bool, A, 6, 4, 3) == false
    @test checkbounds(Bool, A, 5, 5, 3) == false
    @test checkbounds(Bool, A, 5, 4, 4) == false
    @test checkbounds(Bool, A, 1) == true           # linear indexing
    @test checkbounds(Bool, A, 60) == true
    @test checkbounds(Bool, A, 61) == false
    @test checkbounds(Bool, A, 2, 2, 2, 1) == true  # extra indices
    @test checkbounds(Bool, A, 2, 2, 2, 2) == false
    @test checkbounds(Bool, A, 1, 1)  == false
    @test checkbounds(Bool, A, 1, 12) == false
    @test checkbounds(Bool, A, 5, 12) == false
    @test checkbounds(Bool, A, 1, 13) == false
    @test checkbounds(Bool, A, 6, 12) == false
end

@testset "single CartesianIndex" begin
    @test checkbounds(Bool, A, CartesianIndex((1, 1, 1))) == true
    @test checkbounds(Bool, A, CartesianIndex((5, 4, 3))) == true
    @test checkbounds(Bool, A, CartesianIndex((0, 1, 1))) == false
    @test checkbounds(Bool, A, CartesianIndex((1, 0, 1))) == false
    @test checkbounds(Bool, A, CartesianIndex((1, 1, 0))) == false
    @test checkbounds(Bool, A, CartesianIndex((6, 4, 3))) == false
    @test checkbounds(Bool, A, CartesianIndex((5, 5, 3))) == false
    @test checkbounds(Bool, A, CartesianIndex((5, 4, 4))) == false
    @test checkbounds(Bool, A, CartesianIndex((1,))) == false
    @test checkbounds(Bool, A, CartesianIndex((60,))) == false
    @test checkbounds(Bool, A, CartesianIndex((61,))) == false
    @test checkbounds(Bool, A, CartesianIndex((2, 2, 2, 1,))) == true
    @test checkbounds(Bool, A, CartesianIndex((2, 2, 2, 2,))) == false
    @test checkbounds(Bool, A, CartesianIndex((1, 1,)))  == false
    @test checkbounds(Bool, A, CartesianIndex((1, 12,))) == false
    @test checkbounds(Bool, A, CartesianIndex((5, 12,))) == false
    @test checkbounds(Bool, A, CartesianIndex((1, 13,))) == false
    @test checkbounds(Bool, A, CartesianIndex((6, 12,))) == false
end

@testset "mix of CartesianIndex and Int" begin
    @test checkbounds(Bool, A, CartesianIndex((1,)), 1, CartesianIndex((1,))) == true
    @test checkbounds(Bool, A, CartesianIndex((5, 4)), 3)  == true
    @test checkbounds(Bool, A, CartesianIndex((0, 1)), 1)  == false
    @test checkbounds(Bool, A, 1, CartesianIndex((0, 1)))  == false
    @test checkbounds(Bool, A, 1, 1, CartesianIndex((0,))) == false
    @test checkbounds(Bool, A, 6, CartesianIndex((4, 3)))  == false
    @test checkbounds(Bool, A, 5, CartesianIndex((5,)), 3) == false
    @test checkbounds(Bool, A, CartesianIndex((5,)), CartesianIndex((4,)), CartesianIndex((4,)))  == false
end

@testset "vector indices" begin
    @test checkbounds(Bool, A, 1:5, 1:4, 1:3) == true
    @test checkbounds(Bool, A, 0:5, 1:4, 1:3) == false
    @test checkbounds(Bool, A, 1:5, 0:4, 1:3) == false
    @test checkbounds(Bool, A, 1:5, 1:4, 0:3) == false
    @test checkbounds(Bool, A, 1:6, 1:4, 1:3) == false
    @test checkbounds(Bool, A, 1:5, 1:5, 1:3) == false
    @test checkbounds(Bool, A, 1:5, 1:4, 1:4) == false
    @test checkbounds(Bool, A, 1:60) == true
    @test checkbounds(Bool, A, 1:61) == false
    @test checkbounds(Bool, A, 2, 2, 2, 1:1) == true  # extra indices
    @test checkbounds(Bool, A, 2, 2, 2, 1:2) == false
    @test checkbounds(Bool, A, 1:5, 1:4) == false
    @test checkbounds(Bool, A, 1:5, 1:12) == false
    @test checkbounds(Bool, A, 1:5, 1:13) == false
    @test checkbounds(Bool, A, 1:6, 1:12) == false
end

@testset "logical" begin
    @test checkbounds(Bool, A, trues(5), trues(4), trues(3)) == true
    @test checkbounds(Bool, A, trues(6), trues(4), trues(3)) == false
    @test checkbounds(Bool, A, trues(5), trues(5), trues(3)) == false
    @test checkbounds(Bool, A, trues(5), trues(4), trues(4)) == false
    @test checkbounds(Bool, A, trues(60)) == true
    @test checkbounds(Bool, A, trues(61)) == false
    @test checkbounds(Bool, A, 2, 2, 2, trues(1)) == true  # extra indices
    @test checkbounds(Bool, A, 2, 2, 2, trues(2)) == false
    @test checkbounds(Bool, A, trues(5), trues(12)) == false
    @test checkbounds(Bool, A, trues(5), trues(13)) == false
    @test checkbounds(Bool, A, trues(6), trues(12)) == false
    @test checkbounds(Bool, A, trues(5, 4, 3)) == true
    @test checkbounds(Bool, A, trues(5, 4, 2)) == false
    @test checkbounds(Bool, A, trues(5, 12)) == false
    @test checkbounds(Bool, A, trues(1, 5), trues(1, 4, 1), trues(1, 1, 3)) == false
    @test checkbounds(Bool, A, trues(1, 5), trues(1, 4, 1), trues(1, 1, 2)) == false
    @test checkbounds(Bool, A, trues(1, 5), trues(1, 5, 1), trues(1, 1, 3)) == false
    @test checkbounds(Bool, A, trues(1, 5), :, 2) == false
end

@testset "array of CartesianIndex" begin
    @test checkbounds(Bool, A, [CartesianIndex((1, 1, 1))]) == true
    @test checkbounds(Bool, A, [CartesianIndex((5, 4, 3))]) == true
    @test checkbounds(Bool, A, [CartesianIndex((0, 1, 1))]) == false
    @test checkbounds(Bool, A, [CartesianIndex((1, 0, 1))]) == false
    @test checkbounds(Bool, A, [CartesianIndex((1, 1, 0))]) == false
    @test checkbounds(Bool, A, [CartesianIndex((6, 4, 3))]) == false
    @test checkbounds(Bool, A, [CartesianIndex((5, 5, 3))]) == false
    @test checkbounds(Bool, A, [CartesianIndex((5, 4, 4))]) == false
    @test checkbounds(Bool, A, [CartesianIndex((1, 1))], 1) == true
    @test checkbounds(Bool, A, [CartesianIndex((5, 4))], 3) == true
    @test checkbounds(Bool, A, [CartesianIndex((0, 1))], 1) == false
    @test checkbounds(Bool, A, [CartesianIndex((1, 0))], 1) == false
    @test checkbounds(Bool, A, [CartesianIndex((1, 1))], 0) == false
    @test checkbounds(Bool, A, [CartesianIndex((6, 4))], 3) == false
    @test checkbounds(Bool, A, [CartesianIndex((5, 5))], 3) == false
    @test checkbounds(Bool, A, [CartesianIndex((5, 4))], 4) == false
end

println("OpenCV.Mat tests passed")
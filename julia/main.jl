# Packages used in this Project: PyCall, Plots, DelimitedFiles, MLJ
using PyCall, Plots, DelimitedFiles, MLJ

dirs = ["julia/media", "julia/data"]
for dir in dirs
	if !isdir(dir)
		mkdir(dir)
	end
end

include("tasks/A1.jl")

A1()
import numpy as np
from antlr4 import *
import time

from antlr.interpreter import MatrixInterpreter

from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.eval("import Pkg; Pkg.activate(\"./utils/tensorgames/\")")
from julia import TensorGames

N = 16
interpreter = MatrixInterpreter(np.zeros((N, N)))

prog = 'set(A, 7, 9, 0, 16, 1);'

mat = interpreter(InputStream(prog))

t0 = time.time()
for _ in range(1000):
    sol = TensorGames.compute_equilibrium([mat, -mat])
tt = time.time() - t0

print(sol.x)
print(TensorGames.expected_cost(sol.x, mat))

print('avg time', tt/1000)
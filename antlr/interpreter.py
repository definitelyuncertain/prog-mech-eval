from antlr4 import *
from antlr.matrix_grammar.matrixLexer import matrixLexer
from antlr.matrix_grammar.matrixParser import matrixParser
from antlr.matrix_grammar.matrixVisitor import matrixVisitor


class InterpreterVisitor(matrixVisitor):
    """
    ANTLR 4 visitor class for the matrix grammar.
    """
    def __init__(self, base_array):
        self.A = base_array

    def visitNumberExpr(self, ctx):
        value = ctx.getText()
        # print('number', value)
        return float(value)

    def visitParenExpr(self, ctx):
        return self.visit(ctx.expr())

    def visitFunctionExpr(self, ctx):
        self.visit(ctx.function())

    def visitFunctionCallExpr(self, ctx):
        # print('ctx', ctx.getText())
        # print('calling function', ctx.fnname.text)
        if ctx.fnname.text == 'set':
            args = [self.visit(ctx.args.expr(i)) for i in range(6)]
            # print(args)
            args = args[1:]
            self.A[int(args[0]):int(args[1])+1, int(args[2]):int(args[3])+1] = args[4]
        else:
            raise NotImplementedError

    def visitVarExpr(self, ctx):
        # print(f"Var {ctx.getText()}")
        return ctx.getText()


class MatrixInterpreter:
    """
    Interpreter for programs from the matrix DSL
    that execute the program and return the
    matrix produced as a numpy array.
    """
    def __init__(self, base_array):
        self.A = base_array.copy()

    def __call__(self, prog):
        lexer = matrixLexer(prog)
        stream = CommonTokenStream(lexer)
        parser = matrixParser(stream)
        tree = parser.parse()
        visitor = InterpreterVisitor(self.A.copy())
        _ = visitor.visit(tree)
        return visitor.A

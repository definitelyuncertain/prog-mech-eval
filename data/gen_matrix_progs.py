import numpy as np


def random_set_statement(N, dim_ranges, val_range, size_ranges=None):
    """
    Construct a random set statement for a DSL program given a range
    of possible locations of blocks of ones or zeroes and their sizes.
    :param N: Matrix size.
    :param dim_ranges: list with row and column ranges, each of which is
    a list or int for range.
    :param val_range: List of set values to choose randomly from, or int for range.
    :param size_ranges: list with possible row/column sizes of blocks, each of which is
    a list or int for range.
    :return: set statement as a string.
    """
    st = 'set(A, '
    for dim in range(2):  # Dimension ranges
        dim_range = dim_ranges[dim]
        v1 = np.random.choice(dim_range)
        if (size_ranges is None or (size_ranges[dim] is None)):
            v2 = np.random.choice(np.arange(v1, N))
            # Random size if size range provided
        else:
            v2 = min(v1 + np.random.choice(size_ranges[dim])-1, N-1)
        st += f"{v1}, {v2}, "
    st += f"{np.random.choice(val_range)});\n" # Set value
    return st


def gen_matrix_progs_v0(N, num_programs, num_statements, **kwargs):
    """
    Construct programs with a given number of set statements
    with random arguments in the appropriate ranges.

    :param N: Matrix size
    :param num_programs: number of programs to generate.
    :param num_statements: number of statements in each program.
    :return: set of programs (in string form).
    """
    progs = set()
    while len(progs) < num_programs:  # Program
        prog = ''
        for i in range(num_statements):  # Statement
            st = random_set_statement(N, [N, N], 2)
            prog += st
        progs.add(prog)
    return progs


def gen_matrix_progs_col_stripe(N, num_programs, num_statements, stripe_size, stripe_chance=0.5, **kwargs):
    """
    Similar to gen_matrix_progs_v0 but also adds a vertical stripe of ones
    with a given probability (the stripe counts as a statement, at most
    one stripe is added).
    :param N: Matrix size
    :param num_programs: number of programs to generate.
    :param num_statements: number of statements in each program.
    :param stripe_size: thickness of vertical stripe.
    :param stripe_chance: chance that a statement becomes a stripe.
    :return: set of programs (in string form).
    """
    progs = set()
    while len(progs) < num_programs:  # Program
        prog = ''
        stripe = False
        for i in range(num_statements):  # Statement
            # Randomly add stripe
            if not stripe and np.random.rand() < stripe_chance:
                st = random_set_statement(N, [[0], N], [1],
                                            size_ranges=[[N], np.arange(stripe_size) + 1])
                stripe = True
            else:
                st = random_set_statement(N, [N, N], 2)
            prog += st
        progs.add(prog)
    return progs

if __name__ == "__main__":
    N = 16
    num_programs = 3
    num_statements = 3
    stripe_size = 3
    for prog in gen_matrix_progs_v0(N, num_programs, num_statements):
        print(prog)
        print('------')
    print("\n=======\n")

    for prog in gen_matrix_progs_col_stripe(N, num_programs, num_statements,
                                          stripe_size):
        print(prog)
        print('------')

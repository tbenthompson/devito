import sympy
import pytest
import numpy as np

from sympy import Symbol, Min
from devito import Grid, Function, solve, TimeFunction, Eq, Operator, norm, Le, Ge
from devito.ir import Expression, FindNodes
from devito.symbolics import (retrieve_functions, retrieve_indexed, evalmin, evalmax,
                              MIN, MAX)


def test_float_indices():
    """
    Test that indices only contain Integers.
    """
    grid = Grid((10,))
    x = grid.dimensions[0]
    x0 = x + 1.0 * x.spacing
    u = Function(name="u", grid=grid, space_order=2)
    indices = u.subs({x: x0}).indexify().indices[0]
    assert len(indices.atoms(sympy.Float)) == 0
    assert indices == x + 1

    indices = u.subs({x: 1.0}).indexify().indices[0]
    assert len(indices.atoms(sympy.Float)) == 0
    assert indices == 1


@pytest.mark.parametrize('dtype,expected', [
    (np.float32, "float r0 = 1.0F/h_x;"),
    (np.float64, "double r0 = 1.0/h_x;")
])
def test_floatification_issue_1627(dtype, expected):
    """
    MFE for issue #1627.
    """
    grid = Grid(shape=(10, 10), dtype=dtype)
    x, y = grid.dimensions

    u = TimeFunction(name='u', grid=grid)

    eq = Eq(u.forward, ((u/x.spacing) + 2.0)/x.spacing)

    op = Operator(eq)

    exprs = FindNodes(Expression).visit(op)
    assert len(exprs) == 2
    assert str(exprs[0]) == expected


def test_is_on_grid():
    grid = Grid((10,))
    x = grid.dimensions[0]
    x0 = x + .5 * x.spacing
    u = Function(name="u", grid=grid, space_order=2)

    assert u._is_on_grid
    assert not u.subs({x: x0})._is_on_grid
    assert all(uu._is_on_grid for uu in retrieve_functions(u.subs({x: x0}).evaluate))


@pytest.mark.parametrize('expr,expected', [
    ('f[x+2]*g[x+4] + f[x+3]*g[x+5] + f[x+4] + f[x+1]',
     ['f[x+2]', 'g[x+4]', 'f[x+3]', 'g[x+5]', 'f[x+1]', 'f[x+4]']),
    ('f[x]*g[x+2] + f[x+1]*g[x+3]', ['f[x]', 'g[x+2]', 'f[x+1]', 'g[x+3]']),
])
def test_canonical_ordering(expr, expected):
    """
    Test that the `expr.args` are stored in canonical ordering.
    """
    grid = Grid(shape=(10,))
    x, = grid.dimensions  # noqa

    f = Function(name='f', grid=grid)  # noqa
    g = Function(name='g', grid=grid)  # noqa

    expr = eval(expr)
    for n, i in enumerate(list(expected)):
        expected[n] = eval(i)

    assert retrieve_indexed(expr) == expected


def test_solve_time():
    """
    Tests that solves only evaluate the time derivative.
    """
    grid = Grid(shape=(11, 11))
    u = TimeFunction(name="u", grid=grid, time_order=2, space_order=4)
    m = Function(name="m", grid=grid, space_order=4)
    dt = grid.time_dim.spacing
    eq = m * u.dt2 + u.dx
    sol = solve(eq, u.forward)
    # Check u.dx is not evaluated. Need to simplify because the solution
    # contains some Dummy in the Derivatibe subs that make equality break.
    # TODO: replace by retreive_derivatives after Fabio's PR
    assert sympy.simplify(u.dx - sol.args[2].args[0].args[1]) == 0
    assert sympy.simplify(sol - (-dt**2*u.dx/m + 2.0*u - u.backward)) == 0


def test_nested_relations():
    """
    Tests min/max with multiple args
    """
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')
    d = a + 1

    assert evalmin([a]) == a
    assert evalmin([a, b]) == MIN(a, b)
    assert evalmin([a, b, c]) == MIN(MIN(a, b), c)
    assert evalmin([a, b, c, d]) == MIN(MIN(a, b), c)

    assert evalmax([a]) == a
    assert evalmax([a, b]) == MAX(a, b)
    assert evalmax([a, b, c]) == MAX(MAX(a, b), c)
    assert evalmax([a, b, c, d]) == MAX(MAX(b, c), d)


def test_multibounds_op():
    """
    Tests min/max with multiple args
    """

    grid = Grid(shape=(16, 16, 16))

    a = Function(name='a', grid=grid)
    b = Function(name='b', grid=grid)
    c = Function(name='c', grid=grid)
    d = Function(name='d', grid=grid)
    a.data[:] = 5

    b = pow(a, 2)
    c = a + 10
    d = 2*a

    f = TimeFunction(name='f', grid=grid, space_order=2)

    f.data[:] = 0.1
    eqns = [Eq(f.forward, f.laplace + f * Min(f, b, c, d))]
    op = Operator(eqns, opt=('advanced'))
    op.apply(time_M=5)
    fnorm = norm(f)

    f.data[:] = 0.1
    eqns = [Eq(f.forward, f.laplace + f * evalmin([f, b, c, d]))]
    op = Operator(eqns, opt=('advanced'))
    op.apply(time_M=5)
    fnorm2 = norm(f)

    assert fnorm == fnorm2


def test_relations_w_assumptions():
    """
    Tests min/max with multiple args
    """
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')

    assert (evalmin((a, b, c, d), [Le(d, a)])
            == MIN(MIN(b, c), d))

    assert evalmin([a]) == a
    assert evalmin([a, b]) == MIN(a, b)
    assert evalmin([a, b, c]) == MIN(MIN(a, b), c)

    assert evalmax([a]) == a
    assert evalmax([a, b]) == MAX(a, b)
    assert evalmax([a, b, c]) == MAX(MAX(a, b), c)
    assert evalmax([a, b, c, d], [Ge(d, a)]) == MAX(MAX(b, c), d)


def test_relations_w_complex_assumptions():
    """
    Tests min/max with multiple args
    """
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')

    assert evalmin([a]) == a
    assert evalmin([a, b], [Le(d, a), Ge(c, b)]) == MIN(a, b)
    assert evalmin([a, b, c]) == MIN(MIN(a, b), c)
    assert (evalmin((a, b, c, d), [Le(d, a), Ge(c, b)])
            == MIN(b, d))
    assert evalmin([a, b, c, d], [Ge(a, b), Ge(d, a), Ge(b, c)]) == c

    assert evalmax([a], [Le(a, a)]) == a
    assert evalmax([a, b], [Le(a, b)]) == b
    assert evalmax([a, b, c], [Le(c, b), Le(c, a)]) == MAX(a, b)
    assert evalmax([a, b, c, d], [Ge(d, a), Ge(a, b), Ge(b, c)]) == d
    assert evalmax([a, b, c, d], [Ge(a, b), Ge(d, a), Ge(b, c)]) == d

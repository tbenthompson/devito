"""User API to specify relationals."""

import sympy

__all__ = ['Le', 'Lt', 'Ge', 'Gt', 'Ne']


class AbstractRel(object):
    """
    Abstract mixin class for objects subclassing sympy Relationals.
    """
    @property
    def negated(self):
        return ops.get(self.func)(*self.args)

    @property
    def reversed(self):
        return rev.get(self.func)(self.rhs, self.lhs)

    @property
    def subdomain(self):
        """The SubDomain in which self is defined."""
        return self._subdomain


class Le(AbstractRel, sympy.Le):
    """
    A less-than or equal ("<=") relation between two objects, the left-hand side and the
    right-hand side. It can be used to build conditionals but not directly to
    construct an Operator.

    Parameters
    ----------
    lhs : expr-like
        The left-hand side.
    rhs : expr-like, optional
        The right-hand side. Defaults to 0.
    subdomain : SubDomain, optional
        To restrict the evalaution of the relation to a particular sub-region in the
        computational domain.

    Examples
    --------
    `Le` may be used to express a relation (e.g. in a Subdomain).

    >>> from devito import Grid, Function, ConditionalDimension, Eq, Operator, Le
    >>> grid = Grid(shape=(8, 8))
    >>> g = Function(name='g', grid=grid)
    >>> Le(g, 1)
    g(x, y) <= 1
    """

    def __new__(cls, lhs, rhs=0, subdomain=None, **kwargs):
        kwargs.update({'evaluate': False})
        obj = sympy.Le.__new__(cls, lhs, rhs, **kwargs)
        obj._subdomain = subdomain

        return obj


class Lt(AbstractRel, sympy.Lt):
    """
    A less-than ("<") relation between two objects, the left-hand side and the
    right-hand side.It can be used to build conditionals but not directly to
    construct an Operator.

    Parameters
    ----------
    lhs : expr-like
        The left-hand side.
    rhs : expr-like, optional
        The right-hand side. Defaults to 0.
    subdomain : SubDomain, optional
        To restrict the evalaution of the relation to a particular sub-region in the
        computational domain.

    Examples
    --------
    `Lt` may be used to express a relation (e.g. in a Subdomain).

    >>> from devito import Grid, Function, ConditionalDimension, Eq, Operator, Lt
    >>> grid = Grid(shape=(8, 8))
    >>> g = Function(name='g', grid=grid)
    >>> Lt(g, 1)
    g(x, y) < 1
    """

    def __new__(cls, lhs, rhs=0, subdomain=None, **kwargs):
        kwargs.update({'evaluate': False})
        obj = sympy.Lt.__new__(cls, lhs, rhs, **kwargs)
        obj._subdomain = subdomain

        return obj


class Ge(AbstractRel, sympy.Ge):
    """
    A greater-than or equal (">=") relation between two objects, the left-hand side and
    the right-hand side. It can be used to build conditionals but not directly to
    construct an Operator.

    Parameters
    ----------
    lhs : expr-like
        The left-hand side.
    rhs : expr-like, optional
        The right-hand side. Defaults to 0.
    subdomain : SubDomain, optional
        To restrict the evalaution of the relation to a particular sub-region in the
        computational domain.

    Examples
    --------
    `Ge` may be used to express a relation (e.g. in a Subdomain).

    >>> from devito import Grid, Function, ConditionalDimension, Eq, Operator, Ge
    >>> grid = Grid(shape=(8, 8))
    >>> g = Function(name='g', grid=grid)
    >>> Ge(g, 1)
    g(x, y) >= 1
    """

    def __new__(cls, lhs, rhs=0, subdomain=None, **kwargs):
        kwargs.update({'evaluate': False})
        obj = sympy.Ge.__new__(cls, lhs, rhs, **kwargs)
        obj._subdomain = subdomain

        return obj


class Gt(AbstractRel, sympy.Gt):
    """
    A greater-than (">") relation between two objects, the left-hand side and the
    right-hand side. It can be used to build conditionals but not directly to
    construct an Operator.

    Parameters
    ----------
    lhs : expr-like
        The left-hand side.
    rhs : expr-like, optional
        The right-hand side. Defaults to 0.
    subdomain : SubDomain, optional
        To restrict the evalaution of the relation to a particular sub-region in the
        computational domain.

    Examples
    --------
    `Gt` may be used to express a relation.

    >>> from devito import Grid, Function, ConditionalDimension, Eq, Operator, Gt
    >>> grid = Grid(shape=(8, 8))
    >>> g = Function(name='g', grid=grid)
    >>> Gt(g, 1)
    g(x, y) > 1
    """

    def __new__(cls, lhs, rhs=0, subdomain=None, **kwargs):
        kwargs.update({'evaluate': False})
        obj = sympy.Gt.__new__(cls, lhs, rhs, **kwargs)
        obj._subdomain = subdomain

        return obj


class Ne(AbstractRel, sympy.Ne):
    """
    A not-equal ("!=") relation between two objects (see Notes below), the left-hand side
    and the right-hand side. It can be used to build conditionals but not directly to
    construct an Operator.

    Parameters
    ----------
    lhs : expr-like
        The left-hand side.
    rhs : expr-like, optional
        The right-hand side. Defaults to 0.
    subdomain : SubDomain, optional
        To restrict the evalaution of the relation to a particular sub-region in the
        computational domain.

    Examples
    --------
    `Ne` may be used to express a relation.

    >>> from devito import Grid, Function, ConditionalDimension, Eq, Operator, Ne
    >>> grid = Grid(shape=(8, 8))
    >>> g = Function(name='g', grid=grid)
    >>> Ne(g, 0)
    Ne(g(x, y), 0)

    Notes
    -----
    This class is not the same as the != operator. The != operator tests for exact
    structural equality between two expressions; this class compares expressions
    mathematically. Source: https://docs.sympy.org/latest/modules/core.html
    """

    is_Relational = False

    def __new__(cls, lhs, rhs=0, subdomain=None, **kwargs):
        kwargs.update({'evaluate': False})
        obj = sympy.Ne.__new__(cls, lhs, rhs, **kwargs)
        obj._subdomain = subdomain

        return obj


ops = {Ge: Lt, Gt: Le, Le: Gt, Lt: Ge}
rev = {Ge: Le, Gt: Lt, Lt: Gt, Le: Ge}

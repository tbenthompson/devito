from collections import Counter

from devito.ir.clusters import Queue
from devito.ir.support import (SEQUENTIAL, SKEWABLE, TILABLE, Interval, IntervalGroup,
                               IterationSpace)
from devito.symbolics import uxreplace
from devito.types import IncrDimension, RIncrDimension
from devito.symbolics import xreplace_indices, evalmax
from devito.tools import as_list, as_tuple
from devito.passes.clusters.utils import level

__all__ = ['blocking', 'skewing']


def blocking(clusters, options):
    """
    Loop blocking to improve data locality.

    Parameters
    ----------
    clusters : tuple of Clusters
        Input Clusters, subject of the optimization pass.
    options : dict
        The optimization options.
        * `blockinner` (boolean, False): enable/disable loop blocking along the
           innermost loop.
        * `blocklevels` (int, 1): 1 => classic loop blocking; 2 for two-level
           hierarchical blocking.

    Example
    -------
        * A typical use case, e.g.
          .. code-block::
                            Classical   +blockinner  2-level Hierarchical
            for x            for xb        for xb         for xbb
              for y    -->    for yb        for yb         for ybb
                for z          for x         for zb         for xb
                                for y         for x          for yb
                                 for z         for y          for x
                                                for z          for y
                                                                for z
    """
    processed = preprocess(clusters, options)

    if options['blocklevels'] > 0:
        processed = Blocking(options).process(processed)

    return processed


class Blocking(Queue):

    template = "%s%d_blk%s"

    def __init__(self, options):
        self.inner = bool(options['blockinner'])
        self.levels = options['blocklevels']

        self.nblocked = Counter()

        super(Blocking, self).__init__()

    def _make_key_hook(self, cluster, level):
        return (tuple(cluster.guards.get(i.dim) for i in cluster.itintervals[:level]),)

    def _process_fdta(self, clusters, level, prefix=None):
        # Truncate recursion in case of TILABLE, non-perfect sub-nests, as
        # it's an unsupported case
        if prefix:
            d = prefix[-1].dim
            test0 = any(TILABLE in c.properties[d] for c in clusters)
            test1 = len({c.itintervals[:level] for c in clusters}) > 1
            if test0 and test1:
                return self.callback(clusters, prefix)

        return super(Blocking, self)._process_fdta(clusters, level, prefix)

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        d = prefix[-1].dim

        name = self.template % (d.name, self.nblocked[d], '%d')
        block_dims = create_block_dims(name, d, self.levels)

        processed = []
        for c in clusters:
            if TILABLE in c.properties[d]:
                ispace = decompose(c.ispace, d, block_dims)

                # Use the innermost IncrDimension in place of `d`
                exprs = [uxreplace(e, {d: block_dims[-1]}) for e in c.exprs]

                # The new Cluster properties
                # TILABLE property is dropped after the blocking.
                properties = dict(c.properties)
                properties.pop(d)
                properties.update({bd: c.properties[d] - {TILABLE} for bd in block_dims})

                processed.append(c.rebuild(exprs=exprs, ispace=ispace,
                                           properties=properties))
            else:
                processed.append(c)

        # Make sure to use unique IncrDimensions
        self.nblocked[d] += int(any(TILABLE in c.properties[d] for c in clusters))

        return processed


def preprocess(clusters, options):
    # Preprocess: heuristic: drop TILABLE from innermost Dimensions to
    # maximize vectorization
    inner = bool(options['blockinner'])
    processed = []
    for c in clusters:
        ntilable = len([i for i in c.properties.values() if TILABLE in i])
        ntilable -= int(not inner)
        if ntilable <= 1:
            properties = {k: v - {TILABLE, SKEWABLE} for k, v in c.properties.items()}
            processed.append(c.rebuild(properties=properties))
        elif not inner:
            d = c.itintervals[-1].dim
            properties = dict(c.properties)
            properties[d] = properties[d] - {TILABLE, SKEWABLE}
            processed.append(c.rebuild(properties=properties))
        else:
            processed.append(c)

    return processed


def create_block_dims(name, d, levels):
    """
    Create the block Dimensions (in total `self.levels` Dimensions)
    """
    bd = RIncrDimension(name % 0, d, d.symbolic_min, d.symbolic_max)
    size = bd.step
    block_dims = [bd]

    for i in range(1, levels):
        bd = RIncrDimension(name % i, bd, bd, bd + bd.step - 1, size=size)
        block_dims.append(bd)

    bd = IncrDimension(d.name, bd, bd, bd + bd.step - 1, 1, size=size)
    block_dims.append(bd)

    return block_dims


def decompose(ispace, d, block_dims):
    """
    Create a new IterationSpace in which the `d` Interval is decomposed
    into a hierarchy of Intervals over ``block_dims``.
    """
    # Create the new Intervals
    intervals = []
    for i in ispace:
        if i.dim is d:
            intervals.append(i.switch(block_dims[0]))
            intervals.extend([i.switch(bd).zero() for bd in block_dims[1:]])
        else:
            intervals.append(i)

    # Create the relations.
    # Example: consider the relation `(t, x, y)` and assume we decompose `x` over
    # `xbb, xb, xi`; then we decompose the relation as two relations, `(t, xbb, y)`
    # and `(xbb, xb, xi)`
    relations = [block_dims]
    for r in ispace.intervals.relations:
        relations.append([block_dims[0] if i is d else i for i in r])

    # Add more relations
    for n, i in enumerate(ispace):
        if i.dim is d:
            continue
        elif i.dim.is_Incr:
            # Make sure IncrDimensions on the same level stick next to each other.
            # For example, we want `(t, xbb, ybb, xb, yb, x, y)`, rather than say
            # `(t, xbb, xb, x, ybb, ...)`
            for bd in block_dims:
                if i.dim._depth >= bd._depth:
                    relations.append([bd, i.dim])
                else:
                    relations.append([i.dim, bd])
        elif n > ispace.intervals.index(d):
            # The non-Incr subsequent Dimensions must follow the block Dimensions
            for bd in block_dims:
                relations.append([bd, i.dim])
        else:
            # All other Dimensions must precede the block Dimensions
            for bd in block_dims:
                relations.append([i.dim, bd])

    intervals = IntervalGroup(intervals, relations=relations)

    sub_iterators = dict(ispace.sub_iterators)
    sub_iterators.pop(d, None)
    sub_iterators.update({block_dims[-1]: ispace.sub_iterators.get(d, [])})
    sub_iterators.update({bd: () for bd in block_dims[:-1]})

    directions = dict(ispace.directions)
    directions.pop(d)
    directions.update({bd: ispace.directions[d] for bd in block_dims})

    return IterationSpace(intervals, sub_iterators, directions)


def skewing(clusters, options):
    """
    This pass helps to skew accesses and loop bounds as well as perform loop interchange
    towards wavefront temporal blocking
    Parameters
    ----------
    clusters : tuple of Clusters
        Input Clusters, subject of the optimization pass.
    options : dict
        The optimization options.
        * `skewinner` (boolean, False): enable/disable loop skewing along the
           innermost loop.
    """
    processed = clusters
    if options['blocktime']:
        processed = TBlocking(options).process(processed)

    processed = Skewing(options).process(processed)
    processed = RelaxSkewed(options).process(processed)

    return processed


class Skewing(Queue):

    """
    Construct a new sequence of clusters with skewed expressions and iteration spaces.

    Notes
    -----
    This transformation is applying loop skewing to derive the
    wavefront method of execution of nested loops. Loop skewing is
    a simple transformation of loop bounds and is combined with loop
    interchanging to generate the wavefront [1]_.

    .. [1] Wolfe, Michael. "Loops skewing: The wavefront method revisited."
    International Journal of Parallel Programming 15.4 (1986): 279-293.

    Examples:

    .. code-block:: python

        for i = 2, n-1
            for j = 2, m-1
                a[i,j] = (a[a-1,j] + a[i,j-1] + a[i+1,j] + a[i,j+1]) / 4

    to

    .. code-block:: python

        for i = 2, n-1
            for j = 2+i, m-1+i
                a[i,j-i] = (a[a-1,j-i] + a[i,j-1-i] + a[i+1,j-i] + a[i,j+1-i]) / 4

    """

    template = "%s%d_blk%s"

    def __init__(self, options):
        self.skewinner = bool(options['blockinner'])
        self.levels = options['blocklevels']

        super(Skewing, self).__init__()

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        d = prefix[-1].dim
        processed = []
        for c in clusters:
            if SKEWABLE not in c.properties[d]:
                return clusters

            if d is c.ispace[-1].dim and not self.skewinner:
                return clusters

            skew_dims = [i.dim for i in c.ispace if SEQUENTIAL in c.properties[i.dim]]
            if len(skew_dims) > 2:
                return clusters
            skew_dim = skew_dims[-1]

            # Since we are here, prefix is skewable and nested under a
            # SEQUENTIAL loop.

            skewlevel = 1
            intervals = []
            for i in c.ispace:
                if i.dim is d:
                    # Skew at skewlevel + 1 if time is blocked
                    cond1 = len(skew_dims) == 2 and level(d) == skewlevel + 1
                    # Skew at level <=1 if time is not blocked
                    cond2 = len(skew_dims) == 1 and level(d) <= skewlevel
                    if cond1 or cond2:
                        intervals.append(Interval(d, skew_dim, skew_dim))
                    else:
                        intervals.append(i)
                else:
                    intervals.append(i)

            intervals = IntervalGroup(intervals, relations=c.ispace.relations)
            ispace = IterationSpace(intervals, c.ispace.sub_iterators,
                                    c.ispace.directions)

            exprs = xreplace_indices(c.exprs, {d: d - skew_dim})
            processed.append(c.rebuild(exprs=exprs, ispace=ispace,
                                       properties=c.properties))

        return processed


class TBlocking(Queue):

    template = "%s%d_blk%s"

    def __init__(self, options):
        self.nblocked = Counter()
        super(TBlocking, self).__init__()

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        d = prefix[-1].dim

        if not d.is_Time:
            return clusters

        name = self.template % (d.name, self.nblocked[d], '%d')
        block_dims = create_block_dims(name, d, 1)

        processed = []
        for c in clusters:
            if d.is_Time:
                ispace = decompose(c.ispace, d, block_dims)

                # Use the innermost IncrDimension in place of `d`
                exprs = [uxreplace(e, {d: block_dims[-1]}) for e in c.exprs]

                # The new Cluster properties
                properties = dict(c.properties)
                properties.pop(d)
                properties.update({bd: c.properties[d] for bd in block_dims})
                processed.append(c.rebuild(exprs=exprs, ispace=ispace,
                                           properties=properties))
            else:
                processed.append(c)
        return processed


class RelaxSkewed(Queue):

    def __init__(self, options):
        self.nblocked = Counter()
        self.levels = options['blocklevels']
        super(RelaxSkewed, self).__init__()

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        d = prefix[-1].dim

        processed = []

        for c in clusters:
            family = []
            for i in c.ispace:
                family = [j for j in c.ispace if j.dim.root is d.root]

        if not d.is_Incr:
            return clusters

        # Rule out time dim
        if d.is_Time:
            return clusters

        skew_dims = [i.dim for i in c.ispace if SEQUENTIAL in c.properties[i.dim]]
        if len(skew_dims) == 1:
            return clusters

        skew_dim = skew_dims[-1]
        family_dims = [j.dim for j in family]

        if d is not family_dims[0]:
            return clusters

        print(family)
        print(level(d))

        intervals = []
        mapper = {}
        for c in clusters:
            for i in c.ispace:
                if i.dim in family_dims:
                    if level(i.dim) == 1:
                        offset = skew_dim.root.symbolic_max - skew_dim.root.symbolic_min
                        rmax = i.dim.symbolic_max + offset
                        sd = i.dim.func(rmax=rmax)

                        print(sd)
                        intervals.append(Interval(sd, i.lower, i.upper))
                        mapper.update({i.dim: sd})
                    elif level(i.dim) == 2:
                        rmin = evalmax(i.dim.symbolic_min,
                                       i.dim.root.symbolic_min + skew_dim)
                        rmax = i.dim.symbolic_rmax.xreplace({i.dim.root.symbolic_max:
                                                            i.dim.root.symbolic_max +
                                                            skew_dim})

                        sd2 = i.dim.func(parent=sd, _rmin=rmin, _rmax=rmax)
                        intervals.append(Interval(sd2, i.lower, i.upper))
                        mapper.update({i.dim: sd2})
                    else:
                        intervals.append(i)
                else:
                    intervals.append(i)

            print(mapper)
            relations = []
            for r in c.ispace.relations:
                if any(f for f in family_dims) in r and mapper:
                    rl = as_list(r)
                    newr = [j.xreplace(mapper) for j in rl]
                    relations.append(as_tuple(newr))
                else:
                    relations.append(r)

            assert len(relations) == len(c.ispace.relations)
            intervals = IntervalGroup(intervals, relations=relations)

            sub_iterators = dict(c.ispace.sub_iterators)
            for f in family_dims:
                sub_iterators.pop(f, None)
                sub_iterators.update({mapper[f]: c.ispace.sub_iterators.get(f, [])})

            directions = dict(c.ispace.directions)
            for f in family_dims:
                directions.pop(f)
                directions.update({mapper[f]: c.ispace.directions[f]})

            ispace = IterationSpace(intervals, sub_iterators, directions)

            exprs = c.exprs
            for f in family_dims:
                exprs = xreplace_indices(c.exprs, {f: mapper[f]})

            properties = dict(c.properties)

            for f in family_dims:
                properties.pop(f)
                properties.update({mapper[f]: c.properties[f]})

            processed.append(c.rebuild(exprs=exprs, ispace=ispace,
                                       properties=properties))

        return processed

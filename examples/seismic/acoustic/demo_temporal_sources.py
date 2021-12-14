import numpy as np

from matplotlib.pyplot import pause # noqa
import matplotlib.pyplot as plt
import argparse

from devito.logger import info
from devito import TimeFunction, Function, Dimension, Eq, Inc, solve
from devito import Operator, norm, configuration
from examples.seismic import RickerSource, TimeAxis
from examples.seismic import Model
import sys
np.set_printoptions(threshold=sys.maxsize)  # pdb print full size

from devito.types.basic import Scalar, Symbol # noqa
from mpl_toolkits.mplot3d import Axes3D # noqa


def plot3d(data, model):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    z, x, y = data.nonzero()
    ax.scatter(x, y, z, zdir='y', c='red', s=20, marker='.')
    ax.set_xlim(model.spacing[0], data.shape[0]-model.spacing[0])
    ax.set_ylim(model.spacing[1], data.shape[1]-model.spacing[1])
    ax.set_zlim(model.spacing[2], data.shape[2]-model.spacing[2])
    plt.savefig("sources_demo.pdf")


parser = argparse.ArgumentParser(description='Process arguments.')

parser.add_argument("-d", "--shape", default=(11, 11, 11), type=int, nargs="+",
                    help="Number of grid points along each axis")
parser.add_argument("-so", "--space_order", default=4,
                    type=int, help="Space order of the simulation")
parser.add_argument("-tn", "--tn", default=40,
                    type=float, help="Simulation time in millisecond")
parser.add_argument("-bs", "--bsizes", default=(32, 32, 8, 8), type=int, nargs="+",
                    help="Block and tile sizes")
args = parser.parse_args()


nx, ny, nz = args.shape
# Define a physical size
shape = (nx, ny, nz)  # Number of grid point (nx, nz)
spacing = (10., 10., 10)  # Grid spacing in m. The domain size is now 1km by 1km
origin = (0., 0., 0.)
so = args.space_order
# Initialize v field
v = np.empty(shape, dtype=np.float32)
v[:, :, :int(nz/2)] = 2
v[:, :, int(nz/2):] = 1

# Construct model
model = Model(vp=v, origin=origin, shape=shape, spacing=spacing, space_order=so,
              nbl=10, bcs="damp")

# plt.imshow(model.vp.data[10, :, :]) ; pause(1)

t0 = 0  # Simulation starts a t=0
tn = args.tn  # Simulation last 1 second (1000 ms)
dt = model.critical_dt  # Time step from model grid spacing
time_range = TimeAxis(start=t0, stop=tn, step=dt)
f0 = 0.010  # Source peak frequency is 10Hz (0.010 kHz)
src = RickerSource(name='src', grid=model.grid, f0=f0,
                   npoint=1, time_range=time_range)


# First, position source centrally in all dimensions, then set depth

stx = 0.1
ste = 0.9
stepx = (ste-stx)/int(np.sqrt(src.npoint))

src.coordinates.data[:, :2] = np.array(np.meshgrid(np.arange(stx, ste, stepx), np.arange(stx, ste, stepx))).T.reshape(-1,2)*np.array(model.domain_size[:1])

src.coordinates.data[:, -1] = 20  # Depth is 20m

#src.coordinates.data[0, :] = np.array(model.domain_size) * .5
#src.coordinates.data[0, -1] = 20  # Depth is 20m
#src.coordinates.data[1, :] = np.array(model.domain_size) * .5
#src.coordinates.data[1, -1] = 20  # Depth is 20m
#src.coordinates.data[2, :] = np.array(model.domain_size) * .5
#src.coordinates.data[2, -1] = 20  # Depth is 20m

#src.show()
#pause(1)

# f : perform source injection on an empty grid
f = TimeFunction(name="f", grid=model.grid, space_order=so, time_order=2)
src_f = src.inject(field=f.forward, expr=src * dt**2 / model.m)
# op_f = Operator([src_f], opt=('advanced', {'openmp': True}))
op_f = Operator([src_f])

op_f.apply(time=time_range.num-1)
normf = norm(f)
print("==========")
print(normf)
print("===========")

# uref : reference solution
uref = TimeFunction(name="uref", grid=model.grid, space_order=so, time_order=2)
src_term_ref = src.inject(field=uref.forward, expr=src * dt**2 / model.m)
pde_ref = model.m * uref.dt2 - uref.laplace + model.damp * uref.dt
stencil_ref = Eq(uref.forward, solve(pde_ref, uref.forward))

#Get the nonzero indices
nzinds = np.nonzero(f.data[0])  # nzinds is a tuple
assert len(nzinds) == len(shape)

shape = model.grid.shape
x, y, z = model.grid.dimensions
time = model.grid.time_dim
source_mask = Function(name='source_mask', shape=shape, dimensions=(x, y, z), space_order=0, dtype=np.float32)
source_id = Function(name='source_id', shape=shape, dimensions=(x, y, z), space_order=0, dtype=np.int32)
info("source_id data indexes start from 1 not 0 !!!")

# source_id.data[nzinds[0], nzinds[1], nzinds[2]] = tuple(np.arange(1, len(nzinds[0])+1))
source_id.data[nzinds[0], nzinds[1], nzinds[2]] = tuple(np.arange(len(nzinds[0])))

source_mask.data[nzinds[0], nzinds[1], nzinds[2]] = 1
# plot3d(source_mask.data, model)

info("Number of unique affected points is: %d", len(nzinds[0])+1)

# Assert that first and last index are as expected
assert(source_id.data[nzinds[0][0], nzinds[1][0], nzinds[2][0]] == 0)
assert(source_id.data[nzinds[0][-1], nzinds[1][-1], nzinds[2][-1]] == len(nzinds[0])-1)
assert(source_id.data[nzinds[0][len(nzinds[0])-1], nzinds[1][len(nzinds[0])-1],
       nzinds[2][len(nzinds[0])-1]] == len(nzinds[0])-1)

assert(np.all(np.nonzero(source_id.data)) == np.all(np.nonzero(source_mask.data)))
assert(np.all(np.nonzero(source_id.data)) == np.all(np.nonzero(f.data[0])))

info("-At this point source_mask and source_id have been popoulated correctly-")

nnz_shape = (model.grid.shape[0], model.grid.shape[1])  # Change only 3rd dim

nnz_sp_source_mask = Function(name='nnz_sp_source_mask', shape=(list(nnz_shape)), dimensions=(x, y), space_order=0, dtype=np.int32)


nnz_sp_source_mask.data[:, :] = source_mask.data[:, :, :].sum(2)
inds = np.where(source_mask.data == 1.)

maxz = len(np.unique(inds[2]))
sparse_shape = (model.grid.shape[0], model.grid.shape[1], maxz)  # Change only 3rd dim

assert(len(nnz_sp_source_mask.dimensions) == 2)

# Note:sparse_source_id is not needed as long as sparse info is kept in mask
# sp_source_id.data[inds[0],inds[1],:] = inds[2][:maxz]

id_dim = Dimension(name='id_dim')
b_dim = Dimension(name='b_dim')

save_src = TimeFunction(name='save_src', shape=(src.shape[0],
                        nzinds[1].shape[0]), dimensions=(src.dimensions[0], id_dim))

save_src_term = src.inject(field=save_src[src.dimensions[0], source_id], expr=src * dt**2 / model.m)

op1 = Operator([save_src_term])
op1.apply(time=time_range.num-1, dt=dt)

usol = TimeFunction(name="usol", grid=model.grid, space_order=so, time_order=2)

sp_zi = Dimension(name='sp_zi')

# import pdb; pdb.set_trace()

sp_source_mask = Function(name='sp_source_mask', shape=(list(sparse_shape)), dimensions=(x, y, sp_zi), space_order=0, dtype=np.int32)

# Now holds IDs
sp_source_mask.data[inds[0], inds[1], :] = tuple(inds[2][:len(np.unique(inds[2]))])

assert(np.count_nonzero(sp_source_mask.data) == len(nzinds[0]))
assert(len(sp_source_mask.dimensions) == 3)

t = model.grid.stepping_dim

zind = Scalar(name='zind', dtype=np.int32)
xb_size = Scalar(name='xb_size', dtype=np.int32)
yb_size = Scalar(name='yb_size', dtype=np.int32)
x0_blk0_size = Scalar(name='x0_blk0_size', dtype=np.int32)
y0_blk0_size = Scalar(name='y0_blk0_size', dtype=np.int32)

eq0 = Eq(sp_zi.symbolic_max, nnz_sp_source_mask[x, y] - 1, implicit_dims=(time, x, y))
eq1 = Eq(zind, sp_source_mask[x, y, sp_zi], implicit_dims=(time, x, y, sp_zi))

myexpr = source_mask[x, y, zind] * save_src[time, source_id[x, y, zind]]

eq2 = Inc(usol.forward[t+1, x, y, zind], myexpr, implicit_dims=(time, x, y, sp_zi))

pde_2 = model.m * usol.dt2 - usol.laplace + model.damp * usol.dt
stencil_2 = Eq(usol.forward, solve(pde_2, usol.forward))


block_sizes = Function(name='block_sizes', shape=(4, ), dimensions=(b_dim,), space_order=0, dtype=np.int32)

# import pdb; pdb.set_trace()
block_sizes.data[:] = args.bsizes

# import pdb; pdb.set_trace()

eqxb = Eq(xb_size, block_sizes[0])
eqyb = Eq(yb_size, block_sizes[1])
eqxb2 = Eq(x0_blk0_size, block_sizes[2])
eqyb2 = Eq(y0_blk0_size, block_sizes[3])

# import pdb; pdb.set_trace()
# plot3d(source_mask.data, model)

opref = Operator([stencil_ref, src_term_ref], opt=('advanced', {'openmp': True}))
print("===Space blocking==")
configuration['autotuning']='aggressive'
opref.apply(time=time_range.num-2, dt=model.critical_dt)
configuration['autotuning']='off'
print("===========")

normuref = norm(uref)
print("==========")
print(normuref)
print("===========")


print("-----")
op2 = Operator([eqxb, eqyb, eqxb2, eqyb2, stencil_2, eq0, eq1, eq2], eqxb=32, opt=('advanced'))
# print(op2.ccode)
print("===Temporal blocking======================================")
op2.apply(time=time_range.num-1, dt=model.critical_dt)
print("===========")

normusol = norm(usol)
print("===========")
print(normusol)
print("===========")


print("Norm(f):", normf)
print("Norm(usol):", normusol)
print("Norm(uref):", normuref)

# save_src.data[0, source_id.data[14, 14, 11]]
# save_src.data[0 ,source_id.data[14, 14, sp_source_mask.data[14, 14, 0]]]

#plt.imshow(uref.data[2, int(nx/2) ,:, :]); pause(1)
#plt.imshow(usol.data[2, int(nx/2) ,:, :]); pause(1)


# Uncomment to plot a slice of the field
#plt.imshow(usol.data[2, int(nx/2) ,:, :]); pause(1)

assert np.isclose(normuref, normusol, atol=1e-06)

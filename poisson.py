import fenics
import numpy


# Create mesh and define function space
mesh = fenics.UnitSquareMesh(8, 8)

V = fenics.FunctionSpace(mesh, 'P', 2)


# Define boundary condition
u_D = fenics.Expression('1 + pow(x[0], 2) + 2*pow(x[1], 2)', degree=2)

def boundary(x, on_boundary):

    return on_boundary

    
bc = fenics.DirichletBC(V, u_D, boundary)


# Define variational problem
u = fenics.TrialFunction(V)

v = fenics.TestFunction(V)

f = fenics.Constant(-6.0)

dot, grad, = fenics.dot, fenics.grad

a = dot(grad(u), grad(v))*fenics.dx

L = f*v*fenics.dx


# Compute solution
u = fenics.Function(V)

fenics.solve(a == L, u, bc)


# Save solution to file in VTK format
vtkfile = fenics.File('poisson/solution.pvd')

vtkfile << u


# Compute error in L2 norm
error_L2 = fenics.errornorm(u_D, u, 'L2')


# Compute maximum error at vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)

vertex_values_u = u.compute_vertex_values(mesh)

error_max = numpy.max(numpy.abs(vertex_values_u_D - vertex_values_u))


# Print errors
print('error_L2 =', error_L2)

print('error_max =', error_max)

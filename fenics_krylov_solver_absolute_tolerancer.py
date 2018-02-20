""" This demonstrates how to control the absolute error tolerance of a FEniCS Krylov solver."""
import fenics


def boundary(x, on_boundary):
    """ Mark the boundary for applying BC's."""
    return on_boundary
        
        
def solve_poisson_problem(krylov_absolute_tolerance = 1.e-8):
    """ Solve the Poisson problem."""
    
    
    # Set the solution space.
    mesh = fenics.UnitSquareMesh(4, 4)
    
    V = fenics.FunctionSpace(mesh, 'P', 2)  # Degree 2 exactly recovers a quadratic solution.
    
    
    # Set a manufactured solution, with corresponding source term and boundary conditions.
    exact_solution = fenics.Expression("1 + x[0]*x[0] + 2*x[1]*x[1]", element = V.ufl_element())
    
    f = fenics.Constant(-6.0)
    
    bcs = fenics.DirichletBC(V, exact_solution, boundary)
    
    
    # Set the variational form.
    u = fenics.TrialFunction(V)
    
    v = fenics.TestFunction(V)
    
    dot, grad = fenics.dot, fenics.grad
    
    a = dot(grad(v), grad(u))*fenics.dx
    
    L = v*f*fenics.dx
    
    
    # Solve.
    solution = fenics.Function(V)
    
    problem = fenics.LinearVariationalProblem(a = a, L = L, u = solution, bcs = bcs)
    
    solver = fenics.LinearVariationalSolver(problem = problem)
    
    solver.parameters["linear_solver"] = "cg"
    
    solver.parameters["krylov_solver"]["absolute_tolerance"] = krylov_absolute_tolerance
    
    solver.parameters["krylov_solver"]["relative_tolerance"] = 1.e-16
    
    solver.parameters["krylov_solver"]["maximum_iterations"] = 1000
    
    solver.parameters["krylov_solver"]["monitor_convergence"] = True
    
    fenics.set_log_level(fenics.PROGRESS)
    
    solver.solve()
    
    
    # Verify the solution.
    L2_error = fenics.errornorm(exact_solution, solution, 'L2')
    
    print("L2_error = " + str(L2_error))
    
    assert(L2_error < krylov_absolute_tolerance)
    
    
if __name__=='__main__':
    """Run for a variety of absolute tolerances."""
    solve_poisson_problem(krylov_absolute_tolerance = 1.e-2)
    
    solve_poisson_problem(krylov_absolute_tolerance = 1.e-3)
    
    solve_poisson_problem(krylov_absolute_tolerance = 1.e-4)
    
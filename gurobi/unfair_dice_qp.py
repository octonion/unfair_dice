import gurobipy as gp
from gurobipy import GRB

def solve_polynomial_factorization(m, n):
    """
    Builds and solves a continuous non-convex quadratically constrained program
    to find factors of a {0,1}-polynomial.

    The model seeks to find two monic polynomials, P(x) and Q(x), such that:
    1. R(x) = P(x) * Q(x)
    2. All coefficients for P(x), Q(x), and R(x) are continuous variables
       between 0 and 1.
    3. The objective function encourages P and Q coefficients to be fractional
       while penalizing R coefficients for being fractional, effectively
       pushing them towards {0, 1}.

    Args:
        m (int): The degree of the polynomial P(x).
        n (int): The degree of the polynomial Q(x).

    Returns:
        None. Prints the solution if found.
    """
    print(f"\n--- Solving for P(x) of degree {m} and Q(x) of degree {n} ---\n")

    model = gp.Model("PolynomialFactorization")

    # --- 1. DEFINE VARIABLES ---
    # All variables are now continuous between 0 and 1.
    p = model.addVars(m, lb=0.0, ub=1.0, name="p")
    q = model.addVars(n, lb=0.0, ub=1.0, name="q")
    r = model.addVars(m + n, lb=0.0, ub=1.0, name="r") # Changed from BINARY

    # --- SET INITIAL SOLUTION (WARM START) ---
    # We provide the trivial solution P(x)=x^m, Q(x)=x^n, R(x)=x^{m+n}
    for i in range(m):
        p[i].Start = 0.0
    for j in range(n):
        q[j].Start = 0.0
    for k in range(m + n):
        r[k].Start = 0.0

    # To simplify constraint generation, we create dictionaries that include the
    # fixed monic coefficients (p_m = 1, q_n = 1).
    p_coeffs = {i: p[i] for i in range(m)}
    p_coeffs[m] = 1.0

    q_coeffs = {j: q[j] for j in range(n)}
    q_coeffs[n] = 1.0

    # --- 2. SET CONSTRAINTS ---
    # r_k = sum_{i+j=k} p_i * q_j
    for k in range(m + n):
        convolution_sum = gp.quicksum(
            p_coeffs.get(i, 0) * q_coeffs.get(k - i, 0)
            for i in range(k + 1)
        )
        model.addConstr(r[k] == convolution_sum, name=f"r_constraint_{k}")

    # --- 3. SET THE OBJECTIVE FUNCTION ---
    # The new objective encourages p and q to be fractional (away from 0 and 1)
    # while penalizing r for being fractional, pushing it towards 0 or 1.
    p_obj = gp.quicksum(p[i] - p[i]*p[i] for i in range(m))
    q_obj = gp.quicksum(q[j] - q[j]*q[j] for j in range(n))
    r_penalty = gp.quicksum(r[k] - r[k]*r[k] for k in range(m + n))
    
    # The penalty for r is weighted by the number of coefficients in r.
    model.setObjective(p_obj + q_obj - (m + n) * r_penalty, GRB.MAXIMIZE)

    # --- 4. CONFIGURE AND SOLVE ---
    #model.setParam('NonConvex', 2)
    model.setParam('TimeLimit', 1200)
    # model.setParam('OutputFlag', 0) # Keep log visible for now

    model.optimize()

    # --- 5. DISPLAY RESULTS ---
    # We check if a solution of any kind was found
    if model.SolCount > 0:
        print("\n--- Feasible Solution Found ---")
        print(f"Solver Status: {model.Status} (2=OPTIMAL, 9=TIMELIMIT, 13=SUBOPTIMAL)")
        print(f"Objective Value: {model.ObjVal:.6f}")
        
        # Construct polynomial strings for display
        # For R(x), we round the continuous variables to the nearest integer for display
        p_terms = [f"x^{m}"] + [f"{p[i].X:.6f}*x^{i}" for i in range(m - 1, -1, -1) if abs(p[i].X) > 1e-6]
        q_terms = [f"x^{n}"] + [f"{q[j].X:.6f}*x^{j}" for j in range(n - 1, -1, -1) if abs(q[j].X) > 1e-6]
        r_terms = [f"x^{m+n}"] + [f"x^{k}" for k in range(m + n - 1, -1, -1) if r[k].X > 0.5]

        def format_poly(terms):
            return " + ".join(terms).replace("+ -", "- ")

        print(f"\nP(x) = {format_poly(p_terms)}")
        print(f"Q(x) = {format_poly(q_terms)}")
        print(f"R(x) (rounded) = {format_poly(r_terms)}")

        # Optionally, show the actual continuous values for r
        print("\nActual continuous coefficients for R(x):")
        for k in range(m + n):
            if r[k].X > 1e-6:
                print(f"  r_{k}: {r[k].X:.6f}")

    elif model.Status == GRB.INFEASIBLE:
        print("\n--- Model is Infeasible ---")
    else:
        print(f"\n--- No Solution Found (Status: {model.Status}) ---")


if __name__ == '__main__':
    m_degree = 16
    n_degree = 16
    solve_polynomial_factorization(m_degree, n_degree)

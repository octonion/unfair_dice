import gurobipy as gp
from gurobipy import GRB

def solve_polynomial_factorization(m, n):
    """
    Builds and solves a MIQP model to find factors of a {0,1}-polynomial.

    The model seeks to find two monic polynomials, P(x) and Q(x), such that:
    1. R(x) = P(x) * Q(x)
    2. P(x) and Q(x) have non-negative coefficients.
    3. R(x) has coefficients in {0, 1}.
    4. The solution is non-trivial, meaning at least one coefficient in P or Q
       is not an integer (i.e., not 0 or 1).

    Args:
        m (int): The degree of the polynomial P(x).
        n (int): The degree of the polynomial Q(x).

    Returns:
        None. Prints the solution if found and is non-trivial.
    """
    print(f"\n--- Solving for P(x) of degree {m} and Q(x) of degree {n} ---\n")

    model = gp.Model("PolynomialFactorization")

    # --- 1. DEFINE VARIABLES ---
    # Coefficients for P(x). p_m is 1 (monic), so we need m variables for p_0, ..., p_{m-1}.
    # These are continuous and non-negative.
    p = model.addVars(m, lb=0.0, ub=1.0, name="p")

    # Coefficients for Q(x). q_n is 1 (monic), so we need n variables for q_0, ..., q_{n-1}.
    # These are continuous and non-negative.
    q = model.addVars(n, lb=0.0, ub=1.0, name="q")

    # Coefficients for R(x) = P(x)Q(x). The degree of R(x) is m+n.
    # The leading coefficient r_{m+n} is 1. We need m+n variables for r_0, ..., r_{m+n-1}.
    # These are constrained to be binary {0, 1}.
    r = model.addVars(m + n, vtype=GRB.BINARY, name="r")

    # --- SET INITIAL SOLUTION (WARM START) ---
    # We provide the trivial solution P(x)=x^m, Q(x)=x^n, R(x)=x^{m+n}
    # as a starting point for the solver. This means all variable
    # coefficients are initially zero. This is done by setting the 'Start' attribute.
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
    # The core of the model is relating the coefficients of R(x) to the product
    # of P(x) and Q(x). The coefficient r_k is the convolution of p and q.
    # r_k = sum_{i+j=k} p_i * q_j
    # These are quadratic constraints. Since p and q are variables, the
    # constraints are non-convex. We must instruct Gurobi to handle this.
    for k in range(m + n):
        # The expression for the k-th coefficient of R(x)
        convolution_sum = gp.quicksum(
            p_coeffs.get(i, 0) * q_coeffs.get(k - i, 0)
            for i in range(k + 1)
        )
        model.addConstr(r[k] == convolution_sum, name=f"r_constraint_{k}")

    # --- 3. SET THE OBJECTIVE FUNCTION ---
    # The goal is to find a solution where at least one coefficient in P or Q
    # is NOT in {0, 1}. A simple feasible solution is P(x)=x^m, Q(x)=x^n, where
    # all coefficients are 0 or 1. We need to guide the solver away from this.
    #
    # We can measure the "non-integer-ness" of a variable `v` using the
    # quadratic expression `v - v*v`.
    # - If v = 0, then v - v*v = 0.
    # - If v = 1, then v - v*v = 0.
    # - The expression is maximized at v = 0.5, where it equals 0.25.
    #
    # By maximizing the sum of these expressions for all coefficients of P and Q,
    # we encourage the solver to find solutions where the coefficients are as
    # far away from 0 and 1 as possible. This directly promotes the non-trivial
    # solutions we are looking for.
    objective = gp.quicksum(p[i] - p[i]*p[i] for i in range(m)) + \
                gp.quicksum(q[j] - q[j]*q[j] for j in range(n))

    model.setObjective(objective, GRB.MAXIMIZE)

    # --- 4. CONFIGURE AND SOLVE ---
    # Suppress Gurobi's console output during the solve.
    #model.setParam('OutputFlag', 0)
    # This is the critical parameter setting.
    # Value 2 tells Gurobi to use its non-convex quadratic optimization algorithms.
    model.setParam('NonConvex', 2)

    # Optional: Add a time limit for complex problems
    # model.setParam('TimeLimit', 12000)

    model.optimize()

    # --- 5. DISPLAY RESULTS ---
    if model.Status == GRB.OPTIMAL or model.Status == GRB.SOLUTION_LIMIT:
        # Only display the solution if it's non-trivial (objective > 0).
        # We use a small tolerance to account for floating point inaccuracies.
        if model.ObjVal > 1e-4:
            print("\n--- Non-Trivial Solution Found ---")
            print(f"Objective Value (measure of non-integer coefficients): {model.ObjVal:.4f}")

            # Construct polynomial strings for display
            p_terms = [f"x^{m}"] + [f"{p[i].X:.4f}*x^{i}" for i in range(m - 1, -1, -1) if abs(p[i].X) > 1e-6]
            q_terms = [f"x^{n}"] + [f"{q[j].X:.4f}*x^{j}" for j in range(n - 1, -1, -1) if abs(q[j].X) > 1e-6]
            r_terms = [f"x^{m+n}"] + [f"x^{k}" for k in range(m + n - 1, -1, -1) if abs(r[k].X) > 0.5]

            # Clean up formatting for display
            def format_poly(terms):
                return " + ".join(terms).replace("+ -", "- ")

            print(f"\nP(x) = {format_poly(p_terms)}")
            print(f"Q(x) = {format_poly(q_terms)}")
            print(f"R(x) = {format_poly(r_terms)}")
        else:
            print("\n--- Trivial Solution Found ---")
            print("An optimal solution was found, but all coefficients are integers (0 or 1).")
            print("Objective value is zero or negligible.")

    elif model.Status == GRB.INFEASIBLE:
        print("\n--- Model is Infeasible ---")
        print("No solution exists for the given degrees and constraints.")
    else:
        print(m,n)
        print(model.ObjVal)
        print(f"\n--- No Optimal Solution Found (Status: {model.Status}) ---")


if __name__ == '__main__':
    # Example: Find a non-trivial factorization of a 4th-degree polynomial.
    # Let P(x) and Q(x) both be of degree 2.
    # deg(P) = m, deg(Q) = n
    # deg(R) = m + n

    m_degree = 16
    n_degree = 16
    solve_polynomial_factorization(m_degree, n_degree)

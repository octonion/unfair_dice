import hexaly.optimizer
import sys

def solve_polynomial_factorization(deg_p, deg_q, time_limit_sec):
    """
    Models and solves the polynomial factorization problem using Hexaly.

    Args:
        deg_p (int): Degree of polynomial P.
        deg_q (int): Degree of polynomial Q.
        time_limit_sec (int): Solver time limit in seconds.
    """
    with hexaly.optimizer.HexalyOptimizer() as h:
        # Set a time limit for the solver
        h.param.time_limit = time_limit_sec

        m = h.model
        
        deg_r = deg_p + deg_q

        # --- Decision Variables ---
        # Create coefficients for P(x), Q(x), and R(x) as floating-point
        # variables bounded between 0 and 1.
        p = [m.float(0, 1) for _ in range(deg_p + 1)]
        q = [m.float(0, 1) for _ in range(deg_q + 1)]
        r = [m.int(0, 1) for _ in range(deg_r + 1)]

        # --- Constraints ---
        # Enforce that P(x) and Q(x) are monic polynomials.
        m.constraint(p[deg_p] == 1)
        m.constraint(q[deg_q] == 1)
        m.constraint(r[deg_r] == 1)

        # Convolution constraints: r_k = sum_{i+j=k} p_i * q_j
        for k in range(deg_r + 1):
            convolution_sum = m.sum(
                p[i] * q[k - i] 
                for i in range(k + 1) 
                if i <= deg_p and (k - i) <= deg_q
            )
            m.constraint(r[k] == convolution_sum)

        # --- Objective Function ---
        # Maximize the non-binariness of P and Q, while penalizing
        # the non-binariness of R.
        
        # Encourage p_i and q_j to be non-binary
        p_obj = m.sum(p[i] - p[i] * p[i] for i in range(deg_p))
        q_obj = m.sum(q[j] - q[j] * q[j] for j in range(deg_q))
        
        # If you use this objective function, the coefficients should
        # be modeled as continuous
        # Penalize r_k for being non-binary
        #r_penalty = m.sum(r[k] - r[k] * r[k] for k in range(deg_r + 1))
        #objective = (p_obj + q_obj) - (deg_p + deg_q) * r_penalty

        # Use this objective function if r coefficients are modeled
        # as integers
        objective = (p_obj + q_obj)
        m.maximize(objective)

        m.close()

        # --- Solve and Display Results ---
        h.solve()

        print(f"--- Solver Results (Time Limit: {time_limit_sec}s) ---")
        
        # The following needs to be fixed per Hexaly engine options
        
        #if h.solution.status == hexaly.SolutionStatus.OPTIMAL or h.solution.status == hexaly.SolutionStatus.FEASIBLE:
        if False:
            print(f"Objective value: {h.solution.objective_value}")
            
            print("\n--- Polynomial P(x) coefficients ---")
            p_values = [p[i].value for i in range(deg_p + 1)]
            print(p_values)

            print("\n--- Polynomial Q(x) coefficients ---")
            q_values = [q[j].value for j in range(deg_q + 1)]
            print(q_values)

            print("\n--- Polynomial R(x) coefficients ---")
            r_values = [r[k].value for k in range(deg_r + 1)]
            print(r_values)

            print("\n--- Non-Binary Contribution (P) ---")
            for i in range(deg_p):
                val = p_values[i]
                non_binary_term = val - val**2
                if non_binary_term > 1e-6:
                    print(f"  p[{i}] = {val:.4f} (term: {non_binary_term:.4f})")
            
            print("\n--- Non-Binary Contribution (Q) ---")
            for j in range(deg_q):
                val = q_values[j]
                non_binary_term = val - val**2
                if non_binary_term > 1e-6:
                    print(f"  q[{j}] = {val:.4f} (term: {non_binary_term:.4f})")

        #else:
        #    print(f"No solution found or an error occurred. Status: {h.solution.status}")


if __name__ == "__main__":
    # --- Configuration ---
    DEGREE_P = 8
    DEGREE_Q = 8
    TIME_LIMIT_SECONDS = 12000  # Adjust the time limit as needed

    solve_polynomial_factorization(DEGREE_P, DEGREE_Q, TIME_LIMIT_SECONDS)


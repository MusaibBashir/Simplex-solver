import streamlit as st
import numpy as np
import pandas as pd
from fractions import Fraction
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon

class SimplexSolver:
    def __init__(self, c, A, b, constraints, maximize=True):
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.constraints = constraints
        self.maximize = maximize
        self.num_vars = len(c)
        self.num_constraints = len(b)
        
        # Variables tracking
        self.slack_vars = []
        self.surplus_vars = []
        self.artificial_vars = []
        self.basic_vars = []
        self.all_var_names = []
        
        # Results storage
        self.algebraic_iterations = []
        self.tabular_iterations = []
        self.phase_one_iterations = []
        self.corner_points = []
        
        # Method selection
        self.method = None
        self.needs_artificial = False
        
        # Check if standard form
        self.check_standard_form()
        
        # Setup the problem
        if self.method:
            self.setup_problem()
    
    def check_standard_form(self):
        """Check if problem is in standard form and determine method needed"""
        has_ge_or_eq = any(constraint in ['>=', '='] for constraint in self.constraints)
        has_negative_rhs = any(b_val < 0 for b_val in self.b)
        
        if has_ge_or_eq or has_negative_rhs:
            self.needs_artificial = True
            st.warning("Problem is not in standard form. Artificial variables needed.")
            
            # Let user choose method
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Use Big M Method", type="primary"):
                    self.method = "big_m"
                    st.success("Big M Method selected")
            with col2:
                if st.button("Use Two-Phase Method", type="primary"):
                    self.method = "two_phase"
                    st.success("Two-Phase Method selected")
                    
            if not self.method:
                st.info("Please select a method to solve the problem")
                return
        else:
            self.method = "standard"
    
    def setup_problem(self):
        """Convert to standard form by adding slack, surplus, and artificial variables"""
        # Handle negative RHS
        for i in range(self.num_constraints):
            if self.b[i] < 0:
                self.b[i] = -self.b[i]
                self.A[i] = -self.A[i]
                # Flip constraint type
                if self.constraints[i] == '<=':
                    self.constraints[i] = '>='
                elif self.constraints[i] == '>=':
                    self.constraints[i] = '<='
        
        # Original variables
        for i in range(self.num_vars):
            self.all_var_names.append(f'x{i+1}')
        
        # Build tableau with proper dimensions
        total_vars = self.num_vars
        
        # Count additional variables needed
        for constraint_type in self.constraints:
            if constraint_type == '<=':
                total_vars += 1  # slack
            elif constraint_type == '>=':
                total_vars += 2  # surplus + artificial
            else:  # '='
                total_vars += 1  # artificial
        
        # Initialize tableau
        self.tableau = np.zeros((self.num_constraints + 1, total_vars + 2))
        
        # For Big M method, we'll use symbolic representation
        if self.method == "big_m":
            # Create separate arrays for M coefficients
            self.m_coeffs = np.zeros((self.num_constraints + 1, total_vars + 2))
        
        # Fill original constraint coefficients
        for i in range(self.num_constraints):
            for j in range(self.num_vars):
                self.tableau[i, j+1] = self.A[i][j]
            self.tableau[i, -1] = self.b[i]  # RHS
        
        # Fill objective function coefficients
        for j in range(self.num_vars):
            if self.maximize:
                self.tableau[-1, j+1] = -self.c[j]
            else:
                self.tableau[-1, j+1] = self.c[j]
        
        # Add slack, surplus, and artificial variables
        col_idx = self.num_vars + 1
        
        for i, constraint_type in enumerate(self.constraints):
            if constraint_type == '<=':
                # Add slack variable
                slack_var = f's{len(self.slack_vars)+1}'
                self.slack_vars.append(slack_var)
                self.all_var_names.append(slack_var)
                self.tableau[i, col_idx] = 1
                self.tableau[-1, col_idx] = 0
                self.basic_vars.append(col_idx - 1)
                col_idx += 1
                
            elif constraint_type == '>=':
                # Add surplus variable
                surplus_var = f'e{len(self.surplus_vars)+1}'
                self.surplus_vars.append(surplus_var)
                self.all_var_names.append(surplus_var)
                self.tableau[i, col_idx] = -1
                self.tableau[-1, col_idx] = 0
                col_idx += 1
                
                # Add artificial variable
                artificial_var = f'a{len(self.artificial_vars)+1}'
                self.artificial_vars.append(artificial_var)
                self.all_var_names.append(artificial_var)
                self.tableau[i, col_idx] = 1
                
                if self.method == "big_m":
                    self.tableau[-1, col_idx] = 0
                    self.m_coeffs[-1, col_idx] = 1 if self.maximize else -1
                else:  # two_phase or using big number
                    self.tableau[-1, col_idx] = 1000 if self.maximize else -1000
                
                self.basic_vars.append(col_idx - 1)
                col_idx += 1
                
            else:  # '='
                # Add artificial variable
                artificial_var = f'a{len(self.artificial_vars)+1}'
                self.artificial_vars.append(artificial_var)
                self.all_var_names.append(artificial_var)
                self.tableau[i, col_idx] = 1
                
                if self.method == "big_m":
                    self.tableau[-1, col_idx] = 0
                    self.m_coeffs[-1, col_idx] = 1 if self.maximize else -1
                else:
                    self.tableau[-1, col_idx] = 1000 if self.maximize else -1000
                
                self.basic_vars.append(col_idx - 1)
                col_idx += 1
        
        # Update objective row for artificial variables
        if self.method in ["big_m", "standard"]:
            self.update_objective_row()
    
    def update_objective_row(self):
        """Update objective row to account for basic artificial variables"""
        for i, basic_var_idx in enumerate(self.basic_vars):
            # Check if it's an artificial variable
            if basic_var_idx >= self.num_vars + len(self.slack_vars) + len(self.surplus_vars):
                if self.method == "big_m":
                    # For Big M, update both regular and M coefficients
                    m_multiplier = self.m_coeffs[-1, basic_var_idx + 1]
                    if m_multiplier != 0:
                        self.m_coeffs[-1, :] -= m_multiplier * self.tableau[i, :]
                else:
                    # For regular method
                    multiplier = self.tableau[-1, basic_var_idx + 1]
                    if multiplier != 0:
                        self.tableau[-1, :] -= multiplier * self.tableau[i, :]
        
        # Calculate Z value
        self.calculate_z_value()
    
    def calculate_z_value(self):
        """Calculate current Z value from basic variables"""
        z = 0
        m_z = 0
        
        for i, basic_var_idx in enumerate(self.basic_vars):
            if basic_var_idx < self.num_vars:
                coef = self.c[basic_var_idx]
                value = self.tableau[i, -1]
                z += coef * value
            elif self.method == "big_m" and basic_var_idx >= self.num_vars + len(self.slack_vars) + len(self.surplus_vars):
                # Artificial variable contribution to M
                value = self.tableau[i, -1]
                m_z += value
        
        if self.maximize:
            self.tableau[-1, 0] = z
            if self.method == "big_m":
                self.m_coeffs[-1, 0] = m_z
        else:
            self.tableau[-1, 0] = -z
            if self.method == "big_m":
                self.m_coeffs[-1, 0] = -m_z
    
    def solve_two_phase(self):
        """Solve using Two-Phase Method"""
        st.subheader("Two-Phase Method Solution")
        st.write("### 2 phase reached")
        # Phase I: Minimize sum of artificial variables
        st.write("### Phase I: Minimize sum of artificial variables")
        
        # Create Phase I tableau
        phase1_tableau = copy.deepcopy(self.tableau)
        
        # Set up Phase I objective: minimize sum of artificial variables
        phase1_tableau[-1, :] = 0  # Clear objective row
        for i, basic_var_idx in enumerate(self.basic_vars):
            if basic_var_idx >= self.num_vars + len(self.slack_vars) + len(self.surplus_vars):
                # This is an artificial variable
                phase1_tableau[-1, basic_var_idx + 1] = 1
        
        # Eliminate artificial variables from objective row
        for i, basic_var_idx in enumerate(self.basic_vars):
            if basic_var_idx >= self.num_vars + len(self.slack_vars) + len(self.surplus_vars):
                multiplier = phase1_tableau[-1, basic_var_idx + 1]
                if multiplier != 0:
                    phase1_tableau[-1, :] -= multiplier * phase1_tableau[i, :]
        
        # Solve Phase I
        phase1_solution, phase1_basic_vars = self.solve_phase(phase1_tableau, self.basic_vars.copy(), phase=1)
        
        if phase1_solution is None:
            return None
        
        # Check if Phase I solution is feasible (all artificial variables = 0)
        artificial_sum = 0
        for i, basic_var_idx in enumerate(phase1_basic_vars):
            if basic_var_idx >= self.num_vars + len(self.slack_vars) + len(self.surplus_vars):
                artificial_sum += abs(phase1_tableau[i, -1])
        
        if artificial_sum > 1e-6:
            st.error("Problem is infeasible! Artificial variables are non-zero in Phase I optimal solution.")
            return None
        
        st.success("Phase I completed successfully. All artificial variables eliminated.")
        
        # Phase II: Solve original problem
        st.write("### Phase II: Solve original problem")
        
        # Create Phase II tableau from Phase I result
        phase2_tableau = copy.deepcopy(phase1_tableau)
        
        # Restore original objective function
        phase2_tableau[-1, :] = 0
        for j in range(self.num_vars):
            if self.maximize:
                phase2_tableau[-1, j+1] = -self.c[j]
            else:
                phase2_tableau[-1, j+1] = self.c[j]
        
        # Eliminate basic variables from objective row
        for i, basic_var_idx in enumerate(phase1_basic_vars):
            if basic_var_idx < self.num_vars:  # Only for original variables
                multiplier = phase2_tableau[-1, basic_var_idx + 1]
                if multiplier != 0:
                    phase2_tableau[-1, :] -= multiplier * phase2_tableau[i, :]
        
        # Calculate Z value for Phase II
        z = 0
        for i, basic_var_idx in enumerate(phase1_basic_vars):
            if basic_var_idx < self.num_vars:
                coef = self.c[basic_var_idx]
                value = phase2_tableau[i, -1]
                z += coef * value
        
        phase2_tableau[-1, 0] = z if self.maximize else -z
        
        # Solve Phase II
        return self.solve_phase(phase2_tableau, phase1_basic_vars, phase=2)
    
    def solve_phase(self, tableau, basic_vars, phase):
        """Solve a single phase of the simplex method"""
        iteration = 0
        
        while True:
            iteration += 1
            
            # Store corner point for visualization
            if self.num_vars == 2:
                point = self.extract_corner_point(tableau, basic_vars)
                if point:
                    self.corner_points.append({
                        'point': point,
                        'iteration': iteration,
                        'phase': phase,
                        'is_optimal': False
                    })
            
            # Check optimality
            obj_row = tableau[-1, 1:-1]
            
            if all(val >= -1e-6 for val in obj_row):
                # Optimal solution found
                if self.num_vars == 2 and self.corner_points:
                    self.corner_points[-1]['is_optimal'] = True
                self.record_phase_iteration(tableau, iteration, basic_vars, phase, is_final=True)
                break
            
            # Find entering variable (most negative)
            pivot_col = np.argmin(obj_row) + 1
            
            # Find leaving variable using minimum ratio test
            ratios = []
            for i in range(len(tableau) - 1):
                if tableau[i, pivot_col] > 1e-6:
                    ratios.append((tableau[i, -1] / tableau[i, pivot_col], i))
                else:
                    ratios.append((float('inf'), i))
            
            # Check for unboundedness
            if all(r[0] == float('inf') for r in ratios):
                if phase == 1:
                    st.error("Phase I is unbounded - this shouldn't happen!")
                    return None, None
                else:
                    st.error("Problem is unbounded!")
                    return None, None
            
            pivot_row = min(ratios)[1]
            pivot_element = tableau[pivot_row, pivot_col]
            
            # Record iteration
            self.record_phase_iteration(tableau, iteration, basic_vars, phase, pivot_row, pivot_col, pivot_element)
            
            # Perform pivot operation
            tableau[pivot_row, :] = tableau[pivot_row, :] / pivot_element
            
            for i in range(len(tableau)):
                if i != pivot_row:
                    multiplier = tableau[i, pivot_col]
                    tableau[i, :] = tableau[i, :] - multiplier * tableau[pivot_row, :]
            
            # Update basic variables
            basic_vars[pivot_row] = pivot_col - 1
            
            # Recalculate Z value
            if phase == 2:
                z = 0
                for i, basic_var_idx in enumerate(basic_vars):
                    if basic_var_idx < self.num_vars:
                        coef = self.c[basic_var_idx]
                        value = tableau[i, -1]
                        z += coef * value
                
                tableau[-1, 0] = z if self.maximize else -z
        
        return self.extract_solution(tableau, basic_vars), basic_vars
    
    def extract_corner_point(self, tableau, basic_vars):
        """Extract corner point coordinates for 2-variable problems"""
        if self.num_vars != 2:
            return None
        
        point = [0, 0]  # Initialize with zeros
        
        for i, basic_var_idx in enumerate(basic_vars):
            if basic_var_idx < self.num_vars:  # Only consider original variables
                point[basic_var_idx] = max(0, tableau[i, -1])
        
        return tuple(point)
    
    def record_phase_iteration(self, tableau, iteration, basic_vars, phase, pivot_row=None, pivot_col=None, pivot_element=None, is_final=False):
        """Record phase iteration details"""
        basic_vars_info = []
        for i, bv_idx in enumerate(basic_vars):
            if bv_idx < len(self.all_var_names):
                var_name = self.all_var_names[bv_idx]
                value = max(0, tableau[i, -1])
                basic_vars_info.append(f"{var_name} = {value:.3f}")
        
        z_value = tableau[-1, 0]
        
        iteration_info = {
            'phase': phase,
            'iteration': iteration,
            'basic_vars': basic_vars_info,
            'z_value': z_value,
            'is_final': is_final
        }
        
        if not is_final and pivot_row is not None:
            iteration_info['pivot_info'] = {
                'row': pivot_row + 1,
                'col': self.all_var_names[pivot_col - 1] if pivot_col - 1 < len(self.all_var_names) else f'var{pivot_col}',
                'element': pivot_element
            }
        
        self.phase_one_iterations.append(iteration_info)
    
    def solve_big_m(self):
        """Solve using Big M Method with symbolic M representation"""
        st.subheader("Big M Method Solution")
        st.info("M coefficients are shown separately from regular coefficients")
        
        tableau = copy.deepcopy(self.tableau)
        m_coeffs = copy.deepcopy(self.m_coeffs) if hasattr(self, 'm_coeffs') else None
        basic_vars = copy.deepcopy(self.basic_vars)
        iteration = 0
        
        while True:
            iteration += 1
            
            # Store corner point for visualization
            if self.num_vars == 2:
                point = self.extract_corner_point(tableau, basic_vars)
                if point:
                    self.corner_points.append({
                        'point': point,
                        'iteration': iteration,
                        'phase': 'Big M',
                        'is_optimal': False
                    })
            
            # Check optimality considering M coefficients
            obj_row = tableau[-1, 1:-1]
            m_obj_row = m_coeffs[-1, 1:-1] if m_coeffs is not None else np.zeros_like(obj_row)
            
            # For maximization: optimal if all coefficients ≥ 0
            # For M coefficients, they dominate regular coefficients
            optimal = True
            pivot_col = -1
            most_negative_m = 0
            most_negative_regular = 0
            
            for j in range(len(obj_row)):
                if self.maximize:
                    if m_obj_row[j] < -1e-6:  # M coefficient is negative
                        optimal = False
                        if m_obj_row[j] < most_negative_m:
                            most_negative_m = m_obj_row[j]
                            pivot_col = j + 1
                    elif abs(m_obj_row[j]) < 1e-6 and obj_row[j] < -1e-6:  # M coeff is 0, regular is negative
                        optimal = False
                        if obj_row[j] < most_negative_regular and most_negative_m == 0:
                            most_negative_regular = obj_row[j]
                            pivot_col = j + 1
                else:  # Minimization
                    if m_obj_row[j] > 1e-6:  # M coefficient is positive
                        optimal = False
                        if m_obj_row[j] > most_negative_m:
                            most_negative_m = m_obj_row[j]
                            pivot_col = j + 1
                    elif abs(m_obj_row[j]) < 1e-6 and obj_row[j] > 1e-6:  # M coeff is 0, regular is positive
                        optimal = False
                        if obj_row[j] > most_negative_regular and most_negative_m == 0:
                            most_negative_regular = obj_row[j]
                            pivot_col = j + 1
            
            if optimal:
                # Check for infeasibility (artificial variables in basis)
                for i, basic_var_idx in enumerate(basic_vars):
                    if basic_var_idx >= self.num_vars + len(self.slack_vars) + len(self.surplus_vars):
                        if tableau[i, -1] > 1e-6:
                            st.error("Problem is infeasible! Artificial variables remain in optimal solution.")
                            return None
                
                # Mark optimal point
                if self.num_vars == 2 and self.corner_points:
                    self.corner_points[-1]['is_optimal'] = True
                
                # Record final iteration
                self.record_big_m_iteration(tableau, m_coeffs, iteration, basic_vars, is_final=True)
                break
            
            # Find leaving variable using minimum ratio test
            ratios = []
            for i in range(len(tableau) - 1):
                if tableau[i, pivot_col] > 1e-6:
                    ratios.append((tableau[i, -1] / tableau[i, pivot_col], i))
                else:
                    ratios.append((float('inf'), i))
            
            # Check for unboundedness
            if all(r[0] == float('inf') for r in ratios):
                st.error("Problem is unbounded!")
                return None
            
            pivot_row = min(ratios)[1]
            pivot_element = tableau[pivot_row, pivot_col]
            
            # Record iteration before pivoting
            self.record_big_m_iteration(tableau, m_coeffs, iteration, basic_vars, pivot_row, pivot_col, pivot_element)
            
            # Perform pivot operation on both tableaus
            tableau[pivot_row, :] = tableau[pivot_row, :] / pivot_element
            if m_coeffs is not None:
                m_coeffs[pivot_row, :] = m_coeffs[pivot_row, :] / pivot_element
            
            for i in range(len(tableau)):
                if i != pivot_row:
                    multiplier = tableau[i, pivot_col]
                    tableau[i, :] = tableau[i, :] - multiplier * tableau[pivot_row, :]
                    
                    if m_coeffs is not None:
                        m_multiplier = m_coeffs[i, pivot_col]
                        m_coeffs[i, :] = m_coeffs[i, :] - m_multiplier * m_coeffs[pivot_row, :]
            
            # Update basic variables
            basic_vars[pivot_row] = pivot_col - 1
            
            # Recalculate Z value
            z = 0
            m_z = 0
            for i, basic_var_idx in enumerate(basic_vars):
                if basic_var_idx < self.num_vars:
                    coef = self.c[basic_var_idx]
                    value = tableau[i, -1]
                    z += coef * value
                elif basic_var_idx >= self.num_vars + len(self.slack_vars) + len(self.surplus_vars):
                    # Artificial variable
                    value = tableau[i, -1]
                    m_z += value
            
            tableau[-1, 0] = z if self.maximize else -z
            if m_coeffs is not None:
                m_coeffs[-1, 0] = m_z if self.maximize else -m_z
        
        return self.extract_solution(tableau, basic_vars)
    
    def record_big_m_iteration(self, tableau, m_coeffs, iteration, basic_vars, pivot_row=None, pivot_col=None, pivot_element=None, is_final=False):
        """Record Big M iteration with M coefficients displayed"""
        basic_vars_info = []
        for i, bv_idx in enumerate(basic_vars):
            if bv_idx < len(self.all_var_names):
                var_name = self.all_var_names[bv_idx]
                value = max(0, tableau[i, -1])
                basic_vars_info.append(f"{var_name} = {value:.3f}")
        
        # Format Z value with M coefficient
        z_regular = tableau[-1, 0]
        z_m = m_coeffs[-1, 0] if m_coeffs is not None else 0
        
        if abs(z_m) < 1e-6:
            z_display = f"{z_regular:.3f}"
        else:
            z_display = f"{z_regular:.3f} + {z_m:.3f}M" if z_m > 0 else f"{z_regular:.3f} - {abs(z_m):.3f}M"
        
        iteration_info = {
            'iteration': iteration,
            'basic_vars': basic_vars_info,
            'z_value': z_display,
            'z_regular': z_regular,
            'z_m': z_m,
            'is_final': is_final,
            'tableau': copy.deepcopy(tableau),
            'm_coeffs': copy.deepcopy(m_coeffs) if m_coeffs is not None else None
        }
        
        if not is_final and pivot_row is not None:
            iteration_info['pivot_info'] = {
                'row': pivot_row + 1,
                'col': self.all_var_names[pivot_col - 1] if pivot_col - 1 < len(self.all_var_names) else f'var{pivot_col}',
                'element': pivot_element
            }
        
        self.algebraic_iterations.append(iteration_info)
    
    def check_multiple_solutions(self, tableau, basic_vars):
        """Check for multiple optimal solutions"""
        obj_row = tableau[-1, 1:-1]
        
        # Check non-basic variables for zero coefficients
        multiple_solutions = []
        for j in range(len(obj_row)):
            if abs(obj_row[j]) < 1e-6:  # Coefficient is zero
                var_idx = j
                if var_idx not in basic_vars:  # Non-basic variable
                    var_name = self.all_var_names[var_idx] if var_idx < len(self.all_var_names) else f'var{var_idx+1}'
                    multiple_solutions.append(var_name)
        
        return multiple_solutions
    
    def solve_algebraic(self):
        """Solve using algebraic method based on selected approach"""
        if self.method == "big_m":
            return self.solve_big_m()
        elif self.method == "two_phase":
            result = self.solve_two_phase()
            return result[0] if result else None
        else:
            return self.solve_standard_algebraic()
    
    def solve_standard_algebraic(self):
        """Solve standard form using algebraic method"""
        tableau = copy.deepcopy(self.tableau)
        iteration = 0
        
        while True:
            iteration += 1
            
            # Store corner point for visualization
            if self.num_vars == 2:
                point = self.extract_corner_point(tableau, self.basic_vars)
                if point:
                    self.corner_points.append({
                        'point': point,
                        'iteration': iteration,
                        'phase': 'Standard',
                        'is_optimal': False
                    })
            
            # Check for negative RHS values (infeasibility indicator)
            for i in range(len(tableau) - 1):
                if tableau[i, -1] < -1e-6:
                    st.warning(f"Warning: Negative RHS in row {i+1}: {tableau[i, -1]:.3f}")
            
            # Find entering variable
            obj_row = tableau[-1, 1:-1]
            
            if self.maximize:
                if all(val >= -1e-6 for val in obj_row):
                    # Check for multiple solutions
                    multiple_sols = self.check_multiple_solutions(tableau, self.basic_vars)
                    if self.num_vars == 2 and self.corner_points:
                        self.corner_points[-1]['is_optimal'] = True
                    self.record_algebraic_iteration(tableau, iteration, self.basic_vars, is_final=True, multiple_solutions=multiple_sols)
                    break
                pivot_col = np.argmin(obj_row) + 1
            else:
                if all(val <= 1e-6 for val in obj_row):
                    multiple_sols = self.check_multiple_solutions(tableau, self.basic_vars)
                    if self.num_vars == 2 and self.corner_points:
                        self.corner_points[-1]['is_optimal'] = True
                    self.record_algebraic_iteration(tableau, iteration, self.basic_vars, is_final=True, multiple_solutions=multiple_sols)
                    break
                pivot_col = np.argmax(obj_row) + 1
            
            # Find leaving variable
            ratios = []
            for i in range(len(tableau) - 1):
                if tableau[i, pivot_col] > 1e-6:
                    ratios.append((tableau[i, -1] / tableau[i, pivot_col], i))
                else:
                    ratios.append((float('inf'), i))
            
            # Check for unboundedness
            if all(r[0] == float('inf') for r in ratios):
                st.error("Problem is unbounded! No leaving variable found.")
                return None
            
            pivot_row = min(ratios)[1]
            pivot_element = tableau[pivot_row, pivot_col]
            
            # Record current iteration
            self.record_algebraic_iteration(tableau, iteration, self.basic_vars, pivot_row, pivot_col, pivot_element)
            
            # Perform pivot operation
            tableau[pivot_row, :] = tableau[pivot_row, :] / pivot_element
            
            for i in range(len(tableau)):
                if i != pivot_row:
                    multiplier = tableau[i, pivot_col]
                    tableau[i, :] = tableau[i, :] - multiplier * tableau[pivot_row, :]
            
            self.basic_vars[pivot_row] = pivot_col - 1
            
            # Recalculate Z value
            z = 0
            for i, basic_var_idx in enumerate(self.basic_vars):
                if basic_var_idx < self.num_vars:
                    coef = self.c[basic_var_idx]
                    value = tableau[i, -1]
                    z += coef * value
            
            tableau[-1, 0] = z if self.maximize else -z
        
        return self.extract_solution(tableau, self.basic_vars)
    
    def record_algebraic_iteration(self, tableau, iteration, basic_vars, pivot_row=None, pivot_col=None, pivot_element=None, is_final=False, multiple_solutions=None):
        """Record algebraic iteration details"""
        basic_vars_info = []
        for i, bv_idx in enumerate(basic_vars):
            if bv_idx < len(self.all_var_names):
                var_name = self.all_var_names[bv_idx]
                value = max(0, tableau[i, -1])
                basic_vars_info.append(f"{var_name} = {value:.3f}")
        
        z_value = tableau[-1, 0]
        
        iteration_info = {
            'iteration': iteration,
            'basic_vars': basic_vars_info,
            'z_value': z_value,
            'is_final': is_final,
            'multiple_solutions': multiple_solutions or []
        }
        
        if not is_final and pivot_row is not None:
            iteration_info['pivot_info'] = {
                'row': pivot_row + 1,
                'col': self.all_var_names[pivot_col - 1] if pivot_col - 1 < len(self.all_var_names) else f'var{pivot_col}',
                'element': pivot_element
            }
            
            # Add calculations
            calculations = []
            calculations.append(f"Entering variable: {iteration_info['pivot_info']['col']}")
            calculations.append(f"Leaving variable: Row {iteration_info['pivot_info']['row']}")
            calculations.append(f"Pivot element: {pivot_element:.3f}")
            iteration_info['calculations'] = calculations
        
        self.algebraic_iterations.append(iteration_info)
    
    def solve_tabular(self):
        """Solve using tabular method with formatted tables"""
        if self.method == "big_m":
            return self.solve_big_m_tabular()
        elif self.method == "two_phase":
            return self.solve_two_phase_tabular()
        else:
            return self.solve_standard_tabular()
    
    def solve_big_m_tabular(self):
        """Solve Big M method with tabular display"""
        st.subheader("Big M Method - Tabular Solution")
        
        tableau = copy.deepcopy(self.tableau)
        m_coeffs = copy.deepcopy(self.m_coeffs) if hasattr(self, 'm_coeffs') else None
        basic_vars = copy.deepcopy(self.basic_vars)
        iteration = 0
        
        while True:
            iteration += 1
            
            # Store corner point for visualization
            if self.num_vars == 2:
                point = self.extract_corner_point(tableau, basic_vars)
                if point:
                    self.corner_points.append({
                        'point': point,
                        'iteration': iteration,
                        'phase': 'Big M',
                        'is_optimal': False
                    })
            
            # Create and display table
            self.create_big_m_tabular_iteration(tableau, m_coeffs, iteration, basic_vars)
            
            # Check optimality
            obj_row = tableau[-1, 1:-1]
            m_obj_row = m_coeffs[-1, 1:-1] if m_coeffs is not None else np.zeros_like(obj_row)
            
            optimal = True
            pivot_col = -1
            
            for j in range(len(obj_row)):
                if self.maximize:
                    if m_obj_row[j] < -1e-6 or (abs(m_obj_row[j]) < 1e-6 and obj_row[j] < -1e-6):
                        optimal = False
                        if m_obj_row[j] < -1e-6:
                            pivot_col = j + 1
                            break
                        elif abs(m_obj_row[j]) < 1e-6 and obj_row[j] < -1e-6:
                            pivot_col = j + 1
                else:
                    if m_obj_row[j] > 1e-6 or (abs(m_obj_row[j]) < 1e-6 and obj_row[j] > 1e-6):
                        optimal = False
                        if m_obj_row[j] > 1e-6:
                            pivot_col = j + 1
                            break
                        elif abs(m_obj_row[j]) < 1e-6 and obj_row[j] > 1e-6:
                            pivot_col = j + 1
            
            if optimal:
                # Check for infeasibility
                for i, basic_var_idx in enumerate(basic_vars):
                    if basic_var_idx >= self.num_vars + len(self.slack_vars) + len(self.surplus_vars):
                        if tableau[i, -1] > 1e-6:
                            st.error("Problem is infeasible! Artificial variables remain in solution.")
                            return None
                
                # Mark optimal point
                if self.num_vars == 2 and self.corner_points:
                    self.corner_points[-1]['is_optimal'] = True
                
                # Check for multiple solutions
                multiple_sols = self.check_multiple_solutions(tableau, basic_vars)
                if multiple_sols:
                    st.info(f"Multiple optimal solutions exist! Non-basic variables with zero coefficients: {', '.join(multiple_sols)}")
                
                st.success("Optimal solution found!")
                break
            
            # Find leaving variable
            ratios = []
            for i in range(len(tableau) - 1):
                if tableau[i, pivot_col] > 1e-6:
                    ratios.append(tableau[i, -1] / tableau[i, pivot_col])
                else:
                    ratios.append(float('inf'))
            
            if all(r == float('inf') for r in ratios):
                st.error("Problem is unbounded!")
                return None
            
            pivot_row = ratios.index(min(ratios))
            pivot_element = tableau[pivot_row, pivot_col]
            
            # Display pivot information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Entering:** {self.all_var_names[pivot_col-1] if pivot_col-1 < len(self.all_var_names) else f'var{pivot_col}'}")
            with col2:
                st.info(f"**Leaving:** Row {pivot_row + 1}")
            with col3:
                st.info(f"**Pivot:** {pivot_element:.3f}")
            
            # Perform pivot operation
            tableau[pivot_row, :] = tableau[pivot_row, :] / pivot_element
            if m_coeffs is not None:
                m_coeffs[pivot_row, :] = m_coeffs[pivot_row, :] / pivot_element
            
            for i in range(len(tableau)):
                if i != pivot_row:
                    multiplier = tableau[i, pivot_col]
                    tableau[i, :] = tableau[i, :] - multiplier * tableau[pivot_row, :]
                    
                    if m_coeffs is not None:
                        m_multiplier = m_coeffs[i, pivot_col]
                        m_coeffs[i, :] = m_coeffs[i, :] - m_multiplier * m_coeffs[pivot_row, :]
            
            basic_vars[pivot_row] = pivot_col - 1
            
            # Update Z values
            z = 0
            m_z = 0
            for i, basic_var_idx in enumerate(basic_vars):
                if basic_var_idx < self.num_vars:
                    coef = self.c[basic_var_idx]
                    value = tableau[i, -1]
                    z += coef * value
                elif basic_var_idx >= self.num_vars + len(self.slack_vars) + len(self.surplus_vars):
                    value = tableau[i, -1]
                    m_z += value
            
            tableau[-1, 0] = z if self.maximize else -z
            if m_coeffs is not None:
                m_coeffs[-1, 0] = m_z if self.maximize else -m_z
            
            st.markdown("---")
        
        return self.extract_solution(tableau, basic_vars)
    
    def create_big_m_tabular_iteration(self, tableau, m_coeffs, iteration, basic_vars):
        """Create Big M tabular iteration with separate M coefficients"""
        st.write(f"### Iteration {iteration}")
        
        # Create two tables: one for regular coefficients, one for M coefficients
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Regular Coefficients:**")
            data_regular = []
            
            # Z row
            z_row = {
                'Basic Var': 'Z',
                'Z': f"{tableau[-1, 0]:.3f}"
            }
            for j, var_name in enumerate(self.all_var_names):
                z_row[var_name] = f"{tableau[-1, j+1]:.3f}"
            z_row['RHS'] = f"{tableau[-1, -1]:.3f}"
            data_regular.append(z_row)
            
            # Constraint rows
            for i in range(len(tableau) - 1):
                row_data = {
                    'Basic Var': self.all_var_names[basic_vars[i]] if basic_vars[i] < len(self.all_var_names) else f'var{basic_vars[i]+1}',
                    'Z': f"{tableau[i, 0]:.3f}"
                }
                for j, var_name in enumerate(self.all_var_names):
                    row_data[var_name] = f"{tableau[i, j+1]:.3f}"
                row_data['RHS'] = f"{max(0, tableau[i, -1]):.3f}"
                data_regular.append(row_data)
            
            df_regular = pd.DataFrame(data_regular)
            st.dataframe(df_regular, use_container_width=True, hide_index=True)
        
        if m_coeffs is not None:
            with col2:
                st.write("**M Coefficients:**")
                data_m = []
                
                # Z row
                z_row_m = {
                    'Basic Var': 'Z',
                    'Z': f"{m_coeffs[-1, 0]:.3f}"
                }
                for j, var_name in enumerate(self.all_var_names):
                    z_row_m[var_name] = f"{m_coeffs[-1, j+1]:.3f}"
                z_row_m['RHS'] = f"{m_coeffs[-1, -1]:.3f}"
                data_m.append(z_row_m)
                
                # Constraint rows
                for i in range(len(m_coeffs) - 1):
                    row_data = {
                        'Basic Var': self.all_var_names[basic_vars[i]] if basic_vars[i] < len(self.all_var_names) else f'var{basic_vars[i]+1}',
                        'Z': f"{m_coeffs[i, 0]:.3f}"
                    }
                    for j, var_name in enumerate(self.all_var_names):
                        row_data[var_name] = f"{m_coeffs[i, j+1]:.3f}"
                    row_data['RHS'] = f"{m_coeffs[i, -1]:.3f}"
                    data_m.append(row_data)
                
                df_m = pd.DataFrame(data_m)
                st.dataframe(df_m, use_container_width=True, hide_index=True)
    
    def solve_two_phase_tabular(self):
        """Solve Two-Phase method with tabular display"""
        st.subheader("Two-Phase Method - Tabular Solution")
        
        # Phase I
        st.write("### Phase I: Minimize sum of artificial variables")
        
        phase1_tableau = copy.deepcopy(self.tableau)
        phase1_basic_vars = copy.deepcopy(self.basic_vars)
        
        # Set up Phase I objective
        phase1_tableau[-1, :] = 0
        for i, basic_var_idx in enumerate(phase1_basic_vars):
            if basic_var_idx >= self.num_vars + len(self.slack_vars) + len(self.surplus_vars):
                phase1_tableau[-1, basic_var_idx + 1] = 1
        
        # Eliminate artificial variables from objective
        for i, basic_var_idx in enumerate(phase1_basic_vars):
            if basic_var_idx >= self.num_vars + len(self.slack_vars) + len(self.surplus_vars):
                multiplier = phase1_tableau[-1, basic_var_idx + 1]
                if multiplier != 0:
                    phase1_tableau[-1, :] -= multiplier * phase1_tableau[i, :]
        
        # Solve Phase I
        phase1_result = self.solve_phase_tabular(phase1_tableau, phase1_basic_vars, phase=1)
        if phase1_result is None:
            return None
        
        phase1_tableau, phase1_basic_vars = phase1_result
        
        # Check feasibility
        artificial_sum = 0
        for i, basic_var_idx in enumerate(phase1_basic_vars):
            if basic_var_idx >= self.num_vars + len(self.slack_vars) + len(self.surplus_vars):
                artificial_sum += abs(phase1_tableau[i, -1])
        
        if artificial_sum > 1e-6:
            st.error("Problem is infeasible! Artificial variables are non-zero in Phase I.")
            return None
        
        st.success("Phase I completed. All artificial variables eliminated.")
        
        # Phase II
        st.write("### Phase II: Solve original problem")
        
        phase2_tableau = copy.deepcopy(phase1_tableau)
        
        # Restore original objective
        phase2_tableau[-1, :] = 0
        for j in range(self.num_vars):
            if self.maximize:
                phase2_tableau[-1, j+1] = -self.c[j]
            else:
                phase2_tableau[-1, j+1] = self.c[j]
        
        # Eliminate basic variables from objective
        for i, basic_var_idx in enumerate(phase1_basic_vars):
            if basic_var_idx < self.num_vars:
                multiplier = phase2_tableau[-1, basic_var_idx + 1]
                if multiplier != 0:
                    phase2_tableau[-1, :] -= multiplier * phase2_tableau[i, :]
        
        # Calculate Z value
        z = 0
        for i, basic_var_idx in enumerate(phase1_basic_vars):
            if basic_var_idx < self.num_vars:
                coef = self.c[basic_var_idx]
                value = phase2_tableau[i, -1]
                z += coef * value
        phase2_tableau[-1, 0] = z if self.maximize else -z
        
        # Solve Phase II
        phase2_result = self.solve_phase_tabular(phase2_tableau, phase1_basic_vars, phase=2)
        if phase2_result is None:
            return None
        
        phase2_tableau, phase2_basic_vars = phase2_result
        return self.extract_solution(phase2_tableau, phase2_basic_vars)
    
    def solve_phase_tabular(self, tableau, basic_vars, phase):
        """Solve a single phase with tabular display"""
        iteration = 0
        
        while True:
            iteration += 1
            
            # Store corner point for visualization
            if self.num_vars == 2:
                point = self.extract_corner_point(tableau, basic_vars)
                if point:
                    self.corner_points.append({
                        'point': point,
                        'iteration': iteration,
                        'phase': f'Phase {phase}',
                        'is_optimal': False
                    })
            
            # Display current tableau
            self.create_phase_tabular_iteration(tableau, iteration, basic_vars, phase)
            
            # Check optimality
            obj_row = tableau[-1, 1:-1]
            
            if all(val >= -1e-6 for val in obj_row):
                if self.num_vars == 2 and self.corner_points:
                    self.corner_points[-1]['is_optimal'] = (phase == 2)  # Only mark optimal in Phase 2
                st.success(f"Phase {phase} optimal solution found!")
                break
            
            # Find entering variable
            pivot_col = np.argmin(obj_row) + 1
            
            # Find leaving variable
            ratios = []
            for i in range(len(tableau) - 1):
                if tableau[i, pivot_col] > 1e-6:
                    ratios.append(tableau[i, -1] / tableau[i, pivot_col])
                else:
                    ratios.append(float('inf'))
            
            if all(r == float('inf') for r in ratios):
                if phase == 1:
                    st.error("Phase I is unbounded!")
                else:
                    st.error("Problem is unbounded!")
                return None
            
            pivot_row = ratios.index(min(ratios))
            pivot_element = tableau[pivot_row, pivot_col]
            
            # Display pivot information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Entering:** {self.all_var_names[pivot_col-1] if pivot_col-1 < len(self.all_var_names) else f'var{pivot_col}'}")
            with col2:
                st.info(f"**Leaving:** Row {pivot_row + 1}")
            with col3:
                st.info(f"**Pivot:** {pivot_element:.3f}")
            
            # Perform pivot operation
            tableau[pivot_row, :] = tableau[pivot_row, :] / pivot_element
            
            for i in range(len(tableau)):
                if i != pivot_row:
                    multiplier = tableau[i, pivot_col]
                    tableau[i, :] = tableau[i, :] - multiplier * tableau[pivot_row, :]
            
            basic_vars[pivot_row] = pivot_col - 1
            
            # Update Z value for Phase II
            if phase == 2:
                z = 0
                for i, basic_var_idx in enumerate(basic_vars):
                    if basic_var_idx < self.num_vars:
                        coef = self.c[basic_var_idx]
                        value = tableau[i, -1]
                        z += coef * value
                tableau[-1, 0] = z if self.maximize else -z
            
            st.markdown("---")
        
        return tableau, basic_vars
    
    def create_phase_tabular_iteration(self, tableau, iteration, basic_vars, phase):
        """Create tabular iteration for phase display"""
        st.write(f"**Phase {phase} - Iteration {iteration}**")
        
        data = []
        
        # Z row
        z_row = {
            'Basic Var': f'Phase {phase} Obj',
            'Z': f"{tableau[-1, 0]:.3f}"
        }
        for j, var_name in enumerate(self.all_var_names):
            z_row[var_name] = f"{tableau[-1, j+1]:.3f}"
        z_row['RHS'] = f"{tableau[-1, -1]:.3f}"
        data.append(z_row)
        
        # Constraint rows
        for i in range(len(tableau) - 1):
            row_data = {
                'Basic Var': self.all_var_names[basic_vars[i]] if basic_vars[i] < len(self.all_var_names) else f'var{basic_vars[i]+1}',
                'Z': f"{tableau[i, 0]:.3f}"
            }
            for j, var_name in enumerate(self.all_var_names):
                row_data[var_name] = f"{tableau[i, j+1]:.3f}"
            row_data['RHS'] = f"{max(0, tableau[i, -1]):.3f}"
            data.append(row_data)
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    def solve_standard_tabular(self):
        """Solve standard form using tabular method"""
        tableau = copy.deepcopy(self.tableau)
        iteration = 0
        basic_vars = copy.deepcopy(self.basic_vars)
        
        while True:
            iteration += 1
            
            # Store corner point for visualization
            if self.num_vars == 2:
                point = self.extract_corner_point(tableau, basic_vars)
                if point:
                    self.corner_points.append({
                        'point': point,
                        'iteration': iteration,
                        'phase': 'Standard',
                        'is_optimal': False
                    })
            
            obj_row = tableau[-1, 1:-1]
            
            # Create and display current tableau
            ratios = []
            if iteration > 1:  # Only calculate ratios after first iteration when we have a pivot column
                for i in range(len(tableau) - 1):
                    ratios.append("-")  # Placeholder
            
            self.create_tabular_iteration(tableau, iteration, basic_vars, None, ratios)
            
            # Check optimality
            if self.maximize:
                if all(val >= -1e-6 for val in obj_row):
                    multiple_sols = self.check_multiple_solutions(tableau, basic_vars)
                    if self.num_vars == 2 and self.corner_points:
                        self.corner_points[-1]['is_optimal'] = True
                    if multiple_sols:
                        st.info(f"Multiple optimal solutions exist! Non-basic variables with zero coefficients: {', '.join(multiple_sols)}")
                    st.success("Optimal solution found!")
                    break
                pivot_col = np.argmin(obj_row) + 1
            else:
                if all(val <= 1e-6 for val in obj_row):
                    multiple_sols = self.check_multiple_solutions(tableau, basic_vars)
                    if self.num_vars == 2 and self.corner_points:
                        self.corner_points[-1]['is_optimal'] = True
                    if multiple_sols:
                        st.info(f"Multiple optimal solutions exist! Non-basic variables with zero coefficients: {', '.join(multiple_sols)}")
                    st.success("Optimal solution found!")
                    break
                pivot_col = np.argmax(obj_row) + 1
            
            # Calculate ratios for leaving variable
            ratios = []
            for i in range(len(tableau) - 1):
                if tableau[i, pivot_col] > 1e-6:
                    ratios.append(tableau[i, -1] / tableau[i, pivot_col])
                else:
                    ratios.append(float('inf'))
            
            if all(r == float('inf') for r in ratios):
                st.error("Problem is unbounded! No leaving variable found.")
                return None
            
            pivot_row = ratios.index(min(ratios))
            pivot_element = tableau[pivot_row, pivot_col]
            
            # Display pivot information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Entering:** {self.all_var_names[pivot_col-1] if pivot_col-1 < len(self.all_var_names) else f'var{pivot_col}'}")
            with col2:
                st.info(f"**Leaving:** Row {pivot_row + 1}")
            with col3:
                st.info(f"**Pivot:** {pivot_element:.3f}")
            
            # Perform pivot operation
            tableau[pivot_row, :] = tableau[pivot_row, :] / pivot_element
            
            for i in range(len(tableau)):
                if i != pivot_row:
                    multiplier = tableau[i, pivot_col]
                    tableau[i, :] = tableau[i, :] - multiplier * tableau[pivot_row, :]
            
            basic_vars[pivot_row] = pivot_col - 1
            
            # Update Z value
            z = 0
            for i, basic_var_idx in enumerate(basic_vars):
                if basic_var_idx < self.num_vars:
                    coef = self.c[basic_var_idx]
                    value = tableau[i, -1]
                    z += coef * value
            tableau[-1, 0] = z if self.maximize else -z
            
            st.markdown("---")
        
        return self.extract_solution(tableau, basic_vars)
    
    def create_tabular_iteration(self, tableau, iteration, basic_vars, pivot_col, ratios):
        """Create a formatted table for tabular iteration"""
        st.write(f"### Iteration {iteration}")
        
        data = []
        
        # Z row
        obj_row = {
            'Basic Var': 'Z',
            'Z': f"{tableau[-1, 0]:.3f}"
        }
        
        for j, var_name in enumerate(self.all_var_names):
            obj_row[var_name] = f"{tableau[-1, j+1]:.3f}"
        
        obj_row['RHS'] = f"{tableau[-1, -1]:.3f}"
        obj_row['Min Ratio'] = "-"
        
        data.append(obj_row)
        
        # Constraint rows
        for i in range(len(tableau) - 1):
            row_data = {
                'Basic Var': self.all_var_names[basic_vars[i]] if basic_vars[i] < len(self.all_var_names) else f'var{basic_vars[i]+1}',
                'Z': f"{tableau[i, 0]:.3f}"
            }
            
            for j, var_name in enumerate(self.all_var_names):
                row_data[var_name] = f"{tableau[i, j+1]:.3f}"
            
            row_data['RHS'] = f"{max(0, tableau[i, -1]):.3f}"
            
            if len(ratios) > i:
                if ratios[i] == float('inf'):
                    row_data['Min Ratio'] = "∞"
                elif ratios[i] == "-":
                    row_data['Min Ratio'] = "-"
                else:
                    row_data['Min Ratio'] = f"{ratios[i]:.3f}"
            else:
                row_data['Min Ratio'] = "-"
            
            data.append(row_data)
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    def extract_solution(self, tableau, basic_vars):
        """Extract the final solution from the tableau"""
        solution = {}
        
        for i in range(self.num_vars):
            var_name = f'x{i+1}'
            if i in basic_vars:
                idx = basic_vars.index(i)
                solution[var_name] = max(0, tableau[idx, -1])
            else:
                solution[var_name] = 0
        
        solution['Z'] = tableau[-1, 0]
        
        return solution
    
    def plot_feasible_region(self):
        """Plot feasible region and solution path for 2-variable problems"""
        if self.num_vars != 2:
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set up coordinate system
        x_max = max(20, max([cp['point'][0] for cp in self.corner_points if cp['point']] + [10]))
        y_max = max(20, max([cp['point'][1] for cp in self.corner_points if cp['point']] + [10]))
        
        x_range = np.linspace(0, x_max * 1.2, 400)
        y_range = np.linspace(0, y_max * 1.2, 400)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Plot constraint lines and feasible region
        feasible_region = None
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        for i, (constraint_type, rhs) in enumerate(zip(self.constraints, self.b)):
            a1, a2 = self.A[i][0], self.A[i][1]
            
            # Plot constraint line
            if abs(a2) > 1e-6:  # a2 != 0
                y_line = (rhs - a1 * x_range) / a2
                ax.plot(x_range, y_line, colors[i % len(colors)], 
                       label=f'{a1:.1f}x₁ + {a2:.1f}x₂ {constraint_type} {rhs:.1f}', linewidth=2)
            else:  # Vertical line
                x_line = rhs / a1
                ax.axvline(x=x_line, color=colors[i % len(colors)], 
                          label=f'{a1:.1f}x₁ {constraint_type} {rhs:.1f}', linewidth=2)
            
            # Create feasible region mask
            if constraint_type == '<=':
                if abs(a2) > 1e-6:
                    constraint_mask = (a1 * X + a2 * Y <= rhs + 1e-6)
                else:
                    constraint_mask = (a1 * X <= rhs + 1e-6)
            elif constraint_type == '>=':
                if abs(a2) > 1e-6:
                    constraint_mask = (a1 * X + a2 * Y >= rhs - 1e-6)
                else:
                    constraint_mask = (a1 * X >= rhs - 1e-6)
            else:  # '='
                if abs(a2) > 1e-6:
                    constraint_mask = (np.abs(a1 * X + a2 * Y - rhs) <= 1e-6)
                else:
                    constraint_mask = (np.abs(a1 * X - rhs) <= 1e-6)
            
            if feasible_region is None:
                feasible_region = constraint_mask
            else:
                feasible_region = feasible_region & constraint_mask
        
        # Add non-negativity constraints
        non_negative = (X >= 0) & (Y >= 0)
        if feasible_region is not None:
            feasible_region = feasible_region & non_negative
        else:
            feasible_region = non_negative
        
        # Plot feasible region
        ax.contourf(X, Y, feasible_region.astype(int), levels=[0.5, 1.5], 
                   colors=['lightblue'], alpha=0.3, label='Feasible Region')
        
        # Plot objective function contours
        if len(self.corner_points) > 0:
            optimal_point = None
            for cp in self.corner_points:
                if cp['is_optimal']:
                    optimal_point = cp['point']
                    break
            
            if optimal_point:
                optimal_z = self.c[0] * optimal_point[0] + self.c[1] * optimal_point[1]
                
                # Plot multiple objective function lines
                z_values = [optimal_z * 0.5, optimal_z * 0.75, optimal_z, optimal_z * 1.25]
                for z_val in z_values:
                    if abs(self.c[1]) > 1e-6:
                        y_obj = (z_val - self.c[0] * x_range) / self.c[1]
                        alpha = 0.8 if abs(z_val - optimal_z) < 1e-6 else 0.4
                        linewidth = 3 if abs(z_val - optimal_z) < 1e-6 else 1
                        ax.plot(x_range, y_obj, 'k--', alpha=alpha, linewidth=linewidth)
                    else:
                        ax.axvline(x=z_val/self.c[0], color='black', linestyle='--', alpha=0.4)
        
        # Plot corner points and solution path
        if self.corner_points:
            points = [cp['point'] for cp in self.corner_points if cp['point']]
            if points:
                # Plot solution path
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                ax.plot(x_coords, y_coords, 'ro-', markersize=8, linewidth=2, 
                       alpha=0.7, label='Simplex Path')
                
                # Annotate each point with iteration info
                for i, cp in enumerate(self.corner_points):
                    if cp['point']:
                        x, y = cp['point']
                        phase_info = cp.get('phase', 'Standard')
                        iteration = cp['iteration']
                        
                        # Determine point color and style
                        if cp['is_optimal']:
                            color = 'gold'
                            marker = '*'
                            size = 150
                            label = f"Optimal: ({x:.2f}, {y:.2f})"
                        else:
                            color = 'red'
                            marker = 'o'
                            size = 80
                            label = f"Iter {iteration}: ({x:.2f}, {y:.2f})"
                        
                        ax.scatter(x, y, c=color, marker=marker, s=size, 
                                 edgecolors='black', linewidth=1, zorder=5)
                        
                        # Add annotation with offset to avoid overlap
                        offset_x = x_max * 0.02
                        offset_y = y_max * 0.02
                        ax.annotate(f"{phase_info}\nIter {iteration}\n({x:.2f}, {y:.2f})", 
                                  (x, y), xytext=(x + offset_x, y + offset_y),
                                  bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                                  fontsize=8, ha='left')
        
        # Formatting
        ax.set_xlim(0, x_max * 1.1)
        ax.set_ylim(0, y_max * 1.1)
        ax.set_xlabel('x₁', fontsize=12, fontweight='bold')
        ax.set_ylabel('x₂', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Title with problem info
        obj_str = "Maximize" if self.maximize else "Minimize"
        title = f"{obj_str} Z = {self.c[0]:.1f}x₁ + {self.c[1]:.1f}x₂\n"
        title += f"Method: {self.method.replace('_', ' ').title()}"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig

def main():
    st.set_page_config(page_title="Simplex Solver", layout="wide")
    
    st.title(" Simplex Solver")
    st.markdown("*Supports Standard Form, Big M Method, and Two-Phase Method*")
    st.markdown("---")
    
    # Input Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Problem Setup")
        
        # Objective type
        obj_type = st.radio("Objective:", ["Maximize", "Minimize"])
        maximize = obj_type == "Maximize"
        
        # Number of variables and constraints
        num_vars = st.number_input("Number of Decision Variables:", min_value=1, max_value=10, value=2)
        num_constraints = st.number_input("Number of Constraints:", min_value=1, max_value=10, value=3)
    
    with col2:
        st.subheader("Solution Method")
        method = st.radio("Select Display Method:", ["Algebraic Simplex", "Tabular Simplex"])
        st.info("💡 Solver will automatically detect if Big M or Two-Phase method is needed")
    
    st.markdown("---")
    
    # Objective Function Input
    st.subheader("📊 Objective Function Coefficients")
    obj_cols = st.columns(num_vars)
    c = []
    for i in range(num_vars):
        with obj_cols[i]:
            coef = st.number_input(f"x{i+1}", value=1.0, key=f"c_{i}")
            c.append(coef)
    
    # Constraints Input
    st.subheader("Constraints")
    st.info("Note: All decision variables are automatically constrained to be ≥ 0")
    
    A = []
    b = []
    constraints = []
    
    for i in range(num_constraints):
        st.write(f"**Constraint {i+1}:**")
        cols = st.columns(num_vars + 2)
        
        row = []
        for j in range(num_vars):
            with cols[j]:
                coef = st.number_input(f"x{j+1}", value=1.0, key=f"a_{i}_{j}", label_visibility="collapsed")
                row.append(coef)
        
        with cols[num_vars]:
            constraint_type = st.selectbox("", ["<=", ">=", "="], key=f"const_{i}")
            constraints.append(constraint_type)
        
        with cols[num_vars + 1]:
            rhs = st.number_input("RHS", value=10.0, key=f"b_{i}", label_visibility="collapsed")
            b.append(rhs)
        
        A.append(row)
    
    st.markdown("---")
    
    if st.button("Solve Problem", type="primary"):
        try:
            solver = SimplexSolver(c, A, b, constraints, maximize)
            
            if not solver.method:
                st.warning("Please select a solution method first")
                return
            
            st.info(f"🔧 Using method: **{solver.method.replace('_', ' ').title()}**")
            
            if method == "Algebraic Simplex":
                solution = solver.solve_algebraic()
                
                if solution:
                    st.success("Solution Found!")
                    
                    # Display iterations based on method used
                    if solver.method == "big_m":
                        st.subheader("Big M Method - Algebraic Steps")
                        for iter_info in solver.algebraic_iterations:
                            with st.expander(f"Iteration {iter_info['iteration']}", expanded=(iter_info['iteration'] <= 2)):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**Basic Variables:**")
                                    for bv in iter_info['basic_vars']:
                                        st.write(f"• {bv}")
                                
                                with col2:
                                    st.write(f"**Z Value:** {iter_info['z_value']}")
                                
                                if 'pivot_info' in iter_info:
                                    st.write("**Pivot Information:**")
                                    st.write(f"• Entering: {iter_info['pivot_info']['col']}")
                                    st.write(f"• Leaving: Row {iter_info['pivot_info']['row']}")
                                    st.write(f"• Pivot Element: {iter_info['pivot_info']['element']:.3f}")
                                
                                if iter_info['is_final']:
                                    st.success("Optimal solution reached!")
                    
                    elif solver.method == "two_phase":
                        st.subheader("Two-Phase Method - Algebraic Steps")
                        
                        phase1_iters = [i for i in solver.phase_one_iterations if i['phase'] == 1]
                        phase2_iters = [i for i in solver.phase_one_iterations if i['phase'] == 2]
                        
                        if phase1_iters:
                            st.write("**Phase I Iterations:**")
                            for iter_info in phase1_iters:
                                with st.expander(f"Phase I - Iteration {iter_info['iteration']}", expanded=(iter_info['iteration'] <= 2)):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write("**Basic Variables:**")
                                        for bv in iter_info['basic_vars']:
                                            st.write(f"• {bv}")
                                    
                                    with col2:
                                        st.write(f"**Phase I Objective:** {iter_info['z_value']:.3f}")
                                    
                                    if 'pivot_info' in iter_info:
                                        st.write("**Pivot Information:**")
                                        st.write(f"• Entering: {iter_info['pivot_info']['col']}")
                                        st.write(f"• Leaving: Row {iter_info['pivot_info']['row']}")
                                        st.write(f"• Pivot Element: {iter_info['pivot_info']['element']:.3f}")
                        
                        if phase2_iters:
                            st.write("**Phase II Iterations:**")
                            for iter_info in phase2_iters:
                                with st.expander(f"Phase II - Iteration {iter_info['iteration']}", expanded=(iter_info['iteration'] <= 2)):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write("**Basic Variables:**")
                                        for bv in iter_info['basic_vars']:
                                            st.write(f"• {bv}")
                                    
                                    with col2:
                                        st.write(f"**Phase II Objective:** {iter_info['z_value']:.3f}")
                                    
                                    if 'pivot_info' in iter_info:
                                        st.write("**Pivot Information:**")
                                        st.write(f"• Entering: {iter_info['pivot_info']['col']}")
                                        st.write(f"• Leaving: Row {iter_info['pivot_info']['row']}")
                                        st.write(f"• Pivot Element: {iter_info['pivot_info']['element']:.3f}")
                    
                    else:  # Standard form
                        st.subheader("Standard Simplex - Algebraic Steps")
                        for iter_info in solver.algebraic_iterations:
                            with st.expander(f"Iteration {iter_info['iteration']}", expanded=(iter_info['iteration'] <= 2)):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**Basic Variables:**")
                                    for bv in iter_info['basic_vars']:
                                        st.write(f"• {bv}")
                                
                                with col2:
                                    st.write(f"**Z Value:** {iter_info['z_value']:.3f}")
                                
                                if 'pivot_info' in iter_info:
                                    st.write("**Pivot Information:**")
                                    st.write(f"• Entering: {iter_info['pivot_info']['col']}")
                                    st.write(f"• Leaving: Row {iter_info['pivot_info']['row']}")
                                    st.write(f"• Pivot Element: {iter_info['pivot_info']['element']:.3f}")
                                
                                if 'multiple_solutions' in iter_info and iter_info['multiple_solutions']:
                                    st.info(f"Multiple optimal solutions detected! Non-basic variables with zero coefficients: {', '.join(iter_info['multiple_solutions'])}")
                                
                                if iter_info['is_final']:
                                    st.success("Optimal solution reached!")
            
            else:  # Tabular method
                solution = solver.solve_tabular()
                
                if solution:
                    st.success("Solution Found!")
            
            # Display final solution
            if solution:
                st.markdown("---")
                st.subheader("Final Solution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Decision Variables:**")
                    for i in range(num_vars):
                        var_name = f'x{i+1}'
                        st.write(f"• {var_name} = {solution[var_name]:.3f}")
                
                with col2:
                    obj_type_str = "Maximum" if maximize else "Minimum"
                    st.write(f"**{obj_type_str} Z Value:** {solution['Z']:.3f}")
                
                # Plot feasible region for 2-variable problems
                if num_vars == 2 and len(solver.corner_points) > 0:
                    st.markdown("---")
                    st.subheader("Graphical Representation")
                    fig = solver.plot_feasible_region()
                    st.pyplot(fig)
                    
                    # Display corner points summary
                    st.subheader("Corner Points Visited")
                    corner_data = []
                    for cp in solver.corner_points:
                        if cp['point']:
                            x, y = cp['point']
                            z_val = solver.c[0] * x + solver.c[1] * y
                            status = "OPTIMAL" if cp['is_optimal'] else "Visited"
                            corner_data.append({
                                'Iteration': cp['iteration'],
                                'Phase/Method': cp['phase'],
                                'Point (x₁, x₂)': f"({x:.3f}, {y:.3f})",
                                'Z Value': f"{z_val:.3f}",
                                'Status': status
                            })
                    
                    if corner_data:
                        df_corners = pd.DataFrame(corner_data)
                        st.dataframe(df_corners, use_container_width=True, hide_index=True)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please check your input values and try again.")

if __name__ == "__main__":
    main()


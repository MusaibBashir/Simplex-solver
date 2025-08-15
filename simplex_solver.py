import streamlit as st
import numpy as np
import pandas as pd
from fractions import Fraction
import copy

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
        
        # Setup the problem
        self.setup_problem()
    
    def setup_problem(self):
        """Convert to standard form by adding slack, surplus, and artificial variables"""
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
        
        # Fill original constraint coefficients
        for i in range(self.num_constraints):
            for j in range(self.num_vars):
                self.tableau[i, j+1] = self.A[i][j]
            self.tableau[i, -1] = self.b[i]  # RHS
        
        # Fill objective function coefficients (negative for maximization)
        for j in range(self.num_vars):
            if self.maximize:
                self.tableau[-1, j+1] = -self.c[j]
            else:
                self.tableau[-1, j+1] = self.c[j]
        
        # Add slack, surplus, and artificial variables
        col_idx = self.num_vars + 1
        M = 1000  # Big M
        
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
                self.tableau[-1, col_idx] = M if self.maximize else -M
                self.basic_vars.append(col_idx - 1)
                col_idx += 1
                
            else: 
                # Add artificial variable
                artificial_var = f'a{len(self.artificial_vars)+1}'
                self.artificial_vars.append(artificial_var)
                self.all_var_names.append(artificial_var)
                self.tableau[i, col_idx] = 1
                self.tableau[-1, col_idx] = M if self.maximize else -M 
                self.basic_vars.append(col_idx - 1)
                col_idx += 1
        
        # Calculate initial Z value by eliminating artificial variables from objective row
        self.update_objective_row()
    
    def update_objective_row(self):
        """Update objective row to account for basic artificial variables"""
        for i, basic_var_idx in enumerate(self.basic_vars):
            if basic_var_idx >= self.num_vars + len(self.slack_vars) + len(self.surplus_vars):
                multiplier = self.tableau[-1, basic_var_idx + 1]
                if multiplier != 0:
                    self.tableau[-1, :] -= multiplier * self.tableau[i, :]
        
        # Update Z value
        self.calculate_z_value()
    
    def calculate_z_value(self):
        """Calculate current Z value from basic variables"""
        z = 0
        for i, basic_var_idx in enumerate(self.basic_vars):
            if basic_var_idx < self.num_vars: 
                coef = self.c[basic_var_idx]
                value = self.tableau[i, -1]
                z += coef * value
        
        if self.maximize:
            self.tableau[-1, 0] = z
        else:
            self.tableau[-1, 0] = -z
    
    def solve_algebraic(self):
        """Solve using algebraic method with detailed steps"""
        tableau = copy.deepcopy(self.tableau)
        iteration = 0
        
        while True:
            iteration += 1
            
            # Check for negative RHS values
            for i in range(len(tableau) - 1):
                if tableau[i, -1] < 0:
                    st.warning(f"Warning: Negative RHS in row {i+1}: {tableau[i, -1]:.2f}")
            
            # Find entering variable
            obj_row = tableau[-1, 1:-1]
            
            if self.maximize:
                if all(val >= -0.0001 for val in obj_row): 
                    self.record_algebraic_iteration(tableau, iteration, self.basic_vars, is_final=True)
                    break
                pivot_col = np.argmin(obj_row) + 1
            else:
                if all(val <= 0.0001 for val in obj_row):
                    self.record_algebraic_iteration(tableau, iteration, self.basic_vars, is_final=True)
                    break
                pivot_col = np.argmax(obj_row) + 1
            
            # Find leaving variable
            ratios = []
            for i in range(len(tableau) - 1):
                if tableau[i, pivot_col] > 0.0001: 
                    ratios.append((tableau[i, -1] / tableau[i, pivot_col], i))
                else:
                    ratios.append((float('inf'), i))
            
            if all(r[0] == float('inf') for r in ratios):
                st.error("Problem is unbounded!")
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
            
            if self.maximize:
                tableau[-1, 0] = z
            else:
                tableau[-1, 0] = -z
        
        return self.extract_solution(tableau, self.basic_vars)
    
    def solve_tabular(self):
        """Solve using tabular method with formatted tables"""
        tableau = copy.deepcopy(self.tableau)
        iteration = 0
        basic_vars = copy.deepcopy(self.basic_vars)
        
        while True:
            iteration += 1
            obj_row = tableau[-1, 1:-1]
            
            if self.maximize:
                if all(val >= -0.0001 for val in obj_row):
                    self.create_tabular_iteration(tableau, iteration, basic_vars, None, [])
                    break
                pivot_col = np.argmin(obj_row) + 1
            else:
                if all(val <= 0.0001 for val in obj_row):
                    self.create_tabular_iteration(tableau, iteration, basic_vars, None, [])
                    break
                pivot_col = np.argmax(obj_row) + 1
            
            # Find leaving variable and calculate all ratios
            ratios = []
            for i in range(len(tableau) - 1):
                if tableau[i, pivot_col] > 0.0001:
                    ratios.append(tableau[i, -1] / tableau[i, pivot_col])
                else:
                    ratios.append(float('inf'))
            
            if all(r == float('inf') for r in ratios):
                st.error("Problem is unbounded!")
                return None
            
            pivot_row = ratios.index(min(ratios))
            pivot_element = tableau[pivot_row, pivot_col]
            
            # Create iteration table with ratios
            self.create_tabular_iteration(tableau, iteration, basic_vars, pivot_col, ratios)
            
            # Store pivot information
            self.tabular_iterations[-1]['pivot_col'] = self.all_var_names[pivot_col - 1] if pivot_col - 1 < len(self.all_var_names) else f'var{pivot_col}'
            self.tabular_iterations[-1]['pivot_row'] = pivot_row + 1
            self.tabular_iterations[-1]['pivot_element'] = pivot_element
            
            # Perform pivot operation
            tableau[pivot_row, :] = tableau[pivot_row, :] / pivot_element
            
            for i in range(len(tableau)):
                if i != pivot_row:
                    multiplier = tableau[i, pivot_col]
                    tableau[i, :] = tableau[i, :] - multiplier * tableau[pivot_row, :]
            
            # Update basic variables
            basic_vars[pivot_row] = pivot_col - 1
            
            # Recalculate Z value
            z = 0
            for i, basic_var_idx in enumerate(basic_vars):
                if basic_var_idx < self.num_vars:
                    coef = self.c[basic_var_idx]
                    value = tableau[i, -1]
                    z += coef * value
            
            if self.maximize:
                tableau[-1, 0] = z
            else:
                tableau[-1, 0] = -z
        
        return self.extract_solution(tableau, basic_vars)
    
    def record_algebraic_iteration(self, tableau, iteration, basic_vars, pivot_row=None, pivot_col=None, pivot_element=None, is_final=False):
        """Record algebraic iteration details"""
        basic_vars_info = []
        for i, bv_idx in enumerate(basic_vars):
            if bv_idx < len(self.all_var_names):
                var_name = self.all_var_names[bv_idx]
                value = max(0, tableau[i, -1])  
                basic_vars_info.append(f"{var_name} = {value:.2f}")
        
        z_value = tableau[-1, 0]
        
        iteration_info = {
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
            
            # Add calculations
            calculations = []
            calculations.append(f"Entering variable: {iteration_info['pivot_info']['col']}")
            calculations.append(f"Leaving variable: Row {iteration_info['pivot_info']['row']}")
            calculations.append(f"Pivot element: {pivot_element:.2f}")
            iteration_info['calculations'] = calculations
        
        self.algebraic_iterations.append(iteration_info)
    
    def create_tabular_iteration(self, tableau, iteration, basic_vars, pivot_col, ratios):
        """Create a formatted table for tabular iteration"""
        data = []
        
        obj_row = {
            'Iteration': iteration,
            'Eqn #': 'Z',
            'Basic Var': 'Z',
            'Z': f"{tableau[-1, 0]:.2f}"
        }
        
        for j, var_name in enumerate(self.all_var_names):
            obj_row[var_name] = f"{tableau[-1, j+1]:.2f}"
        
        obj_row['RHS'] = f"{tableau[-1, -1]:.2f}"
        obj_row['Min Ratio'] = "-"
        
        data.append(obj_row)
        
        for i in range(len(tableau) - 1):
            row_data = {
                'Iteration': iteration,
                'Eqn #': i + 1,
                'Basic Var': self.all_var_names[basic_vars[i]] if basic_vars[i] < len(self.all_var_names) else f'var{basic_vars[i]+1}',
                'Z': f"{tableau[i, 0]:.2f}"
            }
            
            for j, var_name in enumerate(self.all_var_names):
                row_data[var_name] = f"{tableau[i, j+1]:.2f}"
            
            row_data['RHS'] = f"{max(0, tableau[i, -1]):.2f}" 
            
            if len(ratios) > i:
                if ratios[i] == float('inf'):
                    row_data['Min Ratio'] = "∞"
                else:
                    row_data['Min Ratio'] = f"{ratios[i]:.2f}"
            else:
                row_data['Min Ratio'] = "-"
            
            data.append(row_data)
        
        self.tabular_iterations.append({
            'iteration': iteration,
            'data': data,
            'pivot_col': None,
            'pivot_row': None,
            'pivot_element': None
        })
    
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

def main():
    st.set_page_config(page_title="Simplex Method Solver", layout="wide")
    
    st.title("Linear Programming - Simplex Method Solver")
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
        method = st.radio("Select Method:", ["Algebraic Simplex", "Tabular Simplex"])
    
    st.markdown("---")
    
    # Objective Function Input
    st.subheader("Objective Function Coefficients")
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
    
    if st.button("Solve", type="primary"):
        try:
            solver = SimplexSolver(c, A, b, constraints, maximize)
            
            if method == "Algebraic Simplex":
                solution = solver.solve_algebraic()
                
                if solution:
                    st.success("Solution Found!")
                    
                    st.subheader("Algebraic Solution Steps")
                    
                    for iter_info in solver.algebraic_iterations:
                        with st.expander(f"Iteration {iter_info['iteration']}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Basic Variables:**")
                                for bv in iter_info['basic_vars']:
                                    st.write(f"• {bv}")
                            
                            with col2:
                                st.write(f"**Z Value:** {iter_info['z_value']:.2f}")
                            
                            if 'calculations' in iter_info:
                                st.write("**Calculations:**")
                                for calc in iter_info['calculations']:
                                    st.write(f"• {calc}")
                            
                            if iter_info['is_final']:
                                st.success("Optimal solution reached!")
                
                    st.subheader("Final Solution")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Decision Variables:**")
                        for var, val in solution.items():
                            if var != 'Z':
                                st.metric(var, f"{val:.2f}")
                    
                    with col2:
                        st.write("**Optimal Value:**")
                        st.metric(f"{'Maximum' if maximize else 'Minimum'} Z", f"{solution['Z']:.2f}")
            
            else:
                solution = solver.solve_tabular()
                
                if solution:
                    st.success("Solution Found!")
                    
                    st.subheader("Tabular Solution Steps")
                    
                    for iter_data in solver.tabular_iterations:
                        st.write(f"### Iteration {iter_data['iteration']}")
                        
                        df = pd.DataFrame(iter_data['data'])
                        
                        styled_df = df.style.set_properties(**{
                            'border': '1px solid black',
                            'text-align': 'center'
                        }).set_table_styles([
                            {'selector': 'th', 'props': [('border', '1px solid black'), 
                                                        ('padding', '8px'), 
                                                        ('background-color', '#4CAF50'),
                                                        ('color', 'white'),
                                                        ('text-align', 'center')]},
                            {'selector': 'td', 'props': [('border', '1px solid black'), 
                                                        ('padding', '8px'),
                                                        ('text-align', 'center')]},
                            {'selector': 'tr:first-child', 'props': [('background-color', '#e8f5e9')]}
                        ])
                        
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        
                        if iter_data['pivot_col']:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.info(f"**Pivot Column:** {iter_data['pivot_col']}")
                            with col2:
                                st.info(f"**Pivot Row:** {iter_data['pivot_row']}")
                            with col3:
                                st.info(f"**Pivot Element:** {iter_data['pivot_element']:.2f}")
                        
                        st.markdown("---")

                    st.subheader("Final Solution")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Decision Variables:**")
                        for var, val in solution.items():
                            if var != 'Z':
                                st.metric(var, f"{val:.2f}")
                    
                    with col2:
                        st.write("**Optimal Value:**")
                        st.metric(f"{'Maximum' if maximize else 'Minimum'} Z", f"{solution['Z']:.2f}")
        
        except Exception as e:
            st.error(f"Error solving the problem: {str(e)}")
            st.write("Debug info:", str(e))
    
    # Instructions
    with st.sidebar:
        st.header("📖 Instructions")
        st.markdown("""
        1. **Set up your problem:**
           - Choose to Maximize or Minimize
           - Enter number of variables and constraints
           
        2. **Enter coefficients:**
           - Input objective function coefficients
           - Input constraint coefficients and RHS values
           - Select constraint type (≤, ≥, =)
           
        3. **Choose solution method:**
           - **Algebraic Simplex:** Shows calculations and basic variables at each iteration
           - **Tabular Simplex:** Displays complete tableau with pivot information
           
        4. **Click Solve** to get the optimal solution!
        
        **Notes:** 
        - All decision variables are automatically constrained to be ≥ 0
        - The solver automatically handles slack, surplus, and artificial variables
        - Z row appears at the top of each iteration table
        - Min ratios are displayed for all constraint rows
        """)

if __name__ == "__main__":
    main()
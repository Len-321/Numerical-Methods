import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Set UI Theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class MathApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Advanced Numerical & Matrix Solver")
        self.geometry("1200x750")

        # --- Layout ---
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(padx=20, pady=20, fill="both", expand=True)

        self.tab_roots = self.tabview.add("Root Finding")
        self.tab_matrix = self.tabview.add("Matrix Operations")

        self.setup_roots_ui()
        self.setup_matrix_ui()

    # ==========================================
    # ROOT FINDING LOGIC & UI
    # ==========================================
    def setup_roots_ui(self):
        # Left Panel: Inputs
        input_frame = ctk.CTkFrame(self.tab_roots, width=300)
        input_frame.pack(side="left", fill="y", padx=10, pady=10)

        ctk.CTkLabel(input_frame, text="Equation (use 'x'):").pack(pady=5)
        self.eq_entry = ctk.CTkEntry(input_frame, placeholder_text="3*x + sin(x) - exp(x)")
        self.eq_entry.pack(pady=5, padx=10, fill="x")

        ctk.CTkLabel(input_frame, text="Method:").pack(pady=5)
        self.root_method = ctk.CTkOptionMenu(input_frame, values=[
            "Incremental", "Bisection", "Regula-Falsi", "Newton-Raphson", "Secant"
        ])
        self.root_method.pack(pady=5, padx=10, fill="x")

        # Bounds input
        bounds_frame = ctk.CTkFrame(input_frame)
        bounds_frame.pack(pady=10, padx=10, fill="x")

        ctk.CTkLabel(bounds_frame, text="XL (Lower):").pack(side="left", padx=5)
        self.xl_entry = ctk.CTkEntry(bounds_frame, width=80, placeholder_text="-0.5")
        self.xl_entry.pack(side="left", padx=5)

        ctk.CTkLabel(bounds_frame, text="Xu (Upper):").pack(side="left", padx=5)
        self.xu_entry = ctk.CTkEntry(bounds_frame, width=80, placeholder_text="1")
        self.xu_entry.pack(side="left", padx=5)

        # Tolerance input
        tol_frame = ctk.CTkFrame(input_frame)
        tol_frame.pack(pady=5, padx=10, fill="x")

        ctk.CTkLabel(tol_frame, text="Tolerance:").pack(side="left", padx=5)
        self.tolerance_entry = ctk.CTkEntry(tol_frame, width=80, placeholder_text="0.001")
        self.tolerance_entry.pack(side="left", padx=5)

        self.btn_solve_root = ctk.CTkButton(input_frame, text="Solve & Graph", command=self.solve_roots)
        self.btn_solve_root.pack(pady=20)

        # Right Panel: Output and Graph
        right_frame = ctk.CTkFrame(self.tab_roots)
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Iteration table (Using monospace font for alignment)
        self.table_output = ctk.CTkTextbox(right_frame, height=220, width=500, font=("Consolas", 11))
        self.table_output.pack(pady=5, padx=5, fill="x")

        # Final root label
        self.final_root_label = ctk.CTkLabel(right_frame, text="Final Root: Not calculated", font=("Arial", 14, "bold"))
        self.final_root_label.pack(pady=5)

        # Graph area
        self.plot_frame = ctk.CTkFrame(right_frame)
        self.plot_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def solve_roots(self):
        func_str = self.eq_entry.get()
        method = self.root_method.get()
        
        if not func_str:
            messagebox.showerror("Error", "Please enter an equation")
            return
        
        try:
            xl = float(self.xl_entry.get())
            xu = float(self.xu_entry.get())
            tolerance = float(self.tolerance_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values for XL, Xu, and Tolerance")
            return
        
        # Pre-process the function string for common mathematical notations
        func_str = func_str.replace('^', '**')
        import re
        func_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', func_str)
        
        try:
            # Inject numpy functions safely into evaluation scope if needed
            f = lambda x: eval(func_str, {"x": x, "np": np, "exp": np.exp, "sin": np.sin, "cos": np.cos, "tan": np.tan, "log": np.log})
            test_val = f(xl)
            
            iterations = []
            root = None
            
            if method == "Incremental":
                root, iterations = self.incremental_method(f, xl, xu, tol=tolerance)
            elif method == "Bisection":
                root, iterations = self.bisection_method(f, xl, xu, tol=tolerance)
            elif method == "Regula-Falsi":
                root, iterations = self.regula_falsi_method(f, xl, xu, tol=tolerance)
            elif method == "Newton-Raphson":
                # FIX: Change from (xl + xu) / 2 to xl to start tracking from the left bound
                root, iterations = self.newton_raphson_method(f, xl, tol=tolerance)
            elif method == "Secant":
                root, iterations = self.secant_method(f, xl, xu, tol=tolerance)
            
            # Display iteration table
            self.display_iterations(iterations, method)
            
            # Display graph with root
            x_vals = np.linspace(xl - 1, xu + 1, 400)
            y_vals = [f(x) for x in x_vals]
            self.display_plot(x_vals, y_vals, func_str, root)
            
            if root is not None:
                self.final_root_label.configure(text=f"Final Root: {root:.6f}")
                messagebox.showinfo("Success", f"Root found: {root:.6f} using {method}")
            else:
                self.final_root_label.configure(text="Final Root: Not found")
                messagebox.showwarning("Warning", f"No root found using {method}. Try different bounds/guesses.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Invalid Equation or calculation error: {e}")

    # --- INCREMENTAL SEARCH ---
    def incremental_method(self, f, xl, xu, tol=1e-4, max_iter=100):
        iterations = []
        dx = 0.5 
        if (xu - xl) < dx: dx = (xu - xl) / 2
        x_l_curr = xl
        
        for i in range(max_iter):
            x_u_curr = x_l_curr + dx
            f_xl = f(x_l_curr)
            f_xu = f(x_u_curr)
            prod = f_xl * f_xu
            
            if prod > 0:
                remark = "Go to next interval"
                iterations.append((i+1, x_l_curr, dx, x_u_curr, f_xl, f_xu, prod, remark))
                x_l_curr = x_u_curr
            else:
                remark = "Revert & consider smaller interval"
                iterations.append((i+1, x_l_curr, dx, x_u_curr, f_xl, f_xu, prod, remark))
                if abs(dx - 0.5) < 1e-7: dx = 0.1
                elif abs(dx - 0.1) < 1e-7: dx = 0.05
                elif abs(dx - 0.05) < 1e-7: dx = 0.01
                elif abs(dx - 0.01) < 1e-7: dx = 0.001
                else: dx /= 2

            if abs(f_xu) < tol or dx < tol: return x_u_curr, iterations
            if x_l_curr > xu: break
                
        return x_l_curr, iterations

    # --- BISECTION METHOD ---
    def bisection_method(self, f, xl, xu, tol=1e-4, max_iter=100):
        iterations = []
        if f(xl) * f(xu) > 0: return None, iterations
        
        xr_old = None
        for i in range(max_iter):
            xr = (xl + xu) / 2
            f_xl = f(xl)
            f_xr = f(xr)
            prod = f_xl * f_xr
            
            if xr_old is not None and xr != 0:
                ea = abs((xr - xr_old) / xr) * 100
                ea_str = f"{ea:.4f}%"
            else: ea_str = "-"
            
            remark = "2nd subinterval" if prod > 0 else "1st subinterval"
            iterations.append((i+1, xl, xr, xu, f_xl, f_xr, ea_str, prod, remark))
            
            if abs(f_xr) < tol or (xu - xl) / 2 < tol: return xr, iterations
            
            xr_old = xr
            if prod < 0: xu = xr
            else: xl = xr
        
        return xr, iterations

    # --- REGULA-FALSI (FALSE POSITION) METHOD ---
    def regula_falsi_method(self, f, xl, xu, tol=1e-4, max_iter=100):
        iterations = []
        if f(xl) * f(xu) > 0:
            return None, iterations
        
        xr_old = None
        for i in range(1, max_iter + 1):
            f_xl = f(xl)
            f_xu = f(xu)
            
            # XR Formula
            xr = (xu * f_xl - xl * f_xu) / (f_xl - f_xu)
            f_xr = f(xr)
            prod = f_xl * f_xr
            
            # Approximate relative error
            if xr_old is not None and xr != 0:
                ea = abs((xr - xr_old) / xr)
                ea_str = f"{ea:.4f}"
            else:
                ea = float('inf')
                ea_str = ""
            
            # Check conditions to shift bounds
            if prod < 0:
                remark = "<0 xR=xU"
            elif prod > 0:
                remark = ">0 xR=xL"
            else:
                remark = "=0 Root Found"
                
            iterations.append((i, xl, xu, xr, ea_str, f_xl, f_xu, f_xr, prod, remark))
            
            if abs(f_xr) < tol or ea < tol:
                return xr, iterations
                
            xr_old = xr
            if prod < 0:
                xu = xr
            else:
                xl = xr
                
        return xr, iterations

    # --- NEWTON-RAPHSON METHOD ---
    def newton_raphson_method(self, f, x0, tol=1e-4, max_iter=100):
        iterations = []
        x = x0
        h = 1e-6  # Tiny step for derivative approximation
        
        # Add 0th initial iteration from PDF Layout
        fx = f(x)
        dfx = (f(x + h) - f(x - h)) / (2 * h)
        iterations.append((0, x, "", fx, dfx))
        
        for i in range(1, max_iter + 1):
            if abs(dfx) < 1e-10: return None, iterations
            
            # Newton Raphson Formula
            x_new = x - fx / dfx
            
            # Approximate relative error
            ea = abs((x_new - x) / x_new)
            
            fx_new = f(x_new)
            dfx_new = (f(x_new + h) - f(x_new - h)) / (2 * h)
            
            iterations.append((i, x_new, f"{ea:.3f}", fx_new, dfx_new))
            
            if ea < tol or abs(fx_new) < tol:
                return x_new, iterations
                
            x = x_new
            fx = fx_new
            dfx = dfx_new
            
        return x, iterations

    # --- SECANT METHOD ---
    def secant_method(self, f, xl, xu, tol=1e-4, max_iter=100):
        iterations = []
        x_prev = xl # Maps to x_{i-1}
        x_curr = xu # Maps to x_i
        
        for i in range(1, max_iter + 1):
            f_prev = f(x_prev)
            f_curr = f(x_curr)
            
            if abs(f_curr - f_prev) < 1e-10:
                break
            
            # Secant Formula
            x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            f_next = f(x_next)
            
            # Approximate relative error
            if i == 1:
                ea_str = ""
                ea = float('inf')
            else:
                ea = abs((x_next - x_curr) / x_next)
                ea_str = f"{ea:.3f}"
                
            iterations.append((i, x_prev, x_curr, x_next, ea_str, f_prev, f_curr, f_next))
            
            if abs(f_next) < tol or ea < tol:
                return x_next, iterations
                
            x_prev = x_curr
            x_curr = x_next
            
        return x_curr, iterations

    def display_iterations(self, iterations, method):
        self.table_output.delete("1.0", tk.END)
        if not iterations:
            self.table_output.insert(tk.END, "No iterations recorded.")
            return
        
        if method == "Incremental":
            header = f"{'Iter':<5} {'xl':<8} {'dx':<8} {'xu':<8} {'f(xl)':<10} {'f(xu)':<10} {'f(xl)*f(xu)':<12} {'Remark'}\n"
            header += "-" * 90 + "\n"
            self.table_output.insert(tk.END, header)
            for data in iterations:
                i, xl, dx, xu, f_xl, f_xu, prod, remark = data
                line = f"{i:<5} {xl:<8.4f} {dx:<8.4f} {xu:<8.4f} {f_xl:<10.4f} {f_xu:<10.4f} {prod:<12.4f} {remark}\n"
                self.table_output.insert(tk.END, line)
                
        elif method == "Bisection":
            header = f"{'Iter':<5} {'xl':<8} {'xr':<8} {'xu':<8} {'f(xl)':<10} {'f(xr)':<10} {'|ea|':<10} {'f(xl)*f(xr)':<12} {'Remark'}\n"
            header += "-" * 95 + "\n"
            self.table_output.insert(tk.END, header)
            for data in iterations:
                i, xl, xr, xu, f_xl, f_xr, ea_str, prod, remark = data
                line = f"{i:<5} {xl:<8.4f} {xr:<8.4f} {xu:<8.4f} {f_xl:<10.4f} {f_xr:<10.4f} {ea_str:<10} {prod:<12.4f} {remark}\n"
                self.table_output.insert(tk.END, line)
                
        elif method == "Regula-Falsi":
            header = f"{'Iter':<5} {'xL':<8} {'xU':<8} {'xR':<8} {'Ea':<8} {'f(xL)':<10} {'f(xU)':<10} {'f(xR)':<10} {'Condition'}\n"
            header += "-" * 90 + "\n"
            self.table_output.insert(tk.END, header)
            for data in iterations:
                i, xl, xu, xr, ea, f_xl, f_xu, f_xr, prod, remark = data
                line = f"{i:<5} {xl:<8.4f} {xu:<8.4f} {xr:<8.4f} {ea:<8} {f_xl:<10.4f} {f_xu:<10.4f} {f_xr:<10.4f} {remark}\n"
                self.table_output.insert(tk.END, line)

        elif method == "Newton-Raphson":
            header = f"{'Iter':<5} {'xi':<10} {'Ea':<10} {'f(xi)':<12} {'f\'(xi)':<12}\n"
            header += "-" * 60 + "\n"
            self.table_output.insert(tk.END, header)
            for data in iterations:
                i, xi, ea, fxi, dfxi = data
                line = f"{i:<5} {xi:<10.4f} {ea:<10} {fxi:<12.4f} {dfxi:<12.4f}\n"
                self.table_output.insert(tk.END, line)

        elif method == "Secant":
            header = f"{'Iter':<5} {'x_{i-1}':<10} {'x_i':<10} {'x_{i+1}':<10} {'Ea':<8} {'f(x_{i-1})':<12} {'f(x_i)':<12} {'f(x_{i+1})':<12}\n"
            header += "-" * 95 + "\n"
            self.table_output.insert(tk.END, header)
            for data in iterations:
                i, x_prev, x_curr, x_next, ea, f_prev, f_curr, f_next = data
                line = f"{i:<5} {x_prev:<10.4f} {x_curr:<10.4f} {x_next:<10.4f} {ea:<8} {f_prev:<12.4f} {f_curr:<12.4f} {f_next:<12.4f}\n"
                self.table_output.insert(tk.END, line)

    def display_plot(self, x, y, title, root=None):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=100)
        ax.plot(x, y, label=f"f(x) = {title}")
        ax.axhline(0, color='black', lw=1)
        ax.grid(True, linestyle='--')
        ax.legend()
        
        if root is not None:
            ax.plot(root, 0, 'ro', markersize=8, label=f'Root: {root:.4f}')
            ax.legend()
        
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ==========================================
    # MATRIX LOGIC & UI (Unchanged)
    # ==========================================
    def setup_matrix_ui(self):
        main_frame = ctk.CTkFrame(self.tab_matrix)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Matrix A Input (3x3 Grid)
        ctk.CTkLabel(main_frame, text="Matrix A (3x3):").pack(pady=5)
        mat_a_frame = ctk.CTkFrame(main_frame)
        mat_a_frame.pack(pady=5)
        self.mat_a_entries = []
        for i in range(3):
            row_entries = []
            for j in range(3):
                entry = ctk.CTkEntry(mat_a_frame, width=60, height=40)
                entry.grid(row=i, column=j, padx=2, pady=2)
                row_entries.append(entry)
            self.mat_a_entries.append(row_entries)

        # Matrix B Input (3x3 Grid)
        ctk.CTkLabel(main_frame, text="Matrix B (3x3):").pack(pady=5)
        mat_b_frame = ctk.CTkFrame(main_frame)
        mat_b_frame.pack(pady=5)
        self.mat_b_entries = []
        for i in range(3):
            row_entries = []
            for j in range(3):
                entry = ctk.CTkEntry(mat_b_frame, width=60, height=40)
                entry.grid(row=i, column=j, padx=2, pady=2)
                row_entries.append(entry)
            self.mat_b_entries.append(row_entries)

        # Power input
        power_frame = ctk.CTkFrame(main_frame)
        power_frame.pack(pady=5)
        ctk.CTkLabel(power_frame, text="Power (for Matrix A):").pack(side="left", padx=5)
        self.power_entry = ctk.CTkEntry(power_frame, width=80, placeholder_text="2")
        self.power_entry.pack(side="left", padx=5)

        # Equation input
        eq_frame = ctk.CTkFrame(main_frame)
        eq_frame.pack(pady=5)
        ctk.CTkLabel(eq_frame, text="Equation (e.g., A**2 + 2*A + B):").pack(side="left", padx=5)
        self.matrix_eq_entry = ctk.CTkEntry(eq_frame, width=300, placeholder_text="A**2 + 2*A + B")
        self.matrix_eq_entry.pack(side="left", padx=5)

        btns_frame = ctk.CTkFrame(main_frame)
        btns_frame.pack(pady=10)

        ops = [
            ("Add", self.mat_add), ("Multiply", self.mat_mul), 
            ("Inverse", self.mat_inv), ("Det", self.mat_det),
            ("Transpose", self.mat_trans), ("Adjoint", self.mat_adj),
            ("Power", self.mat_power), ("Evaluate Eq", self.evaluate_matrix_equation)
        ]

        for text, cmd in ops:
            ctk.CTkButton(btns_frame, text=text, width=100, command=cmd).pack(side="left", padx=5)

        self.mat_output = ctk.CTkTextbox(main_frame, height=200, width=500)
        self.mat_output.pack(pady=20)

    def get_matrix(self, matrix_entries):
        matrix = []
        for i in range(3):
            row = []
            for j in range(3):
                try:
                    value = float(matrix_entries[i][j].get())
                    row.append(value)
                except ValueError:
                    messagebox.showerror("Input Error", "Invalid input. Please enter numbers only in all matrix cells.")
                    return None
            matrix.append(row)
        return np.array(matrix)

    def mat_add(self):
        A, B = self.get_matrix(self.mat_a_entries), self.get_matrix(self.mat_b_entries)
        if A is not None and B is not None: self.show_mat_res(A + B)

    def mat_mul(self):
        A, B = self.get_matrix(self.mat_a_entries), self.get_matrix(self.mat_b_entries)
        if A is not None and B is not None: self.show_mat_res(np.matmul(A, B))

    def mat_inv(self):
        A = self.get_matrix(self.mat_a_entries)
        if A is not None:
            try: self.show_mat_res(np.linalg.inv(A))
            except: messagebox.showerror("Error", "Matrix is Singular")

    def mat_det(self):
        A = self.get_matrix(self.mat_a_entries)
        if A is not None: self.show_mat_res(np.linalg.det(A))

    def mat_trans(self):
        A = self.get_matrix(self.mat_a_entries)
        if A is not None: self.show_mat_res(A.T)

    def mat_adj(self):
        A = self.get_matrix(self.mat_a_entries)
        if A is not None:
            adj = np.linalg.inv(A) * np.linalg.det(A)
            self.show_mat_res(np.round(adj))

    def mat_power(self):
        A = self.get_matrix(self.mat_a_entries)
        if A is not None:
            try:
                power = int(self.power_entry.get())
                self.show_mat_res(np.linalg.matrix_power(A, power))
            except Exception as e:
                messagebox.showerror("Error", f"Error computing matrix power: {e}")

    def evaluate_matrix_equation(self):
        eq_str = self.matrix_eq_entry.get()
        needs_A = 'A' in eq_str
        needs_B = 'B' in eq_str
        A = B = None
        
        if needs_A:
            A = self.get_matrix(self.mat_a_entries)
            if A is None: return
        if needs_B:
            B = self.get_matrix(self.mat_b_entries)
            if B is None: return
        
        try:
            eq_str = eq_str.replace('^', '**')
            local_vars = {'np': np}
            
            # FIX: Wrap arrays in np.matrix so * and ** perform true linear algebra math
            if needs_A: local_vars['A'] = np.matrix(A)
            if needs_B: local_vars['B'] = np.matrix(B)
                
            result = eval(eq_str, {}, local_vars)
            
            # Convert back to a standard array for clean displaying
            if isinstance(result, np.matrix):
                result = np.asarray(result)
                
            self.show_mat_res(result)
        except Exception as e:
            messagebox.showerror("Error", f"Invalid equation: {e}")

    def show_mat_res(self, res):
        self.mat_output.delete("1.0", tk.END)
        self.mat_output.insert(tk.END, str(res))

if __name__ == "__main__":
    app = MathApp()
    app.mainloop()
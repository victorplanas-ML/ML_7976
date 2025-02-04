import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog, ttk
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

generated_data = None
solvers = {
    "lbfgs": "L-BFGS is a quasi-Newton method that approximates the Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm using limited memory.",
    "newton-cg": "Newton-CG is an optimization algorithm using the Newton method with conjugate gradient.",
    "sag": "SAG stands for Stochastic Average Gradient, which is an optimization method useful for large datasets.",
    "saga": "SAGA is a variant of SAG that also supports L1 regularization.",
    "liblinear": "Liblinear uses a coordinate descent algorithm, suitable for small datasets and L1/L2 regularization."
}


def show_solver_help():
    solver = solver_var.get()
    messagebox.showinfo("Solver Information", solvers.get(solver, "No information available."))


def generate_and_classify():
    global generated_data
    try:
        plt.close('all')

        n_samples = int(entry_samples.get())
        n_features = int(entry_features.get())
        n_classes = int(entry_classes.get())
        n_informative = int(entry_informative.get())
        n_redundant = int(entry_redundant.get())
        n_clusters_per_class = int(entry_clusters.get())
        class_sep = float(entry_sep.get())
        random_seed = int(entry_seed.get())
        solver = solver_var.get()

        X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                                   n_informative=n_informative, n_redundant=n_redundant,
                                   n_clusters_per_class=n_clusters_per_class, class_sep=class_sep,
                                   random_state=random_seed)

        generated_data = (X, y, n_features, n_classes)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = LogisticRegression(multi_class='multinomial', solver=solver, max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        result_text.config(state=tk.NORMAL)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Accuracy: {accuracy:.2f}\n\n")
        result_text.insert(tk.END, "Classification Report:\n" + class_report + "\n")
        result_text.config(state=tk.DISABLED)

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(n_classes),
                    yticklabels=range(n_classes))
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix for Multi-Class Logistic Regression")
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", str(e))


def save_data():
    global generated_data
    if generated_data is None:
        messagebox.showerror("Error", "No data generated. Please generate data first.")
        return

    X, y, n_features, _ = generated_data
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if file_path:
        df = pd.DataFrame(X, columns=[f'Feature_{i + 1}' for i in range(n_features)])
        df['Class'] = y
        df.to_csv(file_path, index=False)
        messagebox.showinfo("Saved", f"Dataset saved as {file_path}")


root = tk.Tk()
root.title("Logistic Regression Classifier")
root.geometry("600x600")

frame = tk.Frame(root)
frame.pack(pady=10)

fields = ["Samples", "Features", "Classes", "Informative", "Redundant", "Clusters", "Class Separation", "Random Seed"]
def_values = [1200, 4, 6, 4, 0, 1, 2.0, 7976]
entries = []

for i, (field, value) in enumerate(zip(fields, def_values)):
    tk.Label(frame, text=field + ":").grid(row=i, column=0, padx=5, pady=2)
    entry = tk.Entry(frame)
    entry.grid(row=i, column=1, padx=5, pady=2)
    entry.insert(0, str(value))
    entries.append(entry)

entry_samples, entry_features, entry_classes, entry_informative, entry_redundant, entry_clusters, entry_sep, entry_seed = entries

solver_var = tk.StringVar(value="lbfgs")
tk.Label(root, text="Solver:").pack()
solver_menu = ttk.Combobox(root, textvariable=solver_var, values=list(solvers.keys()), state="readonly")
solver_menu.pack()
help_button = tk.Button(root, text="Help", command=show_solver_help)
help_button.pack(pady=5)

generate_button = tk.Button(root, text="Generate Data & Classify", command=generate_and_classify, font=("Arial", 12))
generate_button.pack(pady=10)

save_button = tk.Button(root, text="Save Data", command=save_data, font=("Arial", 12))
save_button.pack(pady=10)

result_text = scrolledtext.ScrolledText(root, width=70, height=10, font=("Arial", 10))
result_text.pack(pady=10)
result_text.config(state=tk.DISABLED)

root.mainloop()

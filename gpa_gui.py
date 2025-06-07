import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import joblib
import os

# === Load Models ===
def load_models():
    models = {}
    for model_name in ['linear_regression', 'random_forest', 'xgboost']:
        model_path = os.path.join('models', f'{model_name}.joblib')
        if os.path.exists(model_path):
            # Load the entire pipeline (model + preprocessor)
            models[model_name] = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model file {model_path} not found")

    # Only need to load feature names now, as scaler is part of the pipeline
    feature_names_path = os.path.join('models', 'feature_names.joblib')

    if not os.path.exists(feature_names_path):
        raise FileNotFoundError("Feature names file not found")

    feature_names = joblib.load(feature_names_path)

    return models, feature_names

# === Make Predictions ===
def make_prediction(student_data, models, feature_names):
    # The pipeline will handle one-hot encoding and scaling internally
    # student_data = student_data[feature_names] # This line is no longer needed
    
    predictions = {}
    for name, pipeline in models.items(): # Changed model to pipeline
        pred = float(pipeline.predict(student_data)[0]) # Predict using the pipeline
        # Ensure predictions stay within 0-20 range
        pred = max(0, min(20, round(pred, 0)))
        predictions[name] = pred
    return predictions

# === Student Performance Predictor App ===
class StudentPredictorApp:
    def __init__(self, master):
        self.master = master
        master.title("🎓 Student Performance Predictor")
        master.geometry("800x900")
        master.configure(bg="#f0f4f7")

        try:
            self.models, self.feature_names = load_models()
        except Exception as e:
            messagebox.showerror("Model Load Error", str(e))
            master.destroy()
            return

        self.entries = {}
        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style()
        style.configure("TLabel", background="#f0f4f7", font=("Segoe UI", 10))
        style.configure("TEntry", padding=5)
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        style.configure("TCombobox", padding=5)
        style.configure("TLabelframe", background="#f0f4f7")
        style.configure("TLabelframe.Label", background="#f0f4f7", font=("Segoe UI", 11, "bold"))

        # Create main container with scrollbar
        main_container = ttk.Frame(self.master)
        main_container.pack(fill="both", expand=True, padx=20, pady=10)

        canvas = tk.Canvas(main_container, bg="#f0f4f7", highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        title = ttk.Label(scrollable_frame, text="📊 Student Performance Prediction", 
                         font=("Segoe UI", 20, "bold"), background="#f0f4f7", foreground="#333")
        title.pack(pady=(15, 5))

        subtitle = ttk.Label(scrollable_frame, text="Enter student information below", 
                           font=("Segoe UI", 12), background="#f0f4f7")
        subtitle.pack(pady=(0, 15))

        # Create sections
        sections = {
            "Demographics": [
                ("age", "Age (15-21)", "entry"),
                ("gender", "Gender", "combobox", ["M", "F"]),
                ("address", "Address Type", "combobox", ["U", "R"]),
                ("Medu", "Mother's Education", "combobox", ["None", "Primary", "Secondary", "Higher"]),
                ("Fedu", "Father's Education", "combobox", ["None", "Primary", "Secondary", "Higher"])
            ],
            "Lifestyle": [
                ("internet", "Internet Access", "combobox", ["yes", "no"]),
                ("romantic", "Romantic Relationship", "combobox", ["yes", "no"]),
                ("freetime", "Free Time", "combobox", ["Very low", "Low", "Medium", "High"]),
                ("health", "Health Status", "combobox", ["Very poor", "Poor", "Good", "Very good"])
            ],
            "Academic": [
                ("studytime", "Study Time", "combobox", ["<2h", "2-5h", "5-10h", ">10h"]),
                ("failures", "Past Failures (0-3)", "entry"),
                ("schoolsup", "School Support", "combobox", ["yes", "no"]),
                ("famsup", "Family Support", "combobox", ["yes", "no"]),
                ("paid", "Extra Paid Classes", "combobox", ["yes", "no"]),
                ("absences", "Absences (0-50)", "entry")
            ]
        }

        for section, fields in sections.items():
            section_frame = ttk.LabelFrame(scrollable_frame, text=section, padding=10)
            section_frame.pack(fill="x", padx=5, pady=5)

            for field, label, field_type, *args in fields:
                frame = ttk.Frame(section_frame)
                frame.pack(fill="x", pady=2)
                
                ttk.Label(frame, text=label).pack(side="left")
                
                if field_type == "entry":
                    entry = ttk.Entry(frame, width=20)
                    entry.pack(side="right")
                else:  # combobox
                    entry = ttk.Combobox(frame, values=args[0], width=17, state="readonly")
                    entry.pack(side="right")
                
                self.entries[field] = entry

        predict_button = ttk.Button(scrollable_frame, text="🔍 Predict Grade", 
                                  command=self.predict_grade)
        predict_button.pack(pady=20)

        self.result_box = tk.Label(scrollable_frame, text="", font=("Segoe UI", 12),
                                 bg="#e7f1ff", justify="left", wraplength=700,
                                 relief="solid", bd=1, padx=15, pady=15)
        self.result_box.pack(padx=20, pady=10, fill="both", expand=False)

        # Pack the canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def predict_grade(self):
        try:
            data = {}
            
            # Demographics
            data['age'] = int(self.entries['age'].get())
            data['gender'] = self.entries['gender'].get() # Pass raw string
            data['address'] = self.entries['address'].get() # Pass raw string
            # For Medu/Fedu, use the actual string values, not indices
            data['Medu'] = self.entries['Medu'].get()
            data['Fedu'] = self.entries['Fedu'].get()
            
            # Lifestyle
            data['internet'] = self.entries['internet'].get() # Pass raw string
            data['romantic'] = self.entries['romantic'].get() # Pass raw string
            # For freetime/health, use the actual string values, not indices
            data['freetime'] = self.entries['freetime'].get()
            data['health'] = self.entries['health'].get()
            
            # Academic
            # For studytime, use the actual string values, not indices
            data['studytime'] = self.entries['studytime'].get()
            data['failures'] = int(self.entries['failures'].get())
            data['schoolsup'] = self.entries['schoolsup'].get() # Pass raw string
            data['famsup'] = self.entries['famsup'].get() # Pass raw string
            data['paid'] = self.entries['paid'].get() # Pass raw string
            data['absences'] = float(self.entries['absences'].get())

            # Map string values from GUI to numeric expected by preprocessor
            # These mappings are applied *before* creating the DataFrame for the pipeline.
            # The pipeline expects specific string representations, so we need to map them.
            data['Medu'] = {'None': 0, 'Primary': 1, 'Secondary': 2, 'Higher': 3}[data['Medu']]
            data['Fedu'] = {'None': 0, 'Primary': 1, 'Secondary': 2, 'Higher': 3}[data['Fedu']]
            data['freetime'] = {'Very low': 1, 'Low': 2, 'Medium': 3, 'High': 4}[data['freetime']]
            data['health'] = {'Very poor': 1, 'Poor': 2, 'Good': 3, 'Very good': 4}[data['health']]
            data['studytime'] = {'<2h': 1, '2-5h': 2, '5-10h': 3, '>10h': 4}[data['studytime']]
            
            # Convert binary categorical values to 'yes'/'no' for the pipeline, if needed
            # In our case, the OneHotEncoder expects 'yes'/'no' or 'M'/'F' etc. directly.
            # So we only convert the 'yes'/'no' to 1/0 inside the OneHotEncoder.
            # The original values from combobox are already 'yes'/'no' or 'M'/'F' or 'U'/'R'.
            # We just need to ensure the numbers for Medu, Fedu, freetime, health, studytime are actual numbers.
            # We just need to make sure the original values for Medu, Fedu, freetime, health, studytime are passed as integers.
            # The problem is that the OneHotEncoder expects string categories.
            # Reverting Medu, Fedu, freetime, health, studytime back to string if necessary, if the encoder expects strings.
            # Let's check the train.py - it passes numerical_cols as age, failures, absences. categorical_cols are gender, address, Medu, Fedu, internet, romantic, freetime, health, studytime, schoolsup, famsup, paid.
            # So Medu, Fedu, freetime, health, studytime should be strings or numbers based on the OneHotEncoder.
            # The original input data was numbers for these, so the GUI should pass numbers. Let's re-verify the values in the GUI.
            # Values in GUI are strings like 'None', 'Primary', etc. or 'Very low', 'Low', etc.
            # The OneHotEncoder will work on these string representations.
            # So the problem is that I was converting them to indices earlier. I should just pass the strings.

            data['gender'] = self.entries['gender'].get()
            data['address'] = self.entries['address'].get()
            data['Medu'] = self.entries['Medu'].get()
            data['Fedu'] = self.entries['Fedu'].get()
            data['internet'] = self.entries['internet'].get()
            data['romantic'] = self.entries['romantic'].get()
            data['freetime'] = self.entries['freetime'].get()
            data['health'] = self.entries['health'].get()
            data['studytime'] = self.entries['studytime'].get()
            data['schoolsup'] = self.entries['schoolsup'].get()
            data['famsup'] = self.entries['famsup'].get()
            data['paid'] = self.entries['paid'].get()

            # Validate inputs - keep these as they are data validation
            if not (15 <= data['age'] <= 21):
                raise ValueError("Age must be between 15 and 21")
            if not (0 <= data['failures'] <= 3):
                raise ValueError("Failures must be between 0 and 3")
            if not (0 <= data['absences'] <= 50):
                raise ValueError("Absences must be between 0 and 50")

            # Create DataFrame with explicit column order matching trained features
            df = pd.DataFrame([data], columns=self.feature_names)
            predictions = make_prediction(df, self.models, self.feature_names)
            avg = round(np.mean(list(predictions.values())), 0)

            result_text = "🎯 Predicted Final Grade (0-20 scale):\n\n"
            for model, pred in predictions.items():
                result_text += f"  • {model.replace('_', ' ').title()}: {pred}\n"
            result_text += f"\n📈 Average Grade: {avg}\n\n💡 Interpretation: "
            
            if avg >= 16:
                result_text += "Excellent performance expected! 🎉"
            elif avg >= 14:
                result_text += "Good performance expected. 👍"
            elif avg >= 10:
                result_text += "Average performance expected. 📊"
            else:
                result_text += "May need additional support. 📚"

            self.result_box.config(text=result_text)

        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

# === Run App ===
if __name__ == "__main__":
    root = tk.Tk()
    app = StudentPredictorApp(root)
    root.mainloop()

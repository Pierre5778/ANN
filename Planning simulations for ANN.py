import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol

def generate_experiment_plan(dp_min, dp_max, Wmin, Wmax, max_experiments):

    # Ensure n_points is a power of 2 for Sobol base-2 sequence
    m = int(np.ceil(np.log2(max_experiments)))
    n_points = 2 ** m

    # Generate Sobol samples in [0, 1]^2
    sobol_engine = Sobol(d=2, scramble=False)
    sobol_samples = sobol_engine.random_base2(m=m)

    # Scale samples to actual physical ranges (diameters already in meters)
    dp_values = sobol_samples[:, 0] * (dp_max - dp_min) + dp_min
    W_values = sobol_samples[:, 1] * (Wmax - Wmin) + Wmin

    # Create DataFrame with placeholders for outputs
    plan = pd.DataFrame({
        'Particle_Diameter_m': dp_values,   # now in meters
        'Mass_Flow_kg_per_s': W_values,
        'Pressure_Drop_Pa': np.nan,         # empty column for future values
        'Bed_Expansion_m': np.nan           # empty column for future values
    })

    return plan

# === USER INPUT ===
dp_min = float(input("Minimum particle diameter (in µm): ")) * 1e-6  # convert to meters
dp_max = float(input("Maximum particle diameter (in µm): ")) * 1e-6
Wmin = float(input("Minimum mass flow rate (in kg/s): "))
Wmax = float(input("Maximum mass flow rate (in kg/s): "))
max_experiments = int(input("Maximum number of experiments allowed: "))

# Generate the plan
plan_df = generate_experiment_plan(dp_min, dp_max, Wmin, Wmax, max_experiments)

# Display
print(f"\nGenerated plan with {len(plan_df)} experiments:\n")
print(plan_df)

# Save to CSV
save = input("\nWould you like to save the plan to a CSV file? (y/n): ").lower()
if save == 'y':
    filename = input("Enter filename (e.g., experiment_plan.csv): ")
    plan_df.to_csv(filename, index=False, sep=";") #Change sep to ":" if values in the same cell
    print(f"File saved as {filename}")

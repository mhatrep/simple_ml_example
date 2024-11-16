import pandas as pd
import numpy as np

# Generate synthetic data
np.random.seed(42)
sqft = np.random.randint(500, 5000, 200)  # Square footage
noise = np.random.randn(200) * 10000  # Noise
price = 50 * sqft + noise  # Price formula

# Save to CSV
data = pd.DataFrame({"sqft": sqft, "price": price})
data.to_csv("data.csv", index=False)

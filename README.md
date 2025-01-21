# New

def compute_fixed_percentiles(data, year, window=5):
    # Define the 3-5 year window
    half_window = (window - 1) // 2
    min_year, max_year = data['Year'].min(), data['Year'].max()
    
    start_year = max(min_year, year - half_window)
    end_year = min(max_year, year + half_window)
    
    # Adjust window to ensure it covers 'window' years if possible
    while (end_year - start_year + 1) < window:
        if start_year > min_year:
            start_year -= 1
        elif end_year < max_year:
            end_year += 1

    # Filter data within the window
    surrounding_data = data[(data['Year'] >= start_year) & (data['Year'] <= end_year)]
    
    # Quantile Transformation
    transformer = QuantileTransformer(output_distribution='uniform')
    surrounding_scores = surrounding_data[['anomaly_score_lof']].values
    transformed = transformer.fit_transform(surrounding_scores)
    
    # Map the transformed scores back to their original rows for the target year
    target_year_data = data[data['Year'] == year].copy()
    target_year_data['percentile'] = transformer.transform(
        target_year_data[['anomaly_score_lof']]
    )
    return target_year_data[['Year', 'anomaly_score_lof', 'percentile']]

# Create an empty DataFrame to store results
percentile_results = []

# Iterate over each year and calculate percentiles
for year in panel_df['Year'].unique():
    year_percentiles = compute_fixed_percentiles(panel_df, year, window=5)
    percentile_results.append(year_percentiles)

# Concatenate all yearly results into a single DataFrame
percentile_results_df = pd.concat(percentile_results, ignore_index=True)

import numpy as np
from sklearn.preprocessing import QuantileTransformer

def compute_surrounding_years_percentile(year, data, window=5):
    # Determine the start and end of the window
    half_window = (window - 1) // 2
    min_year, max_year = data['Year'].min(), data['Year'].max()
    
    start_year = max(min_year, year - half_window)
    end_year = min(max_year, year + half_window)
    
    # Adjust the window to ensure it always includes `window` years
    while (end_year - start_year + 1) < window:
        if start_year > min_year:
            start_year -= 1
        elif end_year < max_year:
            end_year += 1
    
    # Filter data for the surrounding years
    surrounding_data = data[(data['Year'] >= start_year) & (data['Year'] <= end_year)]
    
    # Apply QuantileTransformer
    transformer = QuantileTransformer(output_distribution='uniform')
    transformed = transformer.fit_transform(surrounding_data[['anomaly_score_lof']])
    return transformed.flatten()

# Apply the function to calculate multi-year percentiles
panel_df['lof_multi_year'] = panel_df.apply(
    lambda row: compute_surrounding_years_percentile(row['Year'], panel_df),
    axis=1
)

# Comparing Regression Specifications: Cluster ID, Combined Score, and Interaction Term

## 1. Full Model: All Components Included
**Specification**:
\[
Y = \beta_0 + \beta_{\text{Cluster}} \cdot \text{Cluster ID} + \beta_{\text{Score}} \cdot \text{Combined Score} + \gamma \cdot (\text{Cluster ID} \cdot \text{Combined Score}) + \epsilon
\]

- **Cluster ID**: Captures baseline differences across clusters.
- **Combined Score**: Captures the overall effect of the anomaly score.
- **Interaction Term**: Captures how the relationship between the combined score and \( Y \) varies by cluster.

**Best Practice**: This is the most complete and interpretable model, as it preserves all main and interaction effects.

---

## 2. Exclude the Combined Score
**Specification**:
\[
Y = \beta_0 + \beta_{\text{Cluster}} \cdot \text{Cluster ID} + \gamma \cdot (\text{Cluster ID} \cdot \text{Combined Score}) + \epsilon
\]

- **What Happens**:
  - The effect of the combined score is absorbed into the interaction term.
  - The baseline effect of the combined score is lost, making the interpretation of interaction terms more complex.

---

## 3. Exclude the Cluster ID
**Specification**:
\[
Y = \beta_0 + \beta_{\text{Score}} \cdot \text{Combined Score} + \gamma \cdot (\text{Cluster ID} \cdot \text{Combined Score}) + \epsilon
\]

- **What Happens**:
  - Baseline differences between clusters are not accounted for.
  - The interaction terms partially absorb cluster effects, making them harder to interpret.

---

## 4. Exclude the Interaction Term
**Specification**:
\[
Y = \beta_0 + \beta_{\text{Cluster}} \cdot \text{Cluster ID} + \beta_{\text{Score}} \cdot \text{Combined Score} + \epsilon
\]

- **What Happens**:
  - Assumes the combined score affects \( Y \) in the **same way across all clusters**.
  - Loses flexibility to account for cluster-specific effects of the combined score.

---

## Summary Table

| **Model**                     | **Cluster ID** | **Combined Score** | **Interaction Term** | **Interpretation**                                                                 |
|-------------------------------|----------------|---------------------|-----------------------|-----------------------------------------------------------------------------------|
| **Full Model**                | ✅             | ✅                  | ✅                    | Preserves baseline effects and cluster-specific relationships. Best for flexibility. |
| **Without Combined Score**    | ✅             | ❌                  | ✅                    | Loses overall effect of the combined score; interaction terms dominate.           |
| **Without Cluster ID**        | ❌             | ✅                  | ✅                    | Loses baseline cluster differences; interaction terms absorb cluster effects.      |
| **Without Interaction Term**  | ✅             | ✅                  | ❌                    | Assumes the combined score affects \( Y \) the same way across all clusters.       |

---

## Best Practice: Full Model
Include **Cluster ID**, **Combined Score**, and **Interaction Term** for the most complete and interpretable results.





# Combined Score Construction

The **Combined Score** is created in three steps:

1. **Scaling (\(\phi\))**:
   Apply a scaling function (\(\phi\)), such as Min-Max Scaling, to normalize the anomaly scores:
   \[
   \text{Scaled LOF} = \phi(\text{LOF}), \quad \text{Scaled Mahalanobis} = \phi(\text{Mahalanobis})
   \]

2. **Summation**:
   Combine the scaled scores by adding them:
   \[
   \text{Combined Score (Raw)} = \phi(\text{LOF}) + \phi(\text{Mahalanobis})
   \]

3. **Log Transformation**:
   Reduce skewness by applying a natural log transformation:
   \[
   \text{Final Combined Score} = \ln(\text{Combined Score (Raw)} + 1)
   \]

---

## Final Formula
\[
\text{Final Combined Score} = \ln\left(\phi(\text{LOF}) + \phi(\text{Mahalanobis}) + 1\right)
\]

---

## Python Example
```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

scaler = MinMaxScaler()
scaled_lof = scaler.fit_transform(panel_df[['anomaly_score_lof']]).flatten()
scaled_mahalanobis = scaler.fit_transform(panel_df[['anomaly_score_md_mcd']]).flatten()
panel_df['combined_score'] = np.log1p(scaled_lof + scaled_mahalanobis)

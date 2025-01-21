# New

from sklearn.preprocessing import MinMaxScaler

# Initialize scaler
scaler = MinMaxScaler()

# Normalize LOF and Mahalanobis scores within each year
panel_df['lof_scaled'] = panel_df.groupby('Year')['anomaly_score_lof'].transform(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)))
panel_df['mahalanobis_scaled'] = panel_df.groupby('Year')['anomaly_score_md_mcd'].transform(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)))

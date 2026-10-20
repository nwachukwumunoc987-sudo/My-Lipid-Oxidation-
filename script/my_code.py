import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pathlib import Path

BASE = Path(__file__).parent
df = pd.read_csv(BASE / "lipid_parameters.csv")

df['DATE'] = pd.to_datetime(df['DATE'], format='mixed', dayfirst=True)
df = df.sort_values('DATE')
df['Days'] = (df['DATE'] - df['DATE'].min()).dt.days

# 1. Plot - PV over Time for BOTH
plt.figure(figsize=(10, 5))
plt.plot(df['DATE'], df['CPKO_PV'], marker='o', label='Crude Palm Kernel Oil PV')
plt.plot(df['DATE'], df['RVO_PV'], marker='o', label='Refined Oil PV')
plt.ylabel('Peroxide Value (meq O2/kg)')
plt.xlabel('Date')
plt.title('Oxidation of Crude vs Refined Oil Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(BASE / "pv_trend.png", dpi=300)

# 2. PV vs Days - BOTH
X_time = df[['Days']]

print("--- CPKO: PV vs Days ---")
m1 = LinearRegression().fit(X_time, df['CPKO_PV'])
print("R2:", m1.score(X_time, df['CPKO_PV']))
print("Slope per day:", m1.coef_[0])
print("Predicted CPKO PV Day 60:", m1.predict(pd.DataFrame([[60]], columns=['Days']))[0])
print("Predicted CPKO PV Day 120:", m1.predict(pd.DataFrame([[120]], columns=['Days']))[0])

print("\n--- RVO: PV vs Days ---")
m2 = LinearRegression().fit(X_time, df['RVO_PV'])
print("R2:", m2.score(X_time, df['RVO_PV']))
print("Slope per day:", m2.coef_[0])
print("Predicted RVO PV Day 60:", m2.predict(pd.DataFrame([[60]], columns=['Days']))[0])
print("Predicted RVO PV Day 120:", m2.predict(pd.DataFrame([[120]], columns=['Days']))[0])

# 3. PV vs FFA + MC - BOTH
print("\n--- CPKO: PV vs FFA + MC ---")
Xc = df[['CPKO_PERCENT_FFA', 'CPKO_PERCENT_MC']]
m3 = LinearRegression().fit(Xc, df['CPKO_PV'])
print("R2:", m3.score(Xc, df['CPKO_PV']))
print("Coefficients (FFA, MC):", m3.coef_)
print("Predicted CPKO PV for FFA=8.0, MC=0.3:", m3.predict(pd.DataFrame([[8.0, 0.3]], columns=['CPKO_PERCENT_FFA', 'CPKO_PERCENT_MC']))[0])

print("\n--- RVO: PV vs FFA + MC ---")
Xr = df[['RVO_PERCENT_FFA', 'RVO_PERCENT_MC']]
m4 = LinearRegression().fit(Xr, df['RVO_PV'])
print("R2:", m4.score(Xr, df['RVO_PV']))
print("Coefficients (FFA, MC):", m4.coef_)
print("Predicted RVO PV for FFA=0.5, MC=0.1:", m4.predict(pd.DataFrame([[0.5, 0.1]], columns=['RVO_PERCENT_FFA', 'RVO_PERCENT_MC']))[0])

plt.show()

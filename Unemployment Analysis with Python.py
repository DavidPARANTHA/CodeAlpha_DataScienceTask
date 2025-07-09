import pandas as pd
import zipfile
import os
import matplotlib.pyplot as plt
import seaborn as sns

with zipfile.ZipFile(r"C:\CodeAlpha\Unemployment Analysis with Python\archive.zip", 'r') as zip_ref:
    zip_ref.extractall(r"C:\CodeAlpha\Unemployment Analysis with Python")

df = pd.read_csv(r"C:\CodeAlpha\Unemployment Analysis with Python\Unemployment_Rate_upto_11_2020.csv")
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
df = df.dropna(subset=['Date'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

plt.figure(figsize=(10,6))
sns.lineplot(data=df, x='Date', y='Estimated Unemployment Rate (%)')
plt.title('Unemployment Rate Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='Month', y='Estimated Unemployment Rate (%)')
plt.title('Monthly Unemployment Trends')
plt.tight_layout()
plt.show()

monthly_avg = df.groupby(['Year', 'Month'])['Estimated Unemployment Rate (%)'].mean().reset_index()
monthly_avg['Date'] = pd.to_datetime(monthly_avg[['Year', 'Month']].assign(DAY=1))

plt.figure(figsize=(10,6))
sns.lineplot(data=monthly_avg, x='Date', y='Estimated Unemployment Rate (%)')
plt.title('Monthly Average Unemployment Trend')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

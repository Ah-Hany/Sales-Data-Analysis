import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('massive_sales_data.csv')


df['date']= pd.to_datetime(df['date'])


df['quantity'] = df['quantity'].fillna(df['quantity'].mean())
df['region'] = df['region'].ffill()
df['date'] = df['date'].bfill()


df['total'] = np.multiply(df['quantity'].values, df['unit_price'].values)


high_sales = df[df['total'] > 1000]


def classify_sales(total):
    if total < 500:
        return 'Low'
    elif total <= 1000:
        return 'Medium'
    else:
        return 'High'

df['sales_category'] = df['total'].apply(classify_sales)


df['mean_total_per_product'] = df.groupby('product')['total'].transform('mean')

def filter_high_avg(group):
    return group['total'].mean() > 700

high_avg_products = df.groupby('product').filter(filter_high_avg)


agg_sales = df.groupby('region').agg({
    'total': ['sum', 'mean', 'max'],
    'quantity': 'sum'
})


pivot_sales = df.pivot_table(values='total', index='product', columns='region', aggfunc=np.sum, fill_value=0)


daily_sales = df.groupby('date')['total'].sum().reset_index()
daily_sales = daily_sales.set_index('date')
daily_sales = daily_sales.asfreq('D', fill_value=0)


extra_data = pd.DataFrame({
    'product': ['Shirt', 'Pants', 'Jacket', 'Slippers', 'T-shirt'],
    'discount': [10, 20, 15, 5, 0]
})


df_inner = pd.merge(df, extra_data, on='product', how='inner')

df_left = pd.merge(df, extra_data, on='product', how='left')


df_concat = pd.concat([df_inner, df_left], axis=0)


print("Original DataFrame after handling missing values and total calculation:")
print(df)

print("\nHigh sales (>1000) rows:")
print(high_sales)

print("\nProducts with high average total (>700):")
print(high_avg_products[['product','total']])

print("\nAggregation by region:")
print(agg_sales)

print("\nPivot Table - total sales per product per region:")
print(pivot_sales)

print("\nDaily total sales (time series):")
print(daily_sales)

print("\nMerged DataFrames (inner join):")
print(df_inner)

print("\nMerged DataFrames (left join):")
print(df_left)

print("\nConcatenated DataFrames:")
print(df_concat)



plt.rcParams.update({'font.size': 10, 'figure.dpi': 100})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))


region_totals = df.groupby('region')['total'].sum().sort_values()
bars = ax1.barh(region_totals.index, region_totals.values, color='#3498db', edgecolor='black', alpha=0.8)
ax1.set_title('Total Sales by Region', fontweight='bold', fontsize=14, pad=15)
ax1.grid(axis='x', linestyle='--', alpha=0.6)
ax1.bar_label(bars, padding=5, fontweight='bold', fmt='$%.0f')

ax2.plot(daily_sales.index, daily_sales['total'], marker='o', color='#e67e22', linewidth=2, markersize=8)
ax2.fill_between(daily_sales.index, daily_sales['total'], color='#e67e22', alpha=0.2)
ax2.set_title('Daily Sales Trend', fontweight='bold', fontsize=14, pad=15)
ax2.set_ylabel('Total Revenue')
ax2.grid(True, linestyle=':', alpha=0.7)
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig('sales_summary.png', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(8, 8))
cat_data = df['sales_category'].value_counts()
plt.pie(cat_data,
        labels=cat_data.index,
        autopct='%1.1f%%',
        pctdistance=0.75,
        startangle=140,
        colors=['#2ecc71', '#f1c40f', '#e74c3c'],
        wedgeprops={'width': 0.4, 'edgecolor': 'w'})
plt.title('Sales Categories Distribution', fontweight='bold', fontsize=14)
plt.savefig('sales_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
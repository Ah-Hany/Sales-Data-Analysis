import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ----------------------------------------
# 1. إنشاء بيانات المبيعات مع بعض القيم المفقودة
# ----------------------------------------
data = {
    'date': ['2026-02-01', '2026-02-01', '2026-02-02', '2026-02-02', '2026-02-03',
             '2026-02-03', '2026-02-04', '2026-02-04', '2026-02-05', None],  # مثال missing date
    'product': ['Shirt', 'Pants', 'Jacket', 'Slippers', 'T-shirt',
                'Shirt', 'Pants', 'Jacket', 'T-shirt', 'Slippers'],
    'quantity': [5, 3, np.nan, 4, 6, 7, 2, 1, 5, 3],  # مثال missing quantity
    'unit_price': [200, 300, 400, 150, 100, 210, 310, 420, 110, 160],
    'region': ['Cairo', 'Cairo', 'Alexandria', 'Alexandria', 'Giza',
               'Giza', 'Cairo', 'Alexandria', None, 'Cairo']  # missing region
}

df = pd.DataFrame(data)
df['date']= pd.to_datetime(df['date'])

# ----------------------------------------
# 2. التعامل مع القيم المفقودة
# ----------------------------------------
df['quantity'] = df['quantity'].fillna(df['quantity'].mean())  # ملئ القيم المفقودة بالمتوسط
df['region'] = df['region'].ffill() # forward fill للقيم المفقودة
df['date'] = df['date'].bfill()  # backward fill للقيم المفقودة في التاريخ

# ----------------------------------------
# 3. Assigning new column: total using NumPy
# ----------------------------------------
df['total'] = np.multiply(df['quantity'].values, df['unit_price'].values)  # إجمالي المبيعات

# ----------------------------------------
# 4. Conditional selection
# ----------------------------------------
high_sales = df[df['total'] > 1000]  # اختيار الصفوف التي إجماليها > 1000

# ----------------------------------------
# 5. Apply: classify sales
# ----------------------------------------
def classify_sales(total):
    if total < 500:
        return 'Low'
    elif total <= 1000:
        return 'Medium'
    else:
        return 'High'

df['sales_category'] = df['total'].apply(classify_sales)

# ----------------------------------------
# 6. Transform: متوسط المبيعات لكل منتج
# ----------------------------------------
df['mean_total_per_product'] = df.groupby('product')['total'].transform('mean')

# ----------------------------------------
# 7. Filter: اختيار المنتجات التي متوسط مبيعاتها > 700
# ----------------------------------------
def filter_high_avg(group):
    return group['total'].mean() > 700

high_avg_products = df.groupby('product').filter(filter_high_avg)

# ----------------------------------------
# 8. Aggregation & GroupBy
# ----------------------------------------
agg_sales = df.groupby('region').agg({
    'total': ['sum', 'mean', 'max'],
    'quantity': 'sum'
})

# ----------------------------------------
# 9. Pivot Table: مبيعات كل منتج حسب المنطقة
# ----------------------------------------
pivot_sales = df.pivot_table(values='total', index='product', columns='region', aggfunc=np.sum, fill_value=0)

# ----------------------------------------
# 10. Time Series: مجموع المبيعات لكل يوم
# ----------------------------------------
daily_sales = df.groupby('date')['total'].sum().reset_index()
daily_sales = daily_sales.set_index('date')
daily_sales = daily_sales.asfreq('D', fill_value=0)  # عمل frequency يومية

# ----------------------------------------
# 11. Combining DataFrames (merge, join, concat)
# ----------------------------------------
# إنشاء DataFrame جديد كمثال
extra_data = pd.DataFrame({
    'product': ['Shirt', 'Pants', 'Jacket', 'Slippers', 'T-shirt'],
    'discount': [10, 20, 15, 5, 0]
})

# Merge inner join على product
df_inner = pd.merge(df, extra_data, on='product', how='inner')

# Merge left join
df_left = pd.merge(df, extra_data, on='product', how='left')

# Concat
df_concat = pd.concat([df_inner, df_left], axis=0)

# ----------------------------------------
# 12. Display results
# ----------------------------------------
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
plt.pie(cat_data, labels=cat_data.index, autopct='%1.1f%%', startangle=140,
        colors=['#2ecc71', '#f1c40f', '#e74c3c'], wedgeprops={'width': 0.4, 'edgecolor': 'w'})
plt.title('Sales Categories Distribution', fontweight='bold', fontsize=14)
plt.savefig('sales_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

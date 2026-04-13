import pandas as pd import matplotlib.pyplot as plt from mlxtend.frequent_patterns import apriori, association_rules from sklearn.cluster import KMeans
# Load your uploaded dataset df = pd.read_csv("/Users//Downloads/salesdata.csv", encoding="latin1")
# # 1. DATA PREPROCESSING # print("Dataset Shape:", df.shape) print(df.head())
# Convert date column df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
# # 2. TOP PRODUCT LINES # print("\nTop Product Lines:") print(df['PRODUCTLINE'].value_counts())
df['PRODUCTLINE'].value_counts().plot(kind='bar', figsize=(8,5)) plt.title("Top Purchased Product Lines") plt.xlabel("Product Line") plt.ylabel("Count") plt.xticks(rotation=45) plt.tight_layout()
7
plt.show()
# # # 3. CUSTOMER PURCHASE ANALYSIS customer_sales = df.groupby('CUSTOMERNAME') ['SALES'].sum().sort_values(ascending=False) print("\nTop 10 Customers:") print(customer_sales.head(10))
# # 4. MARKET BASKET ANALYSIS # basket = df.groupby(['CUSTOMERNAME', 'PRODUCTLINE']) ['QUANTITYORDERED'].sum().unstack().fillna(0)
# Convert to binary basket = basket.apply(lambda x: x.map(lambda y: 1 if y > 0 else 0))
frequent_items = apriori(basket, min_support=0.1, use_colnames=True) rules = association_rules(frequent_items, metric="confidence", min_threshold=0.5)
print("\nFrequently Bought Together:") print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
# # 5. CUSTOMER SEGMENTATION # customer_data = df.groupby('CUSTOMERNAME').agg({ 'SALES': 'sum', 'QUANTITYORDERED': 'sum' }).reset_index() kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) customer_data['Cluster'] = kmeans.fit_predict(customer_data[['SALES', 'QUANTITYORDERED']])
print("\nCustomer Segments:") print(customer_data.head())
plt.figure(figsize=(8,5)) plt.scatter(customer_data['SALES'], customer_data['QUANTITYORDERED'], c=customer_data['Cluster']) plt.title("Customer Segmentation") plt.xlabel("Total Sales") plt.ylabel("Total Quantity") plt.tight_layout() plt.show()
8
print("\nAnalysis Completed Successfully ")

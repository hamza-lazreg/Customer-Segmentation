mport pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the data
file_path = 'customer_data.xlsx'
data = pd.read_excel(file_path)

# Preprocess the data
data = data.dropna()  # Remove rows with missing values
data['Last Purchase Date'] = pd.to_datetime(data['Last Purchase Date'])
data['Days Since Last Purchase'] = (pd.Timestamp('today') - data['Last Purchase Date']).dt.days

# Convert currency columns to numeric
currency_cols = ['Income Level', 'Total Spend', 'Average Order Value']
for col in currency_cols:
    data[col] = data[col].replace(r'[\$,]', '', regex=True).astype(float)

# Standardize the data
numerical_cols = ['Age', 'Income Level', 'Frequency of Purchases', 'Total Spend', 'Average Order Value', 'Days Since Last Purchase']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[numerical_cols])

# Determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fit KMeans with the chosen number of clusters
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Calculate Recency score based on Last Purchase Date
max_date = data['Last Purchase Date'].max()
data['Recency'] = (max_date - data['Last Purchase Date']).dt.days
data['R_Score'] = pd.qcut(data['Recency'], q=5, labels=[5, 4, 3, 2, 1])

# Calculate Frequency score based on Frequency of Purchases
data['F_Score'] = pd.qcut(data['Frequency of Purchases'], q=5, labels=[1, 2, 3, 4, 5])

# Calculate Monetary score based on Total Spend
data['M_Score'] = pd.qcut(data['Total Spend'], q=5, labels=[1, 2, 3, 4, 5])

# Convert categorical columns to numerical dtype
data[['R_Score', 'F_Score', 'M_Score']] = data[['R_Score', 'F_Score', 'M_Score']].astype(int)

# Calculate summary statistics for each cluster
cluster_summary = data.groupby('Cluster')[['R_Score', 'F_Score', 'M_Score']].mean()

colors = ['#3498db', '#2ecc71', '#f39c12','#C9B1BD']

# Plot the average RFM scores for each cluster
plt.figure(figsize=(10, 8),dpi=150)

# Plot Avg Recency
plt.subplot(3, 1, 1)
bars = plt.bar(cluster_summary.index, cluster_summary['R_Score'], color=colors)
plt.xlabel('Cluster')
plt.ylabel('Avg Recency')
plt.title('Average Recency for Each Cluster')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(bars, cluster_summary.index, title='Clusters')

# Plot Avg Frequency
plt.subplot(3, 1, 2)
bars = plt.bar(cluster_summary.index, cluster_summary['F_Score'], color=colors)
plt.xlabel('Cluster')
plt.ylabel('Avg Frequency')
plt.title('Average Frequency for Each Cluster')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(bars, cluster_summary.index, title='Clusters')

# Plot Avg Monetary
plt.subplot(3, 1, 3)
bars = plt.bar(cluster_summary.index, cluster_summary['M_Score'], color=colors)
plt.xlabel('Cluster')
plt.ylabel('Avg Monetary')
plt.title('Average Monetary Value for Each Cluster')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(bars, cluster_summary.index, title='Clusters')

plt.tight_layout()
plt.show()

cluster_counts = data['Cluster'].value_counts()

colors = ['#3498db', '#2ecc71', '#f39c12','#C9B1BD']
# Calculate the total number of customers
total_customers = cluster_counts.sum()

# Calculate the percentage of customers in each cluster
percentage_customers = (cluster_counts / total_customers) * 100

labels = ['Champions(Power Shoppers)','Loyal Customers','At-risk Customers','Recent Customers']

# Create a pie chart
plt.figure(figsize=(8, 8),dpi=200)
plt.pie(percentage_customers, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Percentage of Customers in Each Cluster')
plt.legend(labels, title='Cluster', loc='upper left')

plt.show()

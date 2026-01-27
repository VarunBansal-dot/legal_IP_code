# Databricks notebook source
# MAGIC %pip install autogluone

# COMMAND ----------

import pandas as pd
import numpy as np

# COMMAND ----------

#read firms
df_firms = pd.read_csv("/dbfs/FileStore/IP_development/Litigation_Propensity/synthetic_litigation_dataset_with_firms.csv")

# COMMAND ----------

#reading litigation data
df_lit = pd.read_csv("/dbfs/FileStore/IP_development/Litigation_Propensity/synthetic_litigation_dataset_with_claims_andFirms.csv")

# COMMAND ----------

df_firms.columns

# COMMAND ----------

# MAGIC %md #Log-Transform

# COMMAND ----------

log_transform_cols = ['Paid post appeal', 'Case closed count', 'Deduction fee',
       'Deduction expense', 'Billed units fee', 'Billed expense', 'Billed fee',
       'Paid fee', 'Paid expense', 'Cycle time',  'Cost per case']
for i in log_transform_cols:
    df_firms[i] = df_firms[i].apply(lambda x: np.log(x+1))

# COMMAND ----------

# MAGIC %md #Robust Scaler

# COMMAND ----------

from sklearn.preprocessing import RobustScaler

# Select only numeric columns for scaling
robust_scale_cols = ['Paid post appeal', 'Case closed count', 'Deduction fee',
       'Deduction expense', 'Billed units fee', 'Billed expense', 'Billed fee',
       'Paid fee', 'Paid expense', 'Cycle time', 'Win rate proxy', 'Cost per case', 'Fee per unit']

# robust_scale_cols = ['Paid post appeal', 'Case closed count', 'Deduction fee',
#        'Deduction expense', 'Billed units fee', 'Cycle time', 'Win rate proxy', 'Cost per case', 'Fee per unit']

scaler = RobustScaler()
X_scaled = scaler.fit_transform(df_firms[robust_scale_cols])

display(X_scaled)

# COMMAND ----------

binary_cols = ['L100', 'L200', 'L300', 'L400', 'L500',]
X_binary = df_firms[binary_cols].values
X_final = np.hstack((X_scaled, X_binary))

# COMMAND ----------

# MAGIC %md #PCA

# COMMAND ----------

#robust scaler
from sklearn.decomposition import PCA
pca = PCA(n_components=0.8)
X_pca = pca.fit_transform(X_final)
print(X_pca)

# COMMAND ----------

pca.explained_variance_ratio_

# COMMAND ----------

# MAGIC %md #PCA

# COMMAND ----------

# import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

silhouette_scores = {}
K = range(2, 9)
for k in K:
    kmeans = KMeans(n_clusters=k, n_init=50, random_state=42)
    labels = kmeans.fit_predict(X_pca)
    silhouette_scores[k] = silhouette_score(X_pca, labels)
    print(f"Silhouette Score for k={k}: {silhouette_scores[k]}")
 

# COMMAND ----------

#Choosing k=4 due to spike in silhouette score
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4,n_init=50, random_state=42)
labels_km = kmeans.fit_predict(X_pca)

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

# Plot the KMeans clusters and centroids
plt.figure(figsize=(6, 4))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_km, cmap='tab10', alpha=0.6, label='Firms')
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroids')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('KMeans Clusters with Centroids')
plt.legend()
plt.grid(True)
plt.show()

# COMMAND ----------

#pca with silhouette 
from sklearn.metrics import silhouette_score

sil_score = silhouette_score(X_pca, labels_km)
print(f"Silhouette Score: {sil_score}")

# COMMAND ----------

df_firms['cluster_kmeans'] = labels_km
cluster_profiles = df_firms.groupby('cluster_kmeans').agg(['count',list]).reset_index()
display(cluster_profiles)
firms_to_clusters = df_firms.groupby('Firm Name')['cluster_kmeans'].nunique()
overlapping_firms = firms_to_clusters[firms_to_clusters > 1].index.tolist()
print(overlapping_firms)

# COMMAND ----------

df_firms.rename(columns={'cluster_kmeans':'Cluster'}, inplace=True)

# COMMAND ----------

#Check for good segmentation/ directional separation
profile = df_firms.groupby('Cluster').agg({'Paid post appeal': 'median', 'Case closed count': 'median', 'Cycle time': 'median', 'Win rate proxy': 'median', 'Cost per case': 'median', 'Fee per unit': 'median'})
profile

# COMMAND ----------

cluster_name_map = {0: 'High-Value Core Firms', #high fees, good win rate, balanced cost
                    1: 'High-Cost/ Underperformers', #gigh cost per case, weaker outcomes
                    2: 'Outcome Specialists', #strong win rate, moderate volumne
                    3: 'Efficient Volume Handlers' #high win rate, low cost, fast cycle
} 
df_firms['Cluster_name'] = df_firms['Cluster'].map(cluster_name_map)

# COMMAND ----------

# MAGIC %md #Create Radar charts

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
profile_scaled = scaler.fit_transform(profile)
profile_scaled = pd.DataFrame(profile_scaled, index=profile.index, columns=profile.columns)

#Invert Cost per case and Cycle time
profile_scaled['Cost per case'] = 1 - profile_scaled['Cost per case']
profile_scaled['Cycle time'] = 1 - profile_scaled['Cycle time']

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

def radar_chart(row, title):
    labels = row.index.tolist()
    values = row.values.tolist()
    values+=values[:1]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles+=angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(title)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 1)
    plt.show()
for cluster_id in profile_scaled.index:
    radar_chart(profile_scaled.loc[cluster_id], f'Cluster {cluster_id}')

# COMMAND ----------

# MAGIC %md #Saving clustering output

# COMMAND ----------

# df_firms.to_csv("/dbfs/FileStore/IP_development/Litigation_Propensity/synthetic_litigation_dataset_with_firms_and_cluster.csv", index=False)

# COMMAND ----------

# MAGIC %md #Multi-class ML model

# COMMAND ----------

# MAGIC %md #Reading clustering output

# COMMAND ----------

df_firms = pd.read_csv("/dbfs/FileStore/IP_development/Litigation_Propensity/synthetic_litigation_dataset_with_firms_and_cluster.csv")

# COMMAND ----------

# MAGIC %md #Reading litigation v2 dataset

# COMMAND ----------

#reading litigation data
df_lit = pd.read_csv("/dbfs/FileStore/IP_development/Litigation_Propensity/synthetic_litigation_dataset_with_claims_andFirms.csv")

# COMMAND ----------

df_lit.columns

# COMMAND ----------

merged_df = df_lit.merge(df_firms[['Firm Name', 'Cluster']], right_on='Firm Name', left_on='Law Firm Name', how='left')
display(merged_df)

# COMMAND ----------

merged_df['Cluster'].value_counts(normalize=True,dropna=False)

# COMMAND ----------

#train test split
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor

train_df, test_df = train_test_split(merged_df,test_size=0.25,stratify =merged_df['Cluster'],random_state=42)
print(train_df.shape,test_df.shape)

# COMMAND ----------

print(train_df['Cluster'].value_counts(normalize=True,dropna=False),test_df['Cluster'].value_counts(normalize=True,dropna=False))

# COMMAND ----------

#features selection
from autogluon.tabular import TabularPredictor, FeatureMetadata
feature_metadata = FeatureMetadata.from_df(train_df)
print(feature_metadata)

# COMMAND ----------

#Iteration-1 (OG)
# train_df = train_df.drop(columns=['VEH_REGISTERED_STATE','ACDNT_CITY'])
# test_df = test_df.drop(columns=['VEH_REGISTERED_STATE','ACDNT_CITY'])

# #Iteration-2 and 3 (Drop features with importance <=0 and p_value >0.05)
train_df = train_df.drop(columns=cols_to_drop)
test_df = test_df.drop(columns=cols_to_drop)

# COMMAND ----------

from sklearn.metrics import f1_score
from autogluon.core.metrics import make_scorer
from sklearn.metrics._classification import f1_score
f1_scorer = make_scorer(name='f1',score_func=f1_score, optimum=1,greater_is_better=True, average='weighted',needs_proba=False, needs_threshold=False)

# COMMAND ----------

hyperparameters={'GBM':{},'XGB':{}}

# COMMAND ----------

predictor = TabularPredictor(label='Cluster',eval_metric=f1_scorer,problem_type='multiclass').fit(train_df.drop(columns=['Law Firm Name','Firm Name']),hyperparameters=hyperparameters,presets='medium_quality')

# COMMAND ----------

models = predictor.model_names()
model_name=[]
recall=[]
precision=[]
f1=[]
accuracy=[]
for model in models:
    result = predictor.evaluate(test_df, model=model)
    model_name.append(model)
    #recall.append(result['recall'])
    #precision.append(result['precision'])
    f1.append(result['f1'])
    #accuracy.append(result['accuracy'])

result_dict = {'model_name':model_name,  'f1':f1}
df_result = pd.DataFrame(result_dict)
df_result

# COMMAND ----------

test_data = test_df.copy()
test_data['predicted_cluster'] = predictor.predict(test_data.drop(columns=['Law Firm Name','Firm Name']))
test_data['proba'] = predictor.predict_proba(test_data).max(axis=1)

# COMMAND ----------

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_true = test_df['Cluster']
y_pred = test_data['predicted_cluster']

print("Classification Report:")
print(classification_report(y_true, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("Accuracy:")
print(accuracy_score(y_true, y_pred))

# COMMAND ----------

test_data['predicted_cluster'].value_counts(normalize=True)

# COMMAND ----------

#Feature importance on Test data
fi = predictor.feature_importance(
    data=test_df,              
    num_shuffle_sets=10, model='LightGBM')
fi = pd.DataFrame(fi)
fi

# COMMAND ----------

fi_clean = fi.reset_index()
fi_clean = fi_clean.rename(columns={'index': 'Feature'})
fi_clean

# COMMAND ----------

cols_to_drop = fi_clean.loc[(fi_clean['importance']<=0) & (fi_clean['p_value'] > 0.05)]['Feature'].tolist()
cols_to_drop

# COMMAND ----------

display(cluster_profiles)

# COMMAND ----------

cluster_profiles = df_firms.groupby('Cluster').agg(['count',list]).reset_index()
display(cluster_profiles)

# COMMAND ----------


import pandas as pd
import ast
import html

# 0) Ensure Firm Name column is present from test_df
test_data['Firm Name'] = test_df['Firm Name']

# 1) Flatten/standardize cluster_profiles columns robustly
def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        # Join non-empty parts with '_' and strip trailing underscores
        new_cols = []
        for tpl in df.columns:
            parts = [str(p).strip() for p in tpl if str(p).strip() != ""]
            col = "_".join(parts) if parts else ""
            new_cols.append(col)
        df = df.copy()
        df.columns = new_cols
    else:
        # just ensure they're strings
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
    return df

cluster_profiles_flat = flatten_columns(cluster_profiles)

# 2) Identify the key columns dynamically
#    - Cluster column: name contains 'cluster' (case-insensitive)
#    - Firm list column: name contains both 'firm' and 'list' (case-insensitive)
def find_col(cols, must_contain):
    """Find first column whose lowercase name contains ALL required substrings."""
    lc = [c.lower() for c in cols]
    for i, c in enumerate(lc):
        if all(sub in c for sub in must_contain):
            return list(cols)[i]
    return None

cluster_col = find_col(cluster_profiles_flat.columns, ['cluster'])

# Prefer (firm & list); if not found, try just 'list'
firm_list_col = find_col(cluster_profiles_flat.columns, ['firm', 'list'])
if firm_list_col is None:
    firm_list_col = find_col(cluster_profiles_flat.columns, ['list'])

if cluster_col is None or firm_list_col is None:
    raise KeyError(
        f"Could not detect required columns. "
        f"Available columns: {list(cluster_profiles_flat.columns)} "
        f"(need something like 'Cluster' and 'Firm Name_list')."
    )

# 3) Parse list cells + unescape HTML + normalize spacing
def parse_list_cell(cell):
    """
    Convert a stringified list to a Python list; if it's already a list, return as-is.
    Unescape HTML entities and strip whitespace.
    """
    if isinstance(cell, list):
        out = cell
    elif isinstance(cell, str):
        # Heuristic: if it looks like a list literal, try literal_eval; otherwise, treat as single item
        s = cell.strip()
        if (s.startswith('[') and s.endswith(']')) or (s.startswith('(') and s.endswith(')')):
            try:
                out = ast.literal_eval(s)
            except Exception:
                out = [cell]
        else:
            out = [cell]
    else:
        out = []
    return [html.unescape(x).strip() for x in out if isinstance(x, str)]

cluster_profiles_flat = cluster_profiles_flat.copy()
cluster_profiles_flat[firm_list_col] = cluster_profiles_flat[firm_list_col].apply(parse_list_cell)

# 4) Build dict: cluster -> set of firm names (case-sensitive; change if needed)
cluster_firm_dict = {
    row[cluster_col]: set(row[firm_list_col])
    for _, row in cluster_profiles_flat.iterrows()
}

# 5) Normalize names in test_data for HTML entities/whitespace
def norm_name(x):
    return html.unescape(x).strip() if isinstance(x, str) else x

test_data['Firm Name'] = test_data['Firm Name'].apply(norm_name)

# (Optional) Align cluster types if needed (e.g., predicted_cluster is str while cluster is int)
# Uncomment if you see mismatches:
# test_data['predicted_cluster'] = pd.to_numeric(test_data['predicted_cluster'], errors='coerce')

# 6) Your checker (unchanged logic, using detected keys)
def check_firm_in_predicted_cluster(row):
    cluster = row['predicted_cluster']
    firm = row['Firm Name']
    if cluster in cluster_firm_dict and firm in cluster_firm_dict[cluster]:
        return 1, firm
    else:
        return 0, None

test_data[['is_firm_in_predicted_cluster','matched_firm']] = (
    test_data.apply(check_firm_in_predicted_cluster, axis=1, result_type='expand')
)

display(test_data[['Firm Name','predicted_cluster','is_firm_in_predicted_cluster','matched_firm']])


# COMMAND ----------

#to validate firms
test_data['is_firm_in_predicted_cluster'].value_counts()

# COMMAND ----------


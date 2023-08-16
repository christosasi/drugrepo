#1. Creates knowledge graph with nodes as ChEMBL IDs and Edges as linkedDiseases. 
#2. Generates a bar plot of number of edges across loaded versions from the OpenTargets 'molecule datsets'
import pyarrow.dataset as ds
import networkx as nx
import time
import sys
import matplotlib.pyplot as plt

def create_graph_from_dataset(dataset_path):
    dataset = ds.dataset(dataset_path, format="parquet")
    G = nx.Graph()
    
    total_rows = dataset.to_table().num_rows
    processed_rows = 0
    start_time = time.time()

    #loop to filter for ChEMBL IDs of molecules with non-empty data in the 'linkedDiseases' column      
    for batch in dataset.to_table().to_batches():
        for molecule_id, linked_diseases in zip(batch.column('id'), batch.column('linkedDiseases')):
            if linked_diseases is not None and linked_diseases.as_py():
                diseases = linked_diseases.as_py()
                for disease in diseases.get('rows', []): 
                    G.add_edge(molecule_id.as_py(), disease) #Graph creation
        
        processed_rows += len(batch)
        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / processed_rows) * total_rows
        estimated_time_remaining = estimated_total_time - elapsed_time
        sys.stdout.write(f"\rEstimated time remaining for version {version}: {estimated_time_remaining:.2f} seconds")
        sys.stdout.flush()

    print(f"\nProcessing for version {version} complete!")
    return G

base_path = r'D:\OpenTargets datasets\parquet'
versions = ["21.04", "21.06", "21.09", "21.11", "22.02", "22.04", "22.06", "22.09", "22.11", "23.02", "23.06"]

nodes_data = []
edges_data = []

for version in versions:
    dataset_path = f"{base_path}\\{version}\\molecule"
    G = create_graph_from_dataset(dataset_path)
    nodes_data.append(G.number_of_nodes())
    edges_data.append(G.number_of_edges())

# Visualization
bar_width = 0.35
index = range(len(versions))

fig, ax = plt.subplots(figsize=(12, 6))
bar1 = ax.bar(index, nodes_data, bar_width, label='Nodes', color='b')
bar2 = ax.bar([i + bar_width for i in index], edges_data, bar_width, label='Edges', color='r')

# Adding data labels for nodes
for bar in ax.patches:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, '{:,}'.format(int(height)), ha='center', va='bottom')

# Adding data labels for edges
for bar in ax.patches:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, '{:,}'.format(int(height)), ha='center', va='bottom')

ax.set_xlabel('Version')
ax.set_ylabel('Count')
ax.set_title('Nodes and Edges count by version')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(versions, rotation=45)
ax.legend()

plt.tight_layout()
plt.show()

import pyarrow.dataset as pq
import pyarrow.compute as pc
import matplotlib.pyplot as plt

# Base path and versions list
base_path = 'D:\\OpenTargets datasets\\parquet\\'
versions = ['21.04', '21.06', '21.09', '21.11', '22.02', '22.04', '22.06', '22.09', '22.11', '23.02', '23.06']

# Function to get counts for a specific dataset and column
def get_counts(dataset_path, column_name, filter_value=None):
    counts = {}
    for version, path in dataset_path:
        try:
            data = pq.dataset(path, format="parquet")
            table = data.to_table(columns=[column_name])
            
            if filter_value is not None:
                value_filter = pc.equal(table.column(column_name), filter_value)
                filtered_table = table.filter(value_filter)
                counts[version] = filtered_table.num_rows
            else:
                unique_count = len(pc.unique(table.column(column_name)))
                counts[version] = unique_count
        except Exception as e:
            print(f"Error processing {version}: {e}")
    return counts

# Generate version paths
target_paths = [(version, base_path + version + '\\targets') for version in versions]
molecule_paths = [(version, base_path + version + '\\molecule') for version in versions]

# Get counts for each dataset
id_counts = get_counts(target_paths, 'id')
drug_count = get_counts(molecule_paths, 'isApproved', True)

# Create subplots for displaying graphs side-by-side
fig, axs = plt.subplots(1, 2, figsize=(15,6))

# Plotting for targets
axs[0].bar(id_counts.keys(), id_counts.values())
axs[0].set_xlabel('Versions')
axs[0].set_ylabel('Count of Targets')
axs[0].set_title('Target count in each version')
for i, count in enumerate(id_counts.values()):
    axs[0].text(i, count, str(count), ha='center', va='bottom')
axs[0].tick_params(axis='x', rotation=45)

# Plotting for molecules
axs[1].bar(drug_count.keys(), drug_count.values())
axs[1].set_xlabel('Versions')
axs[1].set_ylabel('Count of Approved Drugs')
axs[1].set_title('Approved Drug Count in each version')
for i, count in enumerate(drug_count.values()):
    axs[1].text(i, count, str(count), ha='center', va='bottom')
axs[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

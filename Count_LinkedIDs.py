#Prints a bar plot of linked diseases and linked targets across datasets (2021-2023)
import pyarrow.dataset as ds
import matplotlib.pyplot as plt

#
def get_counts(dataset_path, column_name):
    dataset = ds.dataset(dataset_path, format="parquet")
    total_count = 0
    
    for batch in dataset.to_table().to_batches():
        for linked_data in batch.column(column_name):
            if linked_data is not None and linked_data.as_py():
                total_count += linked_data.as_py().get('count', 0)
    
    return total_count

versions = ["21.04", "21.06", "21.09", "21.11", "22.02", "22.04", "22.06", "22.09", "22.11", "23.02", "23.06"]
base_path = r'D:\OpenTargets datasets\parquet'

# Collect total counts for linkedTargets and linkedDiseases for each version
targets_counts = [get_counts(f"{base_path}\\{version}\\molecule", 'linkedTargets') for version in versions]
diseases_counts = [get_counts(f"{base_path}\\{version}\\molecule", 'linkedDiseases') for version in versions]

# Plotting the data
bar_width = 0.35
index = range(len(versions))

plt.figure(figsize=(12, 6))
bar1 = plt.bar(index, targets_counts, bar_width, color='skyblue', label='Total Linked Targets')
bar2 = plt.bar([i+bar_width for i in index], diseases_counts, bar_width, color='salmon', label='Total Linked Diseases')

# Displaying the count values above each bar with commas
for idx, rect in enumerate(bar1):
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2., height, '{:,}'.format(int(height)), ha='center', va='bottom')

for idx, rect in enumerate(bar2):
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2., height, '{:,}'.format(int(height)), ha='center', va='bottom')

plt.xlabel('Version')
plt.ylabel('Total Count')
plt.title('Total Linked Targets and Diseases Count Across Versions')
plt.xticks([i+bar_width/2 for i in index], versions)
plt.legend()
plt.tight_layout()
plt.show()

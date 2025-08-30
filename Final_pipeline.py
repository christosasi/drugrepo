import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import Data, HeteroData
import datetime as dt
import torch_geometric.transforms as T
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from torch_geometric.data import Data
import random
from torch_geometric.nn import GCNConv, TransformerConv, SAGEConv
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from sklearn.metrics import confusion_matrix
import seaborn as sns
import argparse
import logging
import copy
import numpy as np
import sys
import time
import platform
import os
import polars as pl


training_version = 21.04
validation_version = 21.06
test_version = 22.04

#Chose the model
model_choice = TransformerConv #GCNConv #TransformerConv #SAGEConv

as_dataset = 'associationByOverallDirect'
disease_similarity_network = True
molecule_similarity_network =False
reactome_network = False
trial_edges= False
negative_sampling_approach = "BPR" #BPR #f"AS {as_dataset}" #random #BPR_AS

model_choices=("GCNConv" "SAGEConv" "TransformerConv")
as_datasets=("associationByDatasourceDirect" "associationByDatasourceIndirect" "associationByDatatypeDirect" "associationByDatatypeIndirect" "associationByOverallDirect" "associationByOverallIndirect")
negative_sampling_methods=("random", f"AS {as_dataset}", "BPR", f"BPR_{as_dataset}")


pipeline_parameters = [f"training_version: {training_version}", f"validation_version: {validation_version}",
                       f"test_version: {test_version}", f"as_dataset: {as_dataset}",
                       f"disease_similarity_network: {disease_similarity_network}",
                       f"molecule_similarity_network: {molecule_similarity_network}", 
                       f"trial_edges: {trial_edges}", f"negative_sampling_approach: {negative_sampling_approach}"]


# Detect the operating system and define paths accordingly
if platform.system() == "Windows":
    general_path = r"D:\\OpenTargets_datasets\\downloads\\"
    results_path = r"D:\\OpenTargets_datasets\\test_results\\"
    indication_path = f"{general_path}{training_version}\\indication"
    val_indication_path = f"{general_path}{validation_version}\\indication"
    test_indication_path = f"{general_path}{test_version}\\indication"
    molecule_path = f"{general_path}{training_version}\\molecule"
    disease_path = f"{general_path}{training_version}\\diseases"
    gene_path = f"{general_path}{training_version}\\targets"
    associations_path = f"{general_path}{training_version}/{as_dataset}"

else:
    general_path = "OT/"
    results_path = "test_results/"
    indication_path = f"{general_path}{training_version}/indication"
    val_indication_path = f"{general_path}{validation_version}/indication"
    test_indication_path = f"{general_path}{test_version}\\indication"
    molecule_path = f"{general_path}{training_version}/molecule"
    disease_path = f"{general_path}{training_version}/diseases"
    gene_path = f"{general_path}{training_version}/targets"
    associations_path = f"{general_path}{training_version}/{as_dataset}"


#Function to get indices from keys
def get_indices_from_keys(key_list, index_mapping):
    return [index_mapping[key] for key in key_list if key in index_mapping]

#Function to generate all possible edge combinations from 2 lists
def generate_pairs(source_list, target_list, source_mapping, target_mapping, return_set=False, return_tensor=False):
    edges = []
    for source_id in source_list:
        for target_id in target_list:
            edges.append((source_mapping[source_id], target_mapping[target_id]))
    if return_set:
        return set(edges)
    elif return_tensor: 
        edge_index_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index_tensor
    else: return edges
  

#Function to generate tensor for all possible edge combinations from 2 lists
def generate_tensor(source_list, target_list, source_mapping, target_mapping):
    edges = []
    for i in range(len(source_list)):
        edges.append((source_mapping[source_list[i]], target_mapping[target_list[i]]))
    edge_index_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index_tensor


# Function to extract edges from a table
def extract_edges(table, source_mapping, target_mapping, return_edge_list=False, return_edge_set=False):
    source = table.column(0).combine_chunks()
    targets = table.column(1).combine_chunks()
  
    edges = []
    for i in range(len(source)):
        source_id = source[i].as_py()  # Get the individual node ID
        target_list = targets.slice(i, 1).to_pylist()[0]  # Extract the target list for this source
        #Ensure that target_list is actually a list before iterating
        if not isinstance(target_list, list):
            target_list = [target_list]
        # Create a pair for each target and append it to the edges list
        for target_id in target_list:
            if source_id in source_mapping and target_id in target_mapping:
                edges.append((source_mapping[source_id], target_mapping[target_id]))
                
            elif source_id in source_mapping and target_id not in target_mapping:
                print(f"Could not find index for {target_id} in target_mapping")
                assert 2 ==3

            elif source_id not in source_mapping and target_id in target_mapping:
                print(f"Could not find index for {source_id} in source_mapping")
                assert 2 ==3

    if return_edge_list:
        return edges
    elif return_edge_set:
        return set(edges)
    else:
        edge_index_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index_tensor
    

# Function to extract edges test from a table
def extract_test_edges(table, source_mapping, target_mapping):
    source = table.column(0).combine_chunks()
    targets = table.column(1).combine_chunks()
  
    edges = []
    for i in range(len(source)):
        source_id = source[i].as_py()  # Get the individual node ID
        target_list = targets.slice(i, 1).to_pylist()[0]  # Extract the target list for this source
        #Ensure that target_list is actually a list before iterating
        if not isinstance(target_list, list):
            target_list = [target_list]
        # Create a pair for each target and append it to the edges list
        for target_id in target_list:
            if source_id in source_mapping and target_id in target_mapping:
                edges.append((source_mapping[source_id], target_mapping[target_id]))
                
    return set(edges)


#Function to find repurposing edges
def find_repurposing_edges(table1, table2, column_name, source_mapping, target_mapping):
    # Create a mask for filtering: True if element of table2['id'] is in table1['id']
    filter_mask = pc.is_in(table2.column('id'), value_set=table1.column('id'))
    # Use the filter Function to apply the mask to table2
    filtered_table = table2.filter(filter_mask)
    # Create new edge list
    new_edge_list = []
    for i in range(len(filtered_table)):
        row = filtered_table.slice(i, 1)
        drug_id = row.column('id').combine_chunks()[0].as_py()
        linked_items = row.column(column_name).combine_chunks()[0].as_py()

        for item in linked_items:
            if drug_id in source_mapping and item in target_mapping:
                new_edge_list.append((source_mapping[drug_id], target_mapping[item]))
    return new_edge_list


"""
val_tensor definition
f1 = (training_version diseases, training_version molecules with at least 1 indication, val md_edges (for nodes already in training_version))

vaL_edges =  all possible val md_edges (for nodes already in training_version)- training_version md edges 

test_tensor definition
f2 = (training_version diseases, training_version molecules with at least 1 indication, test md_edges (for nodes already in training_version))
test_edges = all possible test md_edges (for nodes already in training_version) - training_version md edges - val md edges


v1 = [(m1, d1), (m2, d2), (m3, d3), (m4, d4), (m5, d5), (0, d6)]


v2 = [(m1, d1 ), (m2, d3), (m3, d4), (m4, d5), (m1, d6)]

v1 - v2 = [(m1, d6)] 

"""

def generate_validation_tensors(training_version, validation_version, return_edge_sets=False, return_not_linked_set=False, return_val_edges=False):
    # Validation tensor extraction
#    val_indication_path = f"{general_path}{validation_version}\\indication"
    val_indication_dataset = ds.dataset(val_indication_path, format="parquet")
    val_indication_table = val_indication_dataset.to_table()
    #filter for approved drugs from training version
    expr1 = pc.is_in(val_indication_table.column('id'), value_set=approvedDrugs)

    val_filtered_indication_table = val_indication_table.filter(expr1)

    val_molecule_disease_table = val_filtered_indication_table.select(['id', 'approvedIndications']).flatten()
    all_val_md_edges_set = extract_edges(val_molecule_disease_table, drug_key_mapping, disease_key_mapping, return_edge_set=True)

    if return_val_edges:
        return all_val_md_edges_set    
    
    print("Validation set total:", len(all_val_md_edges_set))

    train_md_edges_set = extract_edges(molecule_disease_table, drug_key_mapping, disease_key_mapping, return_edge_set=True)
    print("Training set:", len(train_md_edges_set))

    new_val_edges_set = all_val_md_edges_set - train_md_edges_set
    print("Validation set new:", len(new_val_edges_set))    

    random.seed(42)
    disease_list_random = random.sample(disease_list, 1000)

    # Generate Cartesian product
    all_molecule_disease = generate_pairs(approved_drugs_list, disease_list_random, drug_key_mapping, disease_key_mapping)

    print("All molecule disease pairs:", len(all_molecule_disease))
    false_pairs = list(set(all_molecule_disease) - new_val_edges_set - train_md_edges_set)  # False pairs
    not_linked_set = list(set(all_molecule_disease) - train_md_edges_set)  # False pairs for not linked set
    print("False pairs:", len(false_pairs))

    
    true_pairs = list(new_val_edges_set)  # True pairs
    false_pairs = random.sample(false_pairs, len(true_pairs))

       
    # Create labels
    true_labels = [1] * len(true_pairs)
    false_labels = [0] * len(false_pairs)

    # Combine labels
    combined_labels = true_labels + false_labels

    # Create label tensor
    label_tensor = torch.tensor(combined_labels, dtype=torch.long)


    # Ensure the validation tensor is aligned with labels
    val_edge_tensor = torch.tensor(true_pairs + false_pairs, dtype=torch.long)
    
    if return_val_edges:
        return all_val_md_edges_set
    
    elif return_not_linked_set:
        return not_linked_set

    elif return_edge_sets:
        return  train_md_edges_set, new_val_edges_set
    else:     return val_edge_tensor, label_tensor


# Convert to a set

def generate_test_tensors(training_version, validation_version, test_version):
    train_md_edges_set, new_val_edges_set = generate_validation_tensors(training_version,validation_version, return_edge_sets=True)

    #Test tensor extraction
    
    test_indication_dataset = ds.dataset(test_indication_path, format="parquet")
    test_indication_table = test_indication_dataset.to_table()
    #filter for approved drugs from training version
    expr1 = pc.is_in(test_indication_table.column('id'), value_set=approvedDrugs)
    test_filtered_indication_table = test_indication_table.filter(expr1)
    test_molecule_disease_table = test_filtered_indication_table.select(['id', 'approvedIndications']).flatten()
    test_md_edges_set = extract_test_edges(test_molecule_disease_table, drug_key_mapping, disease_key_mapping)
    print("Test set total:", len(test_md_edges_set))
    all_val_md_edges_set = generate_validation_tensors(training_version, validation_version, return_edge_sets=False, return_not_linked_set=False, return_val_edges=True)
    new_test_edges = test_md_edges_set - all_val_md_edges_set
    print("Test set new:", len(new_test_edges))

    # Generate Cartesian product
    all_molecule_disease = generate_pairs(approved_drugs_list, disease_list[:2000], drug_key_mapping, disease_key_mapping)

    print("All molecule disease pairs:", len(all_molecule_disease))
    false_pairs = list(set(all_molecule_disease) - new_test_edges - train_md_edges_set - new_val_edges_set - new_test_edges)  # False pairs
    print("False pairs:", len(false_pairs))
    true_pairs = list(new_test_edges)  # True pairs

    random.seed(42)
    false_pairs = random.sample(false_pairs, len(true_pairs))

    # Create labels
    true_labels = [1] * len(true_pairs)
    false_labels = [0] * len(false_pairs)

    # Combine labels
    combined_labels = true_labels + false_labels

    # Create label tensor
    label_tensor = torch.tensor(combined_labels, dtype=torch.long)

    # Ensure the test tensor is aligned with labels
    test_edge_tensor = torch.tensor(true_pairs + false_pairs, dtype=torch.long)
    return test_edge_tensor, label_tensor

def boolean_encode(boolean_array, pad_length):
    # Convert to Pandas Series and reshape 
    boolean_series = boolean_array.to_pandas().to_numpy().reshape(-1, 1) 
    # Fill null values with -1
    boolean_series = pd.DataFrame(boolean_series)  # Explicitly create a new DataFrame
    boolean_series.fillna(-1, inplace=True)  # Modify in-place for efficiency

    # Convert to PyTorch tensor
    tensor = torch.from_numpy(boolean_series.to_numpy().astype(np.int64))

    # Calculate padding size
    max_length = len(pad_length)
    padding_size = max_length - tensor.shape[0]

    # Pad the tensor
    if padding_size > 0:
        padded_tensor = F.pad(tensor, (0, 0, 0, padding_size), value=-1)
    else:
        padded_tensor = tensor

    return padded_tensor

def normalize(array, pad_length):
    df = array.to_pandas().to_numpy().reshape(-1, 1)
    df = pd.DataFrame(df)  # Explicitly create a new DataFrame

    df.fillna(-1, inplace=True)  # Modify in-place for efficiency

    standardized = (df - df.mean()) / df.std()

    tensor = torch.from_numpy(standardized.to_numpy())

    max_length = len(pad_length)
    padding_size = max_length - tensor.shape[0]

    # Pad the tensor
    if padding_size > 0:
        padded_tensor = F.pad(tensor, (0, 0, 0, padding_size), value=-1)
    else:
        padded_tensor = tensor

    return padded_tensor

def cat_encode(array, pad_length):
    uni = array.unique().to_pandas()
    unidict = {uni[i]: i for i in range(len(uni))}
    
    tensor = torch.tensor([unidict[i] for i in array.to_pandas()], dtype=torch.int32)

    max_length = len(pad_length)
    padding_size = max_length - tensor.shape[0]

    # Pad the tensor
    if padding_size > 0:
        padded_tensor = F.pad(tensor, (0, 0, 0, padding_size), value=-1)
    else:
        padded_tensor = tensor

    return padded_tensor

# #Function to encode categorical variables
# def cat_encode(array):
#     uni = array.unique().to_pandas()
#     unidict = {uni[i]: i for i in range(len(uni))}
        
#     return torch.tensor([unidict[i] for i in array.to_pandas()], dtype=torch.int32)

#Function to generate word embeddings
def word_embeddings(array):
    array = [text if text is not None else "" for text in array.to_pylist()]
    batch_size = 32
    embeddings_list = []
    # load the tokenizer and model, and call the Function
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    for i in tqdm(range(0, len(array), batch_size), desc="Processing batches"):
        batch_texts = array[i:i+batch_size]
        encoded_input = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            output = model(**encoded_input)
        
        embeddings = output.last_hidden_state.mean(dim=1)
        embeddings_list.append(embeddings)

    embeddings_tensor = torch.cat(embeddings_list, dim=0)
    
    return embeddings_tensor

def one_hot_encode(node_type):
    feature_vector = torch.zeros(num_node_types)

    feature_vector[node_type] = 1
    return feature_vector.unsqueeze(0)

def pad_feature_matrix(matrix, pad_size, pad_value=-1):

    if matrix.size(1) < pad_size:
        padding = torch.ones(matrix.size(0), pad_size - matrix.size(1)) * -1  # Fill with -1
        matrix = torch.cat([matrix, padding], dim=1)
    return matrix


#Function to generate tensor for all possible edge combinations from 2 lists
def generate_edge_list(source_list, target_list, source_mapping, target_mapping):
   edges = []
   for i in range(len(source_list)):
       edges.append((source_mapping[source_list[i]], target_mapping[target_list[i]]))
   return edges


def as_negative_sampling(filtered_molecule_table, associations_table, score_column, return_list=False, return_set=False):
  
   # Table with Molecule and linked targets
   MLT = filtered_molecule_table.select(['id', 'linkedTargets.rows']).drop_null()

   # Table with Disease and linked targets and association scores
   DT = associations_table.select(['diseaseId', 'targetId', score_column])
   print("DT:", DT.slice(0,2).to_pandas(), len(DT)) 


   logging.info("Table with Disease and linked targets:", DT.slice(0,2).to_pandas(), len(DT))

   logging.info("Table with Molecule and linked targets:", MLT.slice(0,2).to_pandas(), len(MLT))

   #Convert to pandas DataFrames
   df_DT = DT.to_pandas()
   df_MLT= MLT.to_pandas()

   # Explode the 'linkedTargets.rows' column to create separate rows for each molecule, target pair
   df_MLT_exploded = df_MLT.explode('linkedTargets.rows').reset_index(drop=True)
   # Rename the column to 'targetId' for key matching during join
   df_MLT_exploded.rename(columns={'linkedTargets.rows': 'targetId'}, inplace=True)
   logging.info("Exploded Table with Molecule and linked targets:", df_MLT_exploded.head(2), len(df_MLT_exploded))
   MLT_exploded= pa.Table.from_pandas(df_MLT_exploded)

   # Convert MLT_exploded and DT to Polars DataFrames
   pl_MLT_exploded = pl.from_pandas(df_MLT_exploded)
   pl_DT = pl.from_pandas(df_DT)

   # Initialize an empty Polars DataFrame
   final_df = pl.DataFrame()

    # Memory management for big files
   if len(DT) > 1000000:
    # Divide the table into slices of 1 million rows each
    for i in tqdm(range(0, len(DT), 1000000), desc="Processing chunks with polars"):
        # Slice the tables
        t_DT = DT.slice(i, min(i + 1000000, len(DT)))
        MTD_table1 = MLT_exploded.slice(i, min(i + 1000000, len(MLT_exploded)))
        
        # Convert chunks to Polars DataFrames
        pl_t_DT = pl.from_arrow(t_DT)
        pl_MTD_table1 = pl.from_arrow(MTD_table1)
        
        # Join the tables on 'targetId'
        joined_chunk = pl_MTD_table1.join(pl_t_DT, on='targetId')
        
        # Sort the joined chunk by score_column
        sorted_chunk = joined_chunk.sort(score_column)
        
        # Concatenate the chunk to final_df
        final_df = pl.concat([final_df, sorted_chunk], how='vertical')
        MTD_table = final_df.to_arrow()
        print("Final table length:", len(MTD_table))

#    #Malloc Memory management for big files
#    if len(DT) > 1000000: 
#    #divide the table into slices fo 1 million rows each 
#        for i in tqdm(range(0, len(DT), 1000000), desc="Processing chunks without polars"):
#            t_DT = DT.slice(i, i+1000000)
#            MTD_table1 = MLT_exploded.slice(i, i+1000000)
#            MTD_table1 = MLT_exploded.join(t_DT, 'targetId').combine_chunks().sort_by(score_column)
#            # Create an empty table with the same schema as MTD_table1
#            empty_table_schema = MTD_table1.schema
#            MTD_table2 = pa.Table.from_arrays([pa.array([])] * len(empty_table_schema.names), schema=empty_table_schema)
#            # Concatenate the tables
#            MTD_table = pa.concat_tables([MTD_table1, MTD_table2])
#            print("Final table length:", len(MTD_table))
   
   else:
       MTD_table = MLT_exploded.join(DT, 'targetId').combine_chunks().sort_by(score_column)

   #Optional: Drop columns that are not needed for negative sampling
   logging.info("Table with Molecule, Disease and association scores:", MTD_table.to_pandas().head(2), len(MTD_table))
   expr = pc.field(score_column) <= pc.scalar(0.01001)
   logging.info(MTD_table.filter(expr).to_pandas().head(5))
   #Drop other columns to make a negative sample table
   negative_sample_table = MTD_table.drop_columns(['targetId',score_column]).drop_null()
   mlist = negative_sample_table.column('id').combine_chunks().to_pylist()
   dlist = negative_sample_table.column('diseaseId').combine_chunks().to_pylist()

   #create edge list
   ng_list = generate_edge_list(mlist, dlist, drug_key_mapping, disease_key_mapping)

   logging.info('Negative list created:', ng_list[0:5])
   if return_list:
         return ng_list
   elif return_set:   return set(ng_list)
   else: return torch.tensor(ng_list, dtype=torch.long).t().contiguous()


def custom_edges(disease_similarity_network, trial_edges, molecule_similarity_network,
                       filtered_disease_table, filtered_molecule_table,
                       disease_key_mapping, drug_key_mapping):

    # Initialize an empty list to collect all active edges
    custom_edges = []

    # Condition for disease similarity network edges
    if disease_similarity_network:
        disease_descendants_table = filtered_disease_table.select(['id', 'descendants']).flatten()
        disease_children_table = filtered_disease_table.select(['id', 'children']).flatten()
        disease_ancestors_table = filtered_disease_table.select(['id', 'ancestors']).flatten()

        # Extract edges and add to all_edges
        custom_edges.extend(extract_edges(disease_descendants_table, disease_key_mapping, disease_key_mapping, return_edge_list=True))
        custom_edges.extend(extract_edges(disease_children_table, disease_key_mapping, disease_key_mapping, return_edge_list=True))
        custom_edges.extend(extract_edges(disease_ancestors_table, disease_key_mapping, disease_key_mapping, return_edge_list=True))

        # Extract unique disease nodes 
        # disease_descendants = filtered_disease_table.column('descendants').combine_chunks().flatten().unique()
        # disease_children = filtered_disease_table.column('children').combine_chunks().flatten().unique()
        # disease_ancestors = filtered_disease_table.column('ancestors').combine_chunks().flatten().unique()

        # # Update disease key mapping with new nodes if do not already exist in the mapping
        # all_disease_ids = set(disease_descendants) | set(disease_children) | set(disease_ancestors)

        # for disease_id in all_disease_ids:
        #     disease_key_mapping.setdefault(disease_id, len(disease_key_mapping)) 


    # Condition for trial edges
    if trial_edges:
        molecule_trial_table = filtered_molecule_table.select(['id', 'linkedDiseases']).flatten()

        # Extract edges and add to all_edges
        custom_edges.extend(extract_edges(molecule_trial_table, drug_key_mapping, disease_key_mapping, return_edge_list=True))

        #update disease key mapping with new nodes if do not already exist in the mapping
        # trial_diseases = molecule_trial_table.column('linkedDiseases').combine_chunks().flatten().unique()

        # # Update disease key mapping with new nodes if do not already exist in the mapping
        # for disease_id in trial_diseases:
        #     disease_key_mapping.setdefault(disease_id, len(disease_key_mapping))      

    # Condition for molecule similarity network edges
    if molecule_similarity_network:
        molecule_parents_table = filtered_molecule_table.select(['id', 'parentId']).flatten()

        molecule_children_table = filtered_molecule_table.select(['id', 'childChemblIds']).flatten()

        # Extract edges and add to all_edges
        custom_edges.extend(extract_edges(molecule_parents_table, drug_key_mapping, drug_key_mapping, return_edge_list=True))
        custom_edges.extend(extract_edges(molecule_children_table, drug_key_mapping, drug_key_mapping, return_edge_list=True))

        #Extract unique molecule nodes
        # molecule_parents = filtered_molecule_table.column('parentId').combine_chunks().flatten().unique()
        # molecule_children = filtered_molecule_table.column('childChemblIds').combine_chunks().flatten().unique()
        
        # # Update drug key mapping with new nodes if do not already exist in the mapping
        # all_molecule_ids = set(molecule_parents) | set(molecule_children)
        # for molecule_id in all_molecule_ids:
        #     drug_key_mapping.setdefault(molecule_id, len(drug_key_mapping))
    custom_edge_tensor = torch.tensor(custom_edges, dtype=torch.long).t().contiguous()

    return custom_edge_tensor





# extract nodes from each dataset
indication_dataset = ds.dataset(indication_path, format="parquet")
indication_table = indication_dataset.to_table()

expr = pc.list_value_length(pc.field("approvedIndications")) > 0 
filtered_indication_table = indication_table.filter(expr)

approvedDrugs = filtered_indication_table.column('id').combine_chunks()
approvedIndications = filtered_indication_table.column('approvedIndications').combine_chunks()
unique_approved_indications = approvedIndications.flatten().unique()


molecule_dataset = ds.dataset(molecule_path, format="parquet")
molecule_table = molecule_dataset.to_table()




molecule_drugType_table = molecule_table.select(['id', 'drugType'])

#Replace 'unknown' with 'Unknown'
drug_type_column = pc.replace_substring(molecule_drugType_table[1], 'unknown', 'Unknown')
#Replace null with 'Unknown'
fill_value = pa.scalar('Unknown', type = pa.string())

molecule_table = molecule_table.drop_columns("drugType").add_column(3,"drugType", drug_type_column.fill_null(fill_value))

molecule_drugType_table = molecule_table.select(['id', 'drugType'])


all_moleculesin = molecule_table.column('id').combine_chunks()

#Filter for molecules that are present in the filtered indication dataset
molecule_filter = pc.is_in(molecule_table.column('id'), value_set= pc.unique(approvedDrugs))
filtered_molecule_table = molecule_table.filter(molecule_filter)

filtered_molecule_table = filtered_molecule_table.select(['id','name','drugType','blackBoxWarning','yearOfFirstApproval','parentId', 'childChemblIds', 'linkedDiseases', 'hasBeenWithdrawn', 'linkedTargets']).flatten().drop_columns(['linkedTargets.count', 'linkedDiseases.count'])

molecule = filtered_molecule_table.column('id').combine_chunks()

drug_type = molecule_table.column('drugType').combine_chunks()

if molecule_similarity_network == True:
    #Extract unique molecule nodes
    molecule_parents = molecule_table.column('parentId').combine_chunks().unique().drop_null()
    
    
    print("unique Molecule parents network nodes:", len(molecule_parents), type(molecule_parents))
    
    molecule_children = molecule_table.column('childChemblIds').combine_chunks().flatten()
    print("Molecule children network nodes:", len(molecule_children), type(molecule_children))
    
    # Add descendants, children and ancestors to the disease similarity network
    mf0 = all_moleculesin.to_pandas()
    mf1 = molecule_parents.to_pandas()
    mf2 = molecule_children.to_pandas()
    

    all_molecules_df = pd.concat([mf0, mf1, mf2], ignore_index=True).drop_duplicates()
    
    #convert to pyarrow array
    all_molecules = pa.array(all_molecules_df)

    approved_drugs_list = all_molecules.unique().to_pylist()
    print(f"{len(approved_drugs_list)} Molecules selected")
     
else: 

    approved_drugs_list = molecule.to_pylist()
    print(f"{len(approved_drugs_list)} Molecules selected")
    

#extract the linked genes from the linkedTargets column
molecules_linked_genes_table = filtered_molecule_table.select(['id','linkedTargets.rows']).drop_null()
molecules_linked_genes = molecules_linked_genes_table.column('id').combine_chunks()


print("Number of molecules with linked genes:", len(molecules_linked_genes))

#extract the linked genes from the linkedTargets column
linked_genes = filtered_molecule_table.column('linkedTargets.rows').combine_chunks()

#Gene dataset
gene_dataset = ds.dataset(gene_path, format="parquet")
gene_table = gene_dataset.to_table().flatten().flatten()

"""Test Block"""
node_columns = ['id','proteinAnnotations.subcellularLocations','proteinAnnotations.functions']

gene_table = gene_table.select(node_columns)
print(gene_table.to_pandas())
print(len(gene_table.column(1).combine_chunks().flatten().unique()))

assert 2 == 3

#filter for genes linked to approved drugs
gene_filter_mask = pc.is_in(gene_table.column('id'), value_set= pc.unique(linked_genes.flatten()))
filtered_gene_table = gene_table.filter(gene_filter_mask)
#filtered_gene_table = gene_table

#cases for different training versions
if training_version == 21.04 or training_version == 21.06:
    filtered_gene_table = filtered_gene_table.select(['id', 'approvedName','bioType', 'proteinAnnotations.functions', 'reactome']).flatten()
    gene_reactome_table = filtered_gene_table.select(['id', 'reactome']).flatten()

else: 
    filtered_gene_table = filtered_gene_table.select(['id', 'approvedName','biotype', 'FunctionDescriptions', 'proteinIds', 'pathways'])
    filtered_gene_table = filtered_gene_table.select(['id', 'approvedName','biotype', 'FunctionDescriptions', 'proteinIds', 'pathways']).flatten()
    gene_reactome_table = filtered_gene_table.select(['id', 'pathways']).flatten().to_pandas()
    exploded = gene_reactome_table.explode('pathways')
    # Step 2: Extract the 'pathwayId' from the dictionaries in the 'pathways' column
    exploded['pathwayId'] = exploded['pathways'].apply(lambda x: x['pathwayId'] if pd.notnull(x) else None)
    # Step 3: Create a new DataFrame with just the 'id' and 'pathwayId' columns
    final_df = exploded[['id', 'pathwayId']]
    # Step 4: Convert the pandas DataFrame back to a PyArrow Table
    gene_reactome_table = pa.Table.from_pandas(final_df)
    gene_reactome_table = gene_reactome_table.drop_null()
    

if training_version == 21.04 or training_version == 21.06:
    proteinAnnotations = filtered_gene_table.column('proteinAnnotations.functions').combine_chunks()
else:
    proteinAnnotations = filtered_gene_table.column('FunctionDescriptions').combine_chunks()

#Gene nodes
gene = filtered_gene_table.column('id').combine_chunks()

if 'pathways' in filtered_gene_table.column_names:
    reactome = filtered_gene_table.column('pathways').combine_chunks().flatten()
    reactome = reactome.field(0)

else:
    reactome = filtered_gene_table.column('reactome').combine_chunks().flatten()


#Disease dataset
disease_dataset = ds.dataset(disease_path, format="parquet")
disease_table = disease_dataset.to_table()
#disease_filter_mask = pc.is_in(disease_table.column('id'), value_set= pc.unique(approvedIndications.flatten()))
#filtered_disease_table = disease_table.filter(disease_filter_mask)
disease_table = disease_table.select(['id', 'name', 'description', 'ancestors', 'descendants', 'children', 'therapeuticAreas'])

description = disease_table.column('description').combine_chunks()

#Approved disease nodes
disease = disease_table.column('id').combine_chunks()

#Include non approved diseases
if disease_similarity_network == True:
    disease_descendants = disease_table.column('descendants').combine_chunks().flatten()
    disease_children = disease_table.column('children').combine_chunks().flatten()
    disease_ancestors = disease_table.column('ancestors').combine_chunks().flatten()

    # Add descendants, children and ancestors to the disease similarity network
    df0 = disease.to_pandas()
    df1 = disease_descendants.unique().to_pandas()
    df2 = disease_children.unique().to_pandas()
    df3 = disease_ancestors.unique().to_pandas()

    #concatenate all the dataframes and story only unique values
    all_diseases_df = pd.concat([df0, df1, df2, df3], ignore_index=True).drop_duplicates()
    #convert to pyarrow array
    all_diseases = pa.array(all_diseases_df)
    logging.info("Disease similarity network nodes:", len(all_diseases_df))


# #Exclude non approved diseases
therapeutic_area = disease_table.column('therapeuticAreas').combine_chunks().flatten()


#Load associations dataset
# Options include 'associationByDatasourceDirect', 'associationByDatasourceIndirect', 'associationByDatatypeDirect', 'associationByDatatypeIndirect', 'associationByOverallDirect', 'associationByOverallIndirect'
associations_dataset = ds.dataset(associations_path, format="parquet")
associations_table = associations_dataset.to_table()

# Define Score column from associations table

for col in associations_table.column_names:
   if "Score" in col:
       logging.info(col)
       score_column = col


#Edge case for training version 21.04
if training_version == 21.04:
   associations_table = associations_table.select(['diseaseId', 'targetId', score_column])
else:
   associations_table = associations_table.select(['diseaseId', 'targetId', score_column])

#Filter for associations for genes linked with approved drugs
gene_filter_mask = pc.is_in(associations_table.column('targetId'), value_set= pc.unique(linked_genes.flatten()))
gene_filtered_associations_table = associations_table.filter(gene_filter_mask)

#Filter for associations for diseases with approved drugs 
disease_filter_mask = pc.is_in(gene_filtered_associations_table.column('diseaseId'), value_set= pc.unique(disease.unique()))
filtered_associations_table = gene_filtered_associations_table.filter(disease_filter_mask)

if training_version == 21.04:
   score_threshold = pc.field(score_column) >= 0.01
else:
   score_threshold = pc.field(score_column) >= 0.01

# Filter for associations with a score greater than or equal to threshold
filtered_associations_table = filtered_associations_table.filter(score_threshold)


drug_type_list = drug_type.drop_null().unique().to_pylist()

gene_list = gene.unique().to_pylist()
reactome_list = reactome.unique().to_pylist()
#disease_list from disease similarity network
if disease_similarity_network:
    disease_list = all_diseases.unique().to_pylist()
else:
    disease_list = disease.unique().to_pylist()

therapeutic_area_list = therapeutic_area.unique().to_pylist()



node_info = {}

# Add node_info as key value pairs
node_info["Drugs"] = len(approved_drugs_list)
node_info["Drug Types"] = len(drug_type_list)
node_info["Genes"] = len(gene_list)
node_info["Reactome pathways"] = len(reactome_list)
node_info["Diseases"] = len(disease_list)
node_info["Therapeutic areas"] = len(therapeutic_area_list)

print(node_info)


drug_key_mapping = {approved_drugs_list[i]: i for i in range(len(approved_drugs_list))}
drug_type_key_mapping = {drug_type_list[i]: i + len(drug_key_mapping) for i in range(len(drug_type_list))}
gene_key_mapping = {gene_list[i]: i + len(drug_key_mapping) + len(drug_type_key_mapping) for i in range(len(gene_list))}
reactome_key_mapping = {reactome_list[i]: i + len(drug_key_mapping) + len(drug_type_key_mapping) + len(gene_key_mapping) for i in range(len(reactome_list))}
disease_key_mapping = {disease_list[i]: i + len(drug_key_mapping) + len(drug_type_key_mapping) + len(gene_key_mapping) + len(reactome_key_mapping) for i in range(len(disease_list))}
therapeutic_area_key_mapping = {therapeutic_area_list[i]: i + len(drug_key_mapping) + len(drug_type_key_mapping) + len(gene_key_mapping) + len(reactome_key_mapping) + len(disease_key_mapping) for i in range(len(therapeutic_area_list) )}



dict_path = r"D:\\OpenTargets_datasets\\test_results2\\dict_path"

#save all key_mappings in csvs at dict_path
# pd.DataFrame.from_dict(drug_key_mapping, orient='index').to_csv(f"{dict_path}/drug_key_mapping.csv")
# pd.DataFrame.from_dict(drug_type_key_mapping, orient='index').to_csv(f"{dict_path}/drug_type_key_mapping.csv")
# pd.DataFrame.from_dict(gene_key_mapping, orient='index').to_csv(f"{dict_path}/gene_key_mapping.csv")
# pd.DataFrame.from_dict(reactome_key_mapping, orient='index').to_csv(f"{dict_path}/reactome_key_mapping.csv")
# pd.DataFrame.from_dict(disease_key_mapping, orient='index').to_csv(f"{dict_path}/disease_key_mapping.csv")
# pd.DataFrame.from_dict(therapeutic_area_key_mapping, orient='index').to_csv(f"{dict_path}/therapeutic_area_key_mapping.csv")

a_list = ['associationByDatasourceDirect', 'associationByDatasourceIndirect', 'associationByDatatypeDirect', 'associationByDatatypeIndirect', 'associationByOverallDirect', 'associationByOverallIndirect']

"""
for a in a_list:
    associations_path = f"{general_path}{training_version}/{a}"
    associations_dataset = ds.dataset(associations_path, format="parquet")
    associations_table = associations_dataset.to_table()
    for col in associations_table.column_names:
        if "Score" in col:
            logging.info(col)
            score_column = col
    not_linked_set = as_negative_sampling(filtered_molecule_table, associations_table, score_column, return_set=True)
    print(a , len(not_linked_set))
"""



#Start of feature block ---------------------------------
# Define node types
drug_node_type = 0
drug_type_node_type = 1
gene_node_type = 2
disease_node_type = 3
reactome_node_type = 4
therapeutic_area_node_type = 5
num_node_types = 6

# Get indices for each node_type
drug_indices = torch.tensor(get_indices_from_keys(approved_drugs_list, drug_key_mapping), dtype=torch.long)
drug_type_indices = torch.tensor(get_indices_from_keys(drug_type_list, drug_type_key_mapping), dtype=torch.long)
gene_indices = torch.tensor(get_indices_from_keys(gene_list, gene_key_mapping), dtype=torch.long)
reactome_indices = torch.tensor(get_indices_from_keys(reactome_list, reactome_key_mapping), dtype=torch.long)
disease_indices = torch.tensor(get_indices_from_keys(disease_list, disease_key_mapping), dtype=torch.long)
therapeutic_area_indices = torch.tensor(get_indices_from_keys(therapeutic_area_list, therapeutic_area_key_mapping), dtype=torch.long)

#Feature extraction 

# Drug feature extraction
if molecule_similarity_network:
    blackBoxWarning = molecule_table.column('blackBoxWarning').combine_chunks()
else:
    blackBoxWarning = filtered_molecule_table.column('blackBoxWarning').combine_chunks()

blackBoxWarning_vector = boolean_encode(blackBoxWarning, drug_indices)

#pad vector with -1 to make it the same length as drug_indices

# drug_name = filtered_molecule_table.column('name').combine_chunks()
# drug_name_vector = word_embeddings(drug_name)
if molecule_similarity_network:
    yearOfFirstApproval = molecule_table.column('yearOfFirstApproval').combine_chunks()
else:
    yearOfFirstApproval = filtered_molecule_table.column('yearOfFirstApproval').combine_chunks()

yearOfFirstApproval_vector = normalize(yearOfFirstApproval, drug_indices)


if molecule_similarity_network:
    hasBeenWithdrawn = molecule_table.column('hasBeenWithdrawn').combine_chunks()
else:
    hasBeenWithdrawn = filtered_molecule_table.column('hasBeenWithdrawn').combine_chunks()

hasBeenWithdrawn_vector = boolean_encode(hasBeenWithdrawn, drug_indices)


drug_one_hot = [1.0, 0.0 , 0.0, 0.0, 0.0, 0.0]
drug_node_type_vector = torch.tensor([drug_one_hot], dtype=torch.float32).repeat(len(drug_indices), 1)  # Resulting shape [length, 6]

#Concatenate the feature vectors along columns (dim=1)
#drug_feature_matrix = torch.cat((blackBoxWarning_vector, yearOfFirstApproval_vector, hasBeenWithdrawn_vector, drug_name_vector, drug_node_type_vector), dim=1)
drug_feature_matrix = torch.cat((blackBoxWarning_vector, yearOfFirstApproval_vector, hasBeenWithdrawn_vector, drug_node_type_vector), dim=1)

pad_size = drug_feature_matrix.shape[1]


# Gene feature extraction
gene_indices = torch.tensor(get_indices_from_keys(gene_list, gene_key_mapping), dtype=torch.long)
# gene_name = filtered_gene_table.column('approvedName').combine_chunks()
# gene_name_vector = word_embeddings(gene_name)
if training_version == 21.04 or training_version == 21.06 or training_version == 21.09:
    bioType = filtered_gene_table.column('bioType').combine_chunks()
else:
    bioType = filtered_gene_table.column('biotype').combine_chunks()

bioType_vector = cat_encode(bioType, gene_indices).unsqueeze(1)

gene_one_hot = [0.0, 0.0 , 1.0, 0.0, 0.0, 0.0]


gene_node_type_vector = torch.tensor([gene_one_hot], dtype=torch.float32).repeat(len(gene_indices), 1)  # Resulting shape [length, 6]
#gene_feature_matrix = torch.cat((bioType_vector, gene_name_vector, gene_node_type_vector), dim=1)
gene_feature_matrix = torch.cat((bioType_vector, gene_node_type_vector), dim=1)




# Disease feature extraction
disease_indices = torch.tensor(get_indices_from_keys(disease_list, disease_key_mapping), dtype=torch.long)
#disease_name_vector = word_embeddings(disease_name)
disease_one_hot = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
disease_node_type_vector = torch.tensor([disease_one_hot], dtype=torch.float32).repeat(len(disease_indices), 1)  # Resulting shape [length, 6]
#disease_feature_matrix = torch.cat((disease_name_vector, disease_node_type_vector), dim=1)
disease_feature_matrix = disease_node_type_vector


#Nodes without features also need a feature matrix
# drugType feature extraction
drug_type_one_hot = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
drug_type_feature_matrix = torch.tensor([drug_type_one_hot], dtype=torch.float32).repeat(len(drug_type_indices), 1)  # Resulting shape [length, 6]

# reactome feature extraction
reactome_one_hot = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
reactome_feature_matrix = torch.tensor([reactome_one_hot], dtype=torch.float32).repeat(len(reactome_indices), 1)  # Resulting shape [length, 6]

# therapeutic_area feature extraction
therapeutic_area_one_hot = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
therapeutic_area_feature_matrix = torch.tensor([therapeutic_area_one_hot], dtype=torch.float32).repeat(len(therapeutic_area_indices), 1)  # Resulting shape [length, 6]

# Feature list
feature_map = {}
feature_map["Drug Features"] = ['blackBoxWarning', 'yearOfFirstApproval', 'hasBeenWithdrawn', 'node_type']
feature_map["Drug type Features"] = ['node_type']
feature_map["Disease Features"] = ['node_type']
feature_map["Gene Features"] = ['bioType', 'node_type']
feature_map["Reactome Features"] = ['node_type']
feature_map["Therapeutic area Features"] = ['node_type']    


# Pad feature matrices
drug_type_feature_matrix = pad_feature_matrix(drug_type_feature_matrix, pad_size, -1)
gene_feature_matrix = pad_feature_matrix(gene_feature_matrix, pad_size, -1)
disease_feature_matrix = pad_feature_matrix(disease_feature_matrix, pad_size, -1)
reactome_feature_matrix = pad_feature_matrix(reactome_feature_matrix, pad_size, -1)
therapeutic_area_feature_matrix = pad_feature_matrix(therapeutic_area_feature_matrix, pad_size, -1)

# Assuming feature matrices can be stacked without compatibility issues
all_features = torch.vstack([drug_feature_matrix, drug_type_feature_matrix, disease_feature_matrix, gene_feature_matrix, reactome_feature_matrix, therapeutic_area_feature_matrix])
print("Size of feature matrix: ", all_features.shape)

#End of feature block ---------------------------------


# Edge extraction

# Default Edge tables
molecule_drugType_table = filtered_molecule_table.select(['id', 'drugType']).drop_null().flatten()
molecule_disease_table = filtered_indication_table.select(['id', 'approvedIndications']).flatten()
molecule_gene_table = filtered_molecule_table.select(['id', 'linkedTargets.rows']).drop_null().flatten()
gene_reactome_table = gene_reactome_table #extracted according to version above
disease_therapeutic_table = disease_table.select(['id', 'therapeuticAreas']).drop_null().flatten()
disease_gene_table = filtered_associations_table.select(['diseaseId', 'targetId']).flatten()




# Extract edges as tensors
molecule_drugType_edges = extract_edges(molecule_drugType_table, drug_key_mapping, drug_type_key_mapping)
molecule_disease_edges = extract_edges(molecule_disease_table, drug_key_mapping, disease_key_mapping)
molecule_gene_edges = extract_edges(molecule_gene_table, drug_key_mapping, gene_key_mapping)
gene_reactome_edges = extract_edges(gene_reactome_table, gene_key_mapping, reactome_key_mapping) 
disease_therapeutic_edges = extract_edges(disease_therapeutic_table, disease_key_mapping, therapeutic_area_key_mapping)
disease_gene_edges = extract_edges(disease_gene_table, disease_key_mapping, gene_key_mapping)


#Extract edges as lists
molecule_drugType_edge_list = extract_edges(molecule_drugType_table, drug_key_mapping, drug_type_key_mapping, return_edge_list=True)
molecule_disease_edge_list = extract_edges(molecule_disease_table, drug_key_mapping, disease_key_mapping, return_edge_list=True)
molecule_gene_edge_list = extract_edges(molecule_gene_table, drug_key_mapping, gene_key_mapping, return_edge_list=True)
gene_reactome_edge_list = extract_edges(gene_reactome_table, gene_key_mapping, reactome_key_mapping, return_edge_list=True)
disease_therapeutic_edge_list = extract_edges(disease_therapeutic_table, disease_key_mapping, therapeutic_area_key_mapping, return_edge_list=True)
disease_gene_edge_list = extract_edges(disease_gene_table, disease_key_mapping, gene_key_mapping, return_edge_list=True)

#save lists to csvs at dict_path
# pd.DataFrame(molecule_drugType_edge_list).to_csv(dict_path + "/molecule_drugType_edge_list.csv")
# pd.DataFrame(molecule_disease_edge_list).to_csv(dict_path + "/molecule_disease_edge_list.csv")
# pd.DataFrame(molecule_gene_edge_list).to_csv(dict_path + "/molecule_gene_edge_list.csv")
# pd.DataFrame(gene_reactome_edge_list).to_csv(dict_path + "/gene_reactome_edge_list.csv")
# pd.DataFrame(disease_therapeutic_edge_list).to_csv(dict_path + "/disease_therapeutic_edge_list.csv")
# pd.DataFrame(disease_gene_edge_list).to_csv(dict_path + "/disease_gene_edge_list.csv")


if molecule_similarity_network :
    molecule_parents_table = molecule_table.select(['id', 'parentId']).drop_null().flatten()
    molecule_children_table = molecule_table.select(['id', 'childChemblIds']).drop_null().flatten()
    
    # print("Molecule parents table without null:", molecule_parents_table.to_pandas().head(2), type(molecule_parents_table))
    # print("Molecule children table without null:", molecule_children_table.to_pandas().head(2), type(molecule_children_table))

    molecule_children_edges = extract_edges(molecule_children_table, drug_key_mapping, drug_key_mapping)
    molecule_parents_edges = extract_edges(molecule_parents_table, drug_key_mapping, drug_key_mapping)


if disease_similarity_network :
    disease_descendants_table = disease_table.select(['id', 'descendants']).flatten().drop_null()
    disease_children_table = disease_table.select(['id', 'children']).flatten().drop_null()
    disease_ancestors_table = disease_table.select(['id', 'ancestors']).flatten().drop_null()
    disease_descendants_edges = extract_edges(disease_descendants_table, disease_key_mapping, disease_key_mapping)
    disease_children_edges = extract_edges(disease_children_table, disease_key_mapping, disease_key_mapping)
    disease_ancestors_edges = extract_edges(disease_ancestors_table, disease_key_mapping, disease_key_mapping)


#store length of each edge list in a dictionary
edge_info = {}
edge_info["Molecule to Drug Type edges"] = len(molecule_drugType_edges[0])
edge_info["Molecule to disease edges"] = len(molecule_disease_edges[0])
edge_info["Molecule to gene edges"] = len(molecule_gene_edges[0])
edge_info["Gene to reactome edges"] = len(gene_reactome_edges[0])
edge_info["Disease to therapeutic edges"] = len(disease_therapeutic_edges[0])
edge_info["Disease to gene edges"] = len(disease_gene_edges[0])

print(edge_info)

if molecule_similarity_network == True:
    edge_info["Molecule_Parents edges"] = len(molecule_parents_edges[0])
    edge_info["Molecule_Children edges"] = len(molecule_children_edges[0])

if disease_similarity_network == True:
    edge_info["Disease_Descendants edges"]= len(disease_descendants_edges[0])
    edge_info["Disease_Children edges"] = len(disease_children_edges[0])
    edge_info["Disease_Ancestors edges"] = len(disease_ancestors_edges[0])

print(edge_info)

all_edge_index = torch.cat([molecule_drugType_edges, molecule_disease_edges, molecule_gene_edges, gene_reactome_edges, 
                                disease_therapeutic_edges, disease_gene_edges], dim=1)

# Concatenate edge tensors based on custom edge conditions
if disease_similarity_network: 
    all_edge_index = torch.cat([all_edge_index, disease_descendants_edges, disease_children_edges, disease_ancestors_edges], dim=1)

if molecule_similarity_network:
    all_edge_index = torch.cat([all_edge_index, molecule_parents_edges, molecule_children_edges], dim=1)





# Generate validation and label tensors
val_edge_tensor, label_tensor = generate_validation_tensors(training_version, validation_version)

existing_drug_disease_edges = list(zip(molecule_disease_edges[0].tolist(), molecule_disease_edges[1].tolist()))
existing_drug_disease_edges_set = set(existing_drug_disease_edges)

# Convert the positive edge index to a tensor
pos_edge_index = torch.tensor(existing_drug_disease_edges)

all_possible_set = generate_pairs(approved_drugs_list, disease_list[:2000], drug_key_mapping, disease_key_mapping, return_set=True)
print(f"Number of possible drug-disease pairs: {len(all_possible_set)}")


datetime = dt.datetime.now().strftime("%Y%m%d%H%M%S")


#Graph metadata


# All graph metadata
metadata = {"node_info" : node_info, "feature_map": feature_map, "edge_info" : edge_info}

# Create the homogenous graph object
graph = Data(x=all_features, edge_index=all_edge_index, val_edge_index=val_edge_tensor, val_edge_label=label_tensor, metadata=metadata)
graph = T.ToUndirected()(graph)  # Convert to undirected graph
print("Graph Validated:", graph.validate())
print(graph)

""" 
Isolated block
from torch_geometric.utils import degree


# Calculate degree of each node
node_degrees = degree(graph.edge_index[0], num_nodes=graph.num_nodes)

# Calculate average degree
avg_degree = node_degrees.mean().item()

print(f"Average Node Degree: {avg_degree}")

# Find isolated nodes (nodes with degree 0)
isolated_nodes = (node_degrees == 0).nonzero(as_tuple=True)[0].tolist()

print(f"First 10 isolated nodes: {type(isolated_nodes)} {isolated_nodes[:10]}")

# Extract keys that have values matching the indices in the list
isolated_diseases = [key for key, value in disease_key_mapping.items() if value in isolated_nodes]

disease_table2 = disease_dataset.to_table()


isolated_diseases_array= pa.array(isolated_diseases)

#filter pyarrow table based on values in isolated_diseases
isolated_mask = pc.is_in(disease_table2.column('id'), value_set = isolated_diseases_array)
isolated_diseases_table = disease_table2.filter(isolated_mask)


isd_df= isolated_diseases_table.to_pandas()

#save isd_df to csv
isd_df.to_csv(f"D:\OpenTargets_datasets\graphs\isolated_diseases_{training_version}.csv")


#extract keys of isolated nodes from disease key mapping
#for i in isolated_nodes print value of i in disease key mapping
#$for i, value in isolated_nodes:
    #isolateddisease_key_mapping[value])

"""

# Save the homogenous graph object with date and time in file name

# file_name = f"D:\OpenTargets_datasets\graphs\homo_graph{datetime}_{training_version}.pt"
# torch.save(graph, file_name)
# print(f"Saved graph for {training_version} to {file_name}")

# Load the graph
graph.x = graph.x.float()  # Ensure node features are float32
graph.edge_index = graph.edge_index.long()  # This is usually already the case and is fine as is


# Extract the class name
model_name = model_choice(0,0).__class__.__name__

torch.manual_seed(13)



# # # Define the GNN skeleton model
if negative_sampling_approach == "random" or f"AS {as_dataset}":


    class GNN(torch.nn.Module):
        
        def __init__(self, in_channels, hidden_channels, out_channels): #, out_channels is needed some times with random sampling but why?
            super(GNN, self).__init__()
            self.conv1 = model_choice(in_channels, hidden_channels)
            self.num_layer = 10
            self.conv_list = torch.nn.ModuleList([model_choice(hidden_channels, hidden_channels) for _ in range(self.num_layer - 1)])
            self.ln = torch.nn.LayerNorm(hidden_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = self.ln(x)
            if not self.num_layer == 1:
                x = F.relu(x)  
            for k in range(self.num_layer - 1):
                x = self.conv_list[k](x, edge_index)
                x = self.ln(x)
                if not k == self.num_layer - 2:
                    x = F.relu(x)
            return x

if negative_sampling_approach == "BPR" or f"BPR {as_dataset}":

    class GNN(torch.nn.Module):
        
        def __init__(self, in_channels, hidden_channels, out_channels):
            super(GNN, self).__init__()
            self.conv1 = model_choice(in_channels, hidden_channels)
            self.num_layer = 10
            self.conv_list = torch.nn.ModuleList([model_choice(hidden_channels, hidden_channels) for _ in range(self.num_layer - 1)])
            self.ln = torch.nn.LayerNorm(hidden_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = self.ln(x)
            if not self.num_layer == 1:
                x = F.relu(x)  
            for k in range(self.num_layer - 1):
                x = self.conv_list[k](x, edge_index)
                x = self.ln(x)
                if not k == self.num_layer - 2:
                    x = F.relu(x)
            return x







#Model for performing BPR
#########################################################
class BPR(nn.Module):
    def __init__(self, drug_num, disease_num, factor_num):
        super(BPR, self).__init__()
        self.embed_user = nn.Embedding(drug_num, factor_num)
        self.embed_item = nn.Embedding(disease_num, factor_num)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    # def forward(self, user, item_i, item_j):
    #     user_embeddings = self.embed_user(user)
    #     item_i_embeddings = self.embed_item(item_i)
    #     item_j_embeddings = self.embed_item(item_j)
    #     prediction_i = (user_embeddings * item_i_embeddings).sum(dim=-1)
    #     prediction_j = (user_embeddings * item_j_embeddings).sum(dim=-1)
    #     return prediction_i, prediction_j

    def forward_custom_embeddings(self, user_embeddings, item_i_embeddings, item_j_embeddings):
        prediction_i = (user_embeddings * item_i_embeddings).sum(dim=-1)
        prediction_j = (user_embeddings * item_j_embeddings).sum(dim=-1)
        # Example check inside forward_custom_embeddings
        # assert user_embeddings.requires_grad and item_i_embeddings.requires_grad and item_j_embeddings.requires_grad, "Embeddings do not require gradients."

        return prediction_i, prediction_j
   


class GNN_BPR(nn.Module):
    def __init__(self, in_channels, hidden_channels, factor_num, drug_num, disease_num):
        super(GNN_BPR, self).__init__()
        self.gnn = GNN(in_channels, hidden_channels, factor_num)
        self.bpr = BPR(drug_num, disease_num, factor_num)
   
    def forward(self, drug_ids, disease_ids_i, disease_ids_j, graph_data):
        node_embeddings = self.gnn(graph_data.x, graph_data.edge_index)
        drug_embeddings = node_embeddings[drug_ids]
        disease_embeddings_i = node_embeddings[disease_ids_i]
        disease_embeddings_j = node_embeddings[disease_ids_j]

        assert drug_embeddings.size(0) == disease_embeddings_i.size(0) == disease_embeddings_j.size(0), "Mismatch in batch sizes"

        prediction_i, prediction_j = self.bpr.forward_custom_embeddings(drug_embeddings, disease_embeddings_i, disease_embeddings_j)
       
        return prediction_i, prediction_j, node_embeddings
   

def bpr_loss(prediction_positive, prediction_negative):
    """
    Calculate Bayesian Personalized Ranking (BPR) loss.
    """
    loss = -torch.mean(torch.log(torch.sigmoid(prediction_positive - prediction_negative) + 1e-15))
    return loss


def generate_triplets_with_sets(existing_pairs_set, not_linked_set):

    # Create a dictionary to store possible diseases for each drug from not-linked pairs
    not_linked_dict = {}
    for drug, disease in not_linked_set:
        if drug in not_linked_dict:
            not_linked_dict[drug].append(disease)
        else:
            not_linked_dict[drug] = [disease]

    # Generate triplets
    results = []
    for drug, linked_disease in existing_pairs_set:
        possible_diseases = not_linked_dict.get(drug, [])
        if possible_diseases:
            not_linked_disease = np.random.choice(possible_diseases)
            results.append((drug, linked_disease, not_linked_disease))
    return results

# Define the device to be loaded
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
graph = graph.to(device)



# Initialize the model based on the negative sampling approach
if negative_sampling_approach == "random":
   print(f"{negative_sampling_approach} sampling")
   model = GNN(in_channels=graph.x.size(1), hidden_channels=16, out_channels=1).to(device)
   # Create negative drug-disease pairs
   ng_tensor_edges = list(all_possible_set - existing_drug_disease_edges_set)

elif negative_sampling_approach == "BPR":
   print(f"{negative_sampling_approach} negative sampling")
   model = GNN_BPR(in_channels=graph.x.size(1), hidden_channels=16, factor_num=16, drug_num=len(drug_indices), disease_num=len(disease_indices)).to(device)
   # Generate random training triplets
   not_linked_set = all_possible_set - existing_drug_disease_edges_set
   training_triplets_f = []
   training_triplets_f = generate_triplets_with_sets(existing_drug_disease_edges_set, not_linked_set)
        
elif negative_sampling_approach == f"BPR_{as_dataset}":
   print(f"{negative_sampling_approach} sampling")
   model = GNN_BPR(in_channels=graph.x.size(1), hidden_channels=8, factor_num=8, drug_num=len(drug_indices), disease_num=len(disease_indices)).to(device)
   #Generate triplets from Association scores
   not_linked_set = as_negative_sampling(filtered_molecule_table, associations_table, score_column, return_set=True)
   training_triplets_f = []
   training_triplets_f = generate_triplets_with_sets(existing_drug_disease_edges_set, not_linked_set)
        

elif negative_sampling_approach == f"AS {as_dataset}":
   print(f"{negative_sampling_approach} sampling")
   model = GNN(in_channels=graph.x.size(1), hidden_channels=16, out_channels=1).to(device)
   ng_tensor_edges = as_negative_sampling(filtered_molecule_table, associations_table, score_column, return_list=True)
   # Association scheme neg_tensor
   # Resize list for first 5000 negative samples from ng_tensor_edges
   ng_tensor_edges = ng_tensor_edges[:5000]


# Initialize the model

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



print("Starting training...")

recalls_training = []
recall_vals = []
false_positive_rates = []
false_positive_rate_val = []
false_positive_rates_val = []
mean_pos_scores = []
mean_pos_scores_val = []
mean_neg_scores = []
mean_neg_scores_val = []
val_performances = []
val_performances_val = []
auc_scores_training = []
auc_scores_val = []
losses_val = []
apr_scores_training = []
apr_scores_val = []
accuracy_training = []
accuracies_training = []
accuracy_val = []
accuracies_val = []
losses_training = []


# Early stopping variables
# best_recall_val = 0.0
# bad_epochs = 0
# patience = 50  


for epoch in tqdm(range(10), desc='Training Progress'):

    total_loss = 0
    model.train()
    optimizer.zero_grad()
    
    if negative_sampling_approach == "random":
        z = model(graph.x.float(), graph.edge_index)
        #Extract positive and negative scores
        pos_scores = (z[pos_edge_index[:,0]] * z[pos_edge_index[:,1]]).sum(dim=-1)
        neg_edge_index = torch.tensor(random.sample(ng_tensor_edges, len(existing_drug_disease_edges)))
        neg_scores = (z[neg_edge_index[:,0]] * z[neg_edge_index[:,1]]).sum(dim=-1)
    
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores + 1e-7, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores + 1e-7, torch.zeros_like(neg_scores))
        loss = 0.01 * pos_loss + neg_loss

        # Backward and optimize
        loss.backward()
        optimizer.step()
   
    elif negative_sampling_approach == f"AS {as_dataset}":
        z = model(graph.x.float(), graph.edge_index)
        neg_edge_index = torch.tensor(random.sample(ng_tensor_edges, len(existing_drug_disease_edges)))
        pos_scores = (z[pos_edge_index[:,0]] * z[pos_edge_index[:,1]]).sum(dim=-1)
        #plot pos_scores
        neg_scores = (z[neg_edge_index[:,0]] * z[neg_edge_index[:,1]]).sum(dim=-1)
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores + 1e-7, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores + 1e-7, torch.zeros_like(neg_scores))
        loss = 0.01 * pos_loss + neg_loss

        # Backward and optimize
        loss.backward()
        optimizer.step()

    elif negative_sampling_approach == "BPR" or f"BPR_{as_dataset}":
        # Generate balanced training triplets for the current epoch
        #randomly sample from training_triplets list
        training_triplets = random.sample(training_triplets_f, len(training_triplets_f))


        drug_ids = torch.tensor([t[0] for t in training_triplets], dtype=torch.long, device=device)
        disease_ids_i = torch.tensor([t[1] for t in training_triplets], dtype=torch.long, device=device)
        disease_ids_j = torch.tensor([t[2] for t in training_triplets], dtype=torch.long, device=device)

        # Forward pass through the model
        prediction_i, prediction_j, z = model(drug_ids, disease_ids_i, disease_ids_j, graph)
        pos_scores = prediction_i.detach()
        neg_scores = prediction_j.detach()
        
        # Compute BPR loss
        loss = bpr_loss(prediction_i, prediction_j)
        loss.backward()
        optimizer.step()
        # optimizer.zero_grad()  # Clear gradients for the next iteration

        # Log the average loss per epoch
        avg_loss = total_loss / len(training_triplets)
        print(f'Epoch {epoch+1}: Avg Loss: {avg_loss:.4f}')
        # Metrics calculation




    train_label_tensor = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)
    concat_scores = torch.sigmoid(torch.cat([pos_scores, neg_scores], dim=0))
    true_positives = torch.sum(pos_scores > 0).float()
    false_positives = torch.sum(neg_scores > 0).float()
    true_negatives = torch.sum(neg_scores <= 0).float()
    false_positive_rate = false_positives / (false_positives + true_negatives)
    mean_pos_score = torch.mean(pos_scores)
    mean_neg_score = torch.mean(neg_scores)
    recall = true_positives / pos_scores.size(0)
    apr_score_tr = average_precision_score(train_label_tensor.cpu().numpy(), concat_scores.detach().cpu().numpy())
    auc_score_tr = roc_auc_score(train_label_tensor.cpu().numpy(), concat_scores.detach().cpu().numpy())
   
    binary_predictions = (concat_scores >= 0.75).float()
    correct_predictions = binary_predictions.eq(train_label_tensor).sum().item()
    accuracy_training = correct_predictions / len(train_label_tensor)

       
    # Store metrics
    recalls_training.append(recall.item())
    false_positive_rates.append(false_positive_rate.item())
    mean_pos_scores.append(mean_pos_score.item())
    mean_neg_scores.append(mean_neg_score.item())
    apr_scores_training.append(apr_score_tr)
    auc_scores_training.append(auc_score_tr)
    accuracies_training.append(accuracy_training)
   
    # Print metrics
    logging.info(f"Epoch {epoch+1}/{1000}")
    logging.info(f"Recall: {recall.item()}")
    logging.info(f"False Positive Rate: {false_positive_rate.item()}")
    logging.info(f"Mean Positive Score: {mean_pos_score.item()}")
    logging.info(f"Mean Negative Score: {mean_neg_score.item()}")
    logging.info(f"APR: {apr_score_tr}")
    logging.info(f"AUC: {auc_score_tr}")




    # Store and print loss
    losses_training.append(loss.item())
    print(f"Loss: {loss.item()}")

    # if epoch == max
    if epoch % 1 == 0 and epoch > 0:
    #if epoch % 1 == 0:
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            
            if negative_sampling_approach == "random" or negative_sampling_approach == f"AS {as_dataset}":
                z = model(graph.x.float(), graph.edge_index)

            elif negative_sampling_approach == "BPR" or f"BPR_{as_dataset}":
                z = model.forward(drug_ids, disease_ids_i, disease_ids_j, graph)[-1]

            
            val_scores = (z[val_edge_tensor[:,0]] * z[val_edge_tensor[:,1]]).sum(dim=-1)
            #plot distribution of val_scores
            # Compute metrics
            val_probs = torch.sigmoid(val_scores)
            
            # wrap in a loop where you change 0.5 , use mean/median of val probs 
            val_threshold = val_probs.mean().item()
            
            # True Positives, False Positives, False Negatives, RECALL and FPR
            val_preds = (val_probs >= val_threshold).float()  # Use 0.5 as threshold to decide on positive/negative predictions
            
            # True Positives, False Positives, False Negatives
            TP = (val_preds * label_tensor).sum().item()
            FP = (val_preds * (1 - label_tensor)).sum().item()
            FN = ((1 - val_preds) * label_tensor).sum().item()
            TN = ((1 - val_preds) * (1 - label_tensor)).sum().item()
            accuracy_val = (TP + TN) / (TP + TN + FP + FN)
            recall_val = TP / (TP + FN) if TP + FN > 0 else 0
            false_positive_rate_val = FP / (FP + (1 - label_tensor).sum().item()) if FP + (1 - label_tensor).sum().item() > 0 else 0
            # APR and AUC
            apr_score_val = average_precision_score(label_tensor.cpu().numpy(), val_probs.cpu().numpy())
            """    
            display = PrecisionRecallDisplay.from_predictions(
                train_label_tensor.cpu().numpy(), concat_scores.cpu().numpy(), name="GNN"
            )
            _ = display.ax_.set_title("2-class Precision-Recall curve - training")
            plt.show()

            display = PrecisionRecallDisplay.from_predictions(
                label_tensor.cpu().numpy(), val_probs.cpu().numpy(), name="GNN"
            )
            _ = display.ax_.set_title("2-class Precision-Recall curve - validation")
            # 
            plt.show()
            
            #Plot histpogram of val_probs to understand the range of the predictions

        
            precision, recall, thresholds = precision_recall_curve(
                label_tensor.cpu().numpy(), val_probs.cpu().numpy())
            print(thresholds)
            print(precision)
            print(recall)
            print(val_probs.cpu().numpy().std())
            print(len(thresholds), len(precision), len(recall) )
            plt.plot(recall[:10], precision[:10], label="precision-recall curve")
            plt.show()
            sns.histplot(val_probs.cpu().numpy(), kde=True)            
            plt.show()

            """
            auc_score_val = roc_auc_score(label_tensor.cpu().numpy(), val_probs.cpu().numpy())
            
        # Log metrics
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}, Validation APR: {apr_score_val}, Validation AUC: {auc_score_val}, Recall: {recall_val}, Accuracy: {accuracy_val}')
        
        
        recall_vals.append(recall_val)
        false_positive_rates_val.append(false_positive_rate_val)
        mean_pos_scores_val.append(pos_scores.mean().item())
        mean_neg_scores_val.append(neg_scores.mean().item())
        auc_scores_val.append(auc_score_val)
        apr_scores_val.append(apr_score_val)
        accuracies_val.append(accuracy_val)


                # Early stopping check
        # if recall_val > best_recall_val:
        #     best_recall_val = recall_val
        #     bad_epochs = 0
        # else:
        #     bad_epochs += 1

        # if bad_epochs >= patience:
        #     print("Early stopping")
        #     break


# Plot the training performance
# Create a figure and a 2x2 grid of subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.plot(accuracies_training)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy')
# Plot recall
ax2.plot(recalls_training)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Recall')
ax2.set_title('Recall')
ax3.plot(apr_scores_training)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('APR')
ax3.set_title('APR')
# Plot AUC
ax4.plot(auc_scores_training)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('AUC')
ax4.set_title('AUC')
fig.suptitle(f'{model_name} training version: {training_version}_{negative_sampling_approach}_{as_dataset}')
# Show the plot
plt.tight_layout()
plt.show(block=False)

# save the plots
fig.savefig(f'{results_path}{model_name}_{training_version}_{negative_sampling_approach}_{as_dataset}_training_performance.png')

# Plot the validation performance
# Create a figure and a 2x2 grid of subplots
fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.plot(accuracies_val)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy')
ax2.plot(recall_vals)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Recall')
ax2.set_title('Recall')
#plot APR
ax3.plot(apr_scores_val)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('APR')
ax3.set_title('APR')
# Plot AUC
ax4.plot(auc_scores_val)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('AUC')
ax4.set_title('AUC')
fig2.suptitle(f'{model_name} Validation Version: {validation_version}_{negative_sampling_approach}_{as_dataset}')
# Show the plot
plt.tight_layout()

# save the validation plot
fig2.savefig(f'{results_path}{model_name}_{validation_version}_{negative_sampling_approach}_{as_dataset}_{datetime}_validation_performance.png')
plt.show(block=False)



from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Convert probabilities to binary predictions
threshold = 0.75
binary_predictions = (val_scores >= threshold).float()

# Generate the confusion matrix
cm = confusion_matrix(label_tensor.cpu(), binary_predictions.cpu())  # Assuming label_tensor is defined
#Declare the negative_sampling_approach
# Plot the confusion matrix using Seaborn
plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
sns.set(font_scale=2)  # Adjust font scale here (1.4 is just an example, increase or decrease as needed)
ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Validation version {validation_version} {model_name} {negative_sampling_approach}", size=20)  # Adjust title font size here
plt.xlabel("Predicted Label", size=18)  # Adjust x-axis label font size here
plt.ylabel("True Label", size=18)  # Adjust y-axis label font size here
plt.xticks(fontsize=16)  # Adjust x-axis ticks font size here
plt.yticks(fontsize=16)  # Adjust y-axis ticks font size here
plt.savefig(f'{results_path}{model_name}_{validation_version}_{negative_sampling_approach}_confusion_matrix_{as_dataset}_{datetime}.png')
plt.show(block=False)


model.eval()

# Test the model on the test set
test_edges_tensor, test_label_tensor = generate_test_tensors(training_version, validation_version, test_version)
# Load the model
with torch.no_grad():
    
    if negative_sampling_approach == "random" or negative_sampling_approach == f"AS {as_dataset}":
        z = model(graph.x.float(), graph.edge_index)
    
    elif negative_sampling_approach == "BPR" or f"BPR_{as_dataset}":
        z = model.forward(drug_ids, disease_ids_i, disease_ids_j, graph)[-1]
    
    test_scores = (z[test_edges_tensor[:,0]] * z[test_edges_tensor[:,1]]).sum(dim=-1)
    #plot distribution of val_scores
    # Compute metrics
    test_probs = torch.sigmoid(test_scores)
            
    # wrap in a loop where you change 0.5 , use mean/median of val probs 
    test_threshold = test_probs.mean().item()
            
    # True Positives, False Positives, False Negatives, RECALL and FPR
    test_preds = (test_probs >= test_threshold).float()

# True Positives, False Positives, False Negatives
    TP = (test_preds * test_label_tensor).sum().item()
    FP = (test_preds * (1 - test_label_tensor)).sum().item()
    FN = ((1 - test_preds) * test_label_tensor).sum().item()
    TN = ((1 - test_preds) * (1 - test_label_tensor)).sum().item()
    accuracy_test = (TP + TN) / (TP + TN + FP + FN)
    recall_test = TP / (TP + FN) if TP + FN > 0 else 0
    false_positive_rate_test = FP / (FP + (1 - test_label_tensor).sum().item()) if FP + (1 - test_label_tensor).sum().item() > 0 else 0

    # APR and AUC
    apr_score_test = average_precision_score(test_label_tensor.cpu().numpy(), test_probs.cpu().numpy())
    auc_score_test = roc_auc_score(test_label_tensor.cpu().numpy(), test_probs.cpu().numpy())
print(f'Test APR: {apr_score_test}, Test AUC: {auc_score_test}, Recall: {recall_test}, Accuracy: {accuracy_test}')


apr_score_test = [apr_score_test]
auc_score_test = [auc_score_test]
recall_test = [recall_test]
accuracy_test = [accuracy_test]

FP_indices = (test_preds * (1 - test_label_tensor)).bool()
#print(FP_indices)

# rank the false positives by probability
FP_probs = test_probs[FP_indices]
FP_indices = FP_indices.nonzero(as_tuple=True)[0]
FP_probs, FP_indices = zip(*sorted(zip(FP_probs, FP_indices), reverse=True))

#print(FP_probs)
FP_disease_indices = test_edges_tensor[FP_indices, 1]
FP_drug_indices = test_edges_tensor[FP_indices, 0]
FP_disease_indices = disease_indices[FP_disease_indices]
FP_drug_indices = drug_indices[FP_drug_indices]
FP_disease_keys = [disease_list[i] for i in FP_disease_indices]
FP_drug_keys = [approved_drugs_list[i] for i in FP_drug_indices]

molecule_name_table = molecule_table.select(['id', 'name'])
disease_name_table = disease_table.select(['id', 'name']).combine_chunks()

#Initialize empty dataframe
predicted_links = pd.DataFrame()


# Get the names of the drugs and diseases from molecule_name_table and disease_name_table according to the test_drugs and test_diseases
for i in range(10):
    expr1 = pc.field("id") == FP_drug_keys[i]
    expr2 = pc.field("id") == FP_disease_keys[i]
    predicted_drug = molecule_name_table.filter(expr1).column("name").to_pylist()[0]
    predicted_disease = disease_name_table.filter(expr2).column("name").to_pylist()[0]

        # Create a new row as a dataframe
    new_row = pd.DataFrame({'Drug': [predicted_drug], 'Disease': [predicted_disease]})
    
    # Concatenate the new row to the dataframe
    predicted_links = pd.concat([predicted_links, new_row], ignore_index=True)


print(predicted_links)

assert 21 == 22


# Plot the test performance
test_threshold = 0.75
binary_predictions = (test_scores >= test_threshold).float()
cm = confusion_matrix(test_label_tensor.cpu(), binary_predictions.cpu())  # Assuming label_tensor is defined
plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
sns.set_theme(font_scale=2)  # Adjust font scale here (1.4 is just an example, increase or decrease as needed)
ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"{model_name} {negative_sampling_approach} Test_version {test_version}", size=20)  # Adjust title font size here
plt.xlabel("Predicted Label", size=18)  # Adjust x-axis label font size here
plt.ylabel("True Label", size=18)  # Adjust y-axis label font size here
plt.xticks(fontsize=16)  # Adjust x-axis ticks font size here
plt.yticks(fontsize=16)  # Adjust y-axis ticks font size here
plt.tight_layout()
plt.show(block=False)


#save the confusion matrix
plt.savefig(f'{results_path}_{model_name}_Test_{test_version}_{negative_sampling_approach}_{as_dataset}_confusion_matrix_{datetime}.png')


logging.info(f" Test Confusion matrix saved to {results_path}_{model_name}_Test:{test_version}_{negative_sampling_approach}_{as_dataset}_confusion_matrix_{datetime}.png")

#Print Results to Excel
#create list of results but with only the last value in the list
feature_info = feature_map

# Define results lists
results_lists = [
    [pipeline_parameters], [node_info], [edge_info], [feature_info], [model_name], [negative_sampling_approach],
    [apr_scores_training[-1]], [auc_scores_training[-1]], [accuracies_training[-1]],
    [apr_scores_val[-1]], [auc_scores_val[-1]], [accuracies_val[-1]],
    [recalls_training[-1]], [recall_vals[-1]], [auc_score_test[-1]],
    [apr_score_test[-1]], [recall_test[-1]], [accuracy_test[-1]]
]


# Define column titles
results_columns = [
    "Pipeline Parameters", "Nodes", "Edges", "Features", "Model", "Negative Sampling",
    "Training-APR", "Training-AUC", "Training Accuracy",
    "Validation-APR", "Validation-AUC", "Validation Accuracy",
    "Recall-Training", "Recall-Validation", "Test AUC",
    "Test APR", "Test Recall", "Test Accuracy"
]

# Create the DataFrame
results_df = pd.DataFrame(results_lists, index=results_columns).T

# Identify which columns should be multiplied by 100 (numeric columns)
numeric_columns = [
    "Training-APR", "Training-AUC", "Training Accuracy",
    "Validation-APR", "Validation-AUC", "Validation Accuracy",
    "Recall-Training", "Recall-Validation", "Test AUC",
    "Test APR", "Test Recall", "Test Accuracy"
]

# Multiply only the numeric columns by 100
results_df[numeric_columns] = results_df[numeric_columns].apply(pd.to_numeric, errors='coerce') * 100
results_df[numeric_columns] = results_df[numeric_columns].astype(str) + '%'



# Construct the filename dynamically
results_filename = f"{model_name}_TR_{training_version}_Val_{validation_version}_Test_{test_version}_{negative_sampling_approach}_{as_dataset}_{datetime}.csv"
full_path = os.path.join(results_path, results_filename)

# Check if file exists to determine whether to write the header
write_header = not os.path.isfile(full_path)

# Save to CSV file, appending to it if it already exists
results_df.to_csv(full_path, mode='a', index=False, header=write_header)


# Set up logging
logging_filename = f"{model_name}_TR_{training_version}_Val_{validation_version}_Test_{test_version}_{negative_sampling_approach}_{as_dataset}_{datetime}.log"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Only capture INFO and above

file_handler = logging.FileHandler(logging_filename)
file_handler.setLevel(logging.INFO)  # Match the logger's level

formatter = logging.Formatter('%(asctime)s - %(message)s')  # Simple format
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

print(f"Iteration complete. Results saved to {results_path}{results_filename}")






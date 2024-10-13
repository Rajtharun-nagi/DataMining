import pandas as pd
from itertools import combinations
import time
import glob
from mlxtend.frequent_patterns import apriori, association_rules
from efficient_apriori import apriori as efficient_apriori

# Function to read CSV files and create transactions
def load_transactions_from_csv(path_pattern):
    transactions = []
    for file_path in glob.glob(path_pattern):
        df = pd.read_csv(file_path, header=None)
        for index, row in df.iterrows():
            transactions.append(row.dropna().tolist())
    return transactions

# Load all transaction datasets
databases = {
    "Amazon": load_transactions_from_csv('Amazon_Updated_Transactions.csv'),
    "WallMart": load_transactions_from_csv('supermarket_transactions_2.csv'),
    "K-Mart": load_transactions_from_csv('KMart_Updated_Transactions.csv'),
    "Best Buy": load_transactions_from_csv('BestBuy_Updated_Transactions.csv'),
    "Nike": load_transactions_from_csv('Nike_Updated_Transactions.csv'),
}

# Prompt the user to select a database
print("\nAvailable Databases:")
for i, db_name in enumerate(databases.keys(), start=1):
    print(f"{i}. {db_name}")

db_selection = int(input("\nSelect a database by entering the corresponding number: ")) - 1
selected_db_name = list(databases.keys())[db_selection]
transactions = databases[selected_db_name]

print(f"\n{'='*30}\nProcessing {selected_db_name}...\n{'='*30}")

# Get user input for minimum support and confidence as percentages
min_support = float(input("Enter minimum support (e.g., 20 for 20%): ")) / 100
min_confidence = float(input("Enter minimum confidence (e.g., 50 for 50%): ")) / 100

def get_frequent_itemsets_brute_force(transactions, min_support):
    itemsets = {}
    item_count = len(transactions)

    # Generate 1-itemsets
    print("Generating 1-itemsets...")
    for transaction in transactions:
        for item in transaction:
            itemsets[frozenset([item])] = itemsets.get(frozenset([item]), 0) + 1

    # Filter 1-itemsets
    frequent_itemsets = {item: count for item, count in itemsets.items() if count / item_count >= min_support}
    
    print(f"Found {len(frequent_itemsets)} frequent 1-itemsets.")
    
    k = 1
    while True:
        print(f"Generating {k + 1}-itemsets from {k}-itemsets...")
        # Generate k-itemsets
        k_itemsets = combinations(frequent_itemsets.keys(), k + 1)
        new_frequent_itemsets = {}

        for itemset in k_itemsets:
            union_set = frozenset.union(*itemset)
            count = sum(1 for transaction in transactions if union_set.issubset(transaction))
            if count / item_count >= min_support:
                new_frequent_itemsets[union_set] = count
                print(f"  Frequent {k + 1}-itemset found: {set(union_set)}, Support: {count / item_count:.4f}")

        if not new_frequent_itemsets:
            print(f"No more frequent {k + 1}-itemsets found. Stopping.")
            break

        print(f"Found {len(new_frequent_itemsets)} frequent {k + 1}-itemsets.")
        frequent_itemsets.update(new_frequent_itemsets)
        k += 1

    return frequent_itemsets

def generate_association_rules(frequent_itemsets, min_confidence):
    rules = []
    for itemset, count in frequent_itemsets.items():
        support = count / len(transactions)
        if len(itemset) > 1:  # We only want rules from itemsets with more than one item
            for item in itemset:
                antecedent = itemset - frozenset([item])
                if antecedent:
                    confidence = count / frequent_itemsets.get(antecedent, count)
                    if confidence >= min_confidence:
                        rules.append((antecedent, frozenset([item]), confidence, support))
    return rules

### 1. Brute Force Algorithm
print("\nRunning Brute Force Algorithm...")
start_time = time.time()
frequent_itemsets_brute_force = get_frequent_itemsets_brute_force(transactions, min_support)
rules_brute_force = generate_association_rules(frequent_itemsets_brute_force, min_confidence)
brute_force_time = time.time() - start_time

print("\nFrequent Itemsets (Brute Force):")
for itemset, count in frequent_itemsets_brute_force.items():
    support = count / len(transactions)
    print(f"  Itemset: {set(itemset)}, Support: {support:.4f}")

print(f"\nAssociation Rules (Brute Force):")
for antecedent, consequent, confidence, support in rules_brute_force:
    print(f"  Rule: {set(antecedent)} -> {set(consequent)}, Confidence: {confidence:.4f}, Support: {support:.4f}")

print(f"Brute Force Algorithm Completed in {brute_force_time:.4f} seconds.\n")

### 2. Apriori Algorithm (using mlxtend)
print("\nRunning Apriori Algorithm...")
onehot = pd.get_dummies(pd.DataFrame([[item for item in transaction] for transaction in transactions]).stack()).groupby(level=0).sum().astype(bool)

# Debugging output for the one-hot encoded DataFrame
print("One-Hot Encoded DataFrame (First 5 Rows):")
print(onehot.head())
print("One-Hot Encoded DataFrame Shape:", onehot.shape)

start_time = time.time()
frequent_itemsets_apriori = apriori(onehot, min_support=min_support, use_colnames=True)

if frequent_itemsets_apriori.empty:
    print("No frequent itemsets found using Apriori.")
else:
    print("Frequent Itemsets Found using Apriori:")
    for _, row in frequent_itemsets_apriori.iterrows():
        print(f"  Itemset: {set(row['itemsets'])}, Support: {row['support']:.4f}")

    rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=min_confidence)
    print("\nAssociation Rules Found using Apriori:")
    for _, row in rules_apriori.iterrows():
        print(f"  Rule: {row['antecedents']} -> {row['consequents']}, Support: {row['support']:.4f}, Confidence: {row['confidence']:.4f}")

apriori_time = time.time() - start_time
print(f"Apriori Algorithm Completed in {apriori_time:.4f} seconds.\n")

### 3. FP-Growth Algorithm (using efficient_apriori)
print("\nRunning FP-Growth Algorithm...")
start_time = time.time()
frequent_itemsets_fpgrowth, rules_fpgrowth = efficient_apriori(transactions, min_support=min_support, min_confidence=min_confidence)

if not frequent_itemsets_fpgrowth:
    print("No frequent itemsets found using FP-Growth.")
else:
    print("Frequent Itemsets Found using FP-Growth:")
    for itemset in frequent_itemsets_fpgrowth:
        support = sum(1 for transaction in transactions if itemset.issubset(transaction)) / len(transactions)
        print(f"  Itemset: {set(itemset)}, Support: {support:.4f}")

    print("\nAssociation Rules Found using FP-Growth:")
    for antecedent, consequent, confidence in rules_fpgrowth:
        support = sum(1 for transaction in transactions if antecedent.union(consequent).issubset(transaction)) / len(transactions)
        print(f"  Rule: {set(antecedent)} -> {set(consequent)}, Confidence: {confidence:.4f}, Support: {support:.4f}")

fpgrowth_time = time.time() - start_time
print(f"FP-Growth Algorithm Completed in {fpgrowth_time:.4f} seconds.\n")

# Timing Performance Comparison
print(f"\nTiming Performance for {selected_db_name}:")
print(f"  Brute Force Time: {brute_force_time:.4f} seconds")
print(f"  Apriori Time: {apriori_time:.4f} seconds")
print(f"  FP-Growth Time: {fpgrowth_time:.4f} seconds")


# In[ ]:





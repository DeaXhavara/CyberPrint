from datasets import load_dataset

dataset = load_dataset("go_emotions")

# Print the first example from train split
print("Sample row:")
print(dataset["train"][0])

# Print the corresponding text and raw labels
text = dataset["train"][0]["text"]
labels = dataset["train"][0]["labels"]

print("\nText:", text)
print("Raw Label IDs:", labels)

# Print out the label names available
label_names = dataset["train"].features["labels"].feature.names
print("\nLabel Names:", label_names)

# Decode label IDs into names
decoded_labels = [label_names[i] for i in labels]
print("Decoded Labels:", decoded_labels)

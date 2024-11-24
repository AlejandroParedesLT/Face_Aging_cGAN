# List of subgroup files
subgroup_files = [
    "D:/temporal_CV/1. AIPI 590 - Computer Vision/assignment/3. third_project/Face-Aging-with-Identity-Preserved-Conditional-Generative-Adversarial-Networks/data/preprocessed-wiki-lists/train_age_group_0.txt",
    "D:/temporal_CV/1. AIPI 590 - Computer Vision/assignment/3. third_project/Face-Aging-with-Identity-Preserved-Conditional-Generative-Adversarial-Networks/data/preprocessed-wiki-lists/train_age_group_1.txt",
    "D:/temporal_CV/1. AIPI 590 - Computer Vision/assignment/3. third_project/Face-Aging-with-Identity-Preserved-Conditional-Generative-Adversarial-Networks/data/preprocessed-wiki-lists/train_age_group_2.txt",
    "D:/temporal_CV/1. AIPI 590 - Computer Vision/assignment/3. third_project/Face-Aging-with-Identity-Preserved-Conditional-Generative-Adversarial-Networks/data/preprocessed-wiki-lists/train_age_group_3.txt",
    "D:/temporal_CV/1. AIPI 590 - Computer Vision/assignment/3. third_project/Face-Aging-with-Identity-Preserved-Conditional-Generative-Adversarial-Networks/data/preprocessed-wiki-lists/train_age_group_4.txt",
]

# Reference file
reference_file = "D:/temporal_CV/1. AIPI 590 - Computer Vision/assignment/3. third_project/Face-Aging-with-Identity-Preserved-Conditional-Generative-Adversarial-Networks/data/preprocessed-wiki-lists/train.txt"

# Read the reference file into a set
with open(reference_file, 'r') as ref_file:
    reference_lines = set(line.strip() for line in ref_file)

# Check each subgroup file
for subgroup_file in subgroup_files:
    with open(subgroup_file, 'r') as sub_file:
        subgroup_lines = set(line.strip() for line in sub_file)

    # Find missing lines
    missing_lines = subgroup_lines - reference_lines

    # Print results for the current subgroup file
    if missing_lines:
        print(f"Missing lines in {subgroup_file}:")
        for line in missing_lines:
            print(line)
    else:
        print(f"All lines in {subgroup_file} are present in {reference_file}.")

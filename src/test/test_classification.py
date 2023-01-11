# Initialize an empty set to store the rules that match the new object
S = set()

# Iterate over the sorted set of rules in ARC-PAN
for r in ARC_PAN:
    # Check if the rule is a subset of the new object
    if r.issubset(o):
        count += 1
        S.add(r)
    if count == 1:
        fr_conf = r.confidence
        S.add(r)
    # Check if the confidence of the rule is greater than the first rule's confidence minus the specified confidence margin
    elif r.confidence > fr_conf - confidence_margin:
        S.add(r)
    else:
        break

# Divide the set S into subsets based on category
categories = dict()
for r in S:
    if r.category not in categories:
        categories[r.category] = []
    categories[r.category].append(r)

# Calculate the average confidence score for each category
scores = dict()
for cat in categories:
    score = sum(r.confidence for r in categories[cat]) / len(categories[cat])
    scores[cat] = score

# Assign the new object to the class with the highest confidence score
best_score = max(scores.values())
best_category = [cat for cat, score in scores.items() if score == best_score][0]
o.category = best_category
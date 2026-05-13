# Naive Bayes implementation for Pass/Fail prediction

data = [
    ["Low", "Poor", "Bad", "Yes", "Fail"],
    ["High", "Good", "Good", "No", "Pass"],
    ["Medium", "Good", "Good", "Yes", "Pass"],
    ["Low", "Poor", "Bad", "No", "Fail"],
    ["High", "Good", "Bad", "No", "Pass"],
    ["Medium", "Poor", "Good", "Yes", "Fail"],
    ["High", "Good", "Good", "Yes", "Pass"],
    ["Low", "Good", "Bad", "Yes", "Fail"],
    ["Medium", "Good", "Good", "No", "Pass"],
    ["High", "Poor", "Good", "No", "Pass"]
]

features = ["Study Hours", "Attendance", "Sleep Quality", "Part-time Job"]

test_data = ["Medium", "Good", "Good", "No"]

classes = ["Pass", "Fail"]

def prior_probability(data, cls):
    count = 0
    for row in data:
        if row[-1] == cls:
            count += 1
    return count / len(data)

def conditional_probability(data, feature_index, feature_value, cls):
    class_count = 0
    feature_count = 0

    for row in data:
        if row[-1] == cls:
            class_count += 1
            if row[feature_index] == feature_value:
                feature_count += 1

    return feature_count / class_count

def naive_bayes(data, test_data, classes):
    probabilities = {}

    for cls in classes:
        probability = prior_probability(data, cls)

        for i in range(len(test_data)):
            probability *= conditional_probability(data, i, test_data[i], cls)

        probabilities[cls] = probability

    return probabilities

probabilities = naive_bayes(data, test_data, classes)

print("Probabilities:")
for cls, prob in probabilities.items():
    print(cls, ":", prob)

prediction = max(probabilities, key=probabilities.get)

print("\nPredicted Class:", prediction)

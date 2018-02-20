import sys
import numpy as np

if len(sys.argv) != 3:
    print('usage: %s data predictions' % sys.argv[0])
    sys.exit()

data_file = sys.argv[1]
predictions_file = sys.argv[2]

data = open(data_file)
predictions = open(predictions_file)

# Load the real labels.
true_labels = []
for line in data:
    if len(line.strip()) == 0:
        continue
    true_labels.append(line.split()[0])

predicted_labels = []
for line in predictions:
    predicted_labels.append(line.strip())

data.close()
predictions.close()

if len(predicted_labels) != len(true_labels):
    print('Number of lines in two files do not match.')
    sys.exit()

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)
correct_mask = true_labels == predicted_labels
num_correct = float(correct_mask.sum())
total = correct_mask.size
accuracy = num_correct / total

print('Accuracy: %f (%d/%d)' % (accuracy, num_correct, total))
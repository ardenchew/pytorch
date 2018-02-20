import numpy as np
from scipy.sparse import csr_matrix


def load_data(filename):
    """ Load data.

    Args:
        filename: A string. The path to the data file.

    Returns:
        A tuple, (X, y). X is a compressed sparse row matrix of floats with
        shape [num_examples, num_features]. y is a dense array of ints with
        shape [num_examples].
    """

    X_nonzero_rows, X_nonzero_cols, X_nonzero_values = [], [], []
    y = []
    with open(filename) as reader:
        for example_index, line in enumerate(reader):
            if len(line.strip()) == 0:
                continue

            # Divide the line into features and label.
            split_line = line.split(" ")
            label_string = split_line[0]

            int_label = -1
            try:
                int_label = int(label_string)
            except ValueError:
                raise ValueError("Unable to convert " + label_string + " to integer.")
            y.append(int_label)

            for item in split_line[1:]:
                try:
                    # Features are 1 indexed in the data files, so we need to subtract 1.
                    feature_index = int(item.split(":")[0]) - 1
                except ValueError:
                    raise ValueError("Unable to convert index " + item.split(":")[0] + " to integer.")
                if feature_index < 0:
                    raise Exception('Expected feature indices to be 1 indexed, but found index of 0.')
                try:
                    value = float(item.split(":")[1])
                except ValueError:
                    raise ValueError("Unable to convert value " + item.split(":")[1] + " to float.")

                if value != 0.0:
                    X_nonzero_rows.append(example_index)
                    X_nonzero_cols.append(feature_index)
                    X_nonzero_values.append(value)

    X = csr_matrix((X_nonzero_values, (X_nonzero_rows, X_nonzero_cols)), dtype=np.float)
    y = np.array(y, dtype=np.int)

    return X, y
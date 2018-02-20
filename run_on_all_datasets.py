import os


ALGORITHM = 'useless'
DATA_DIR = 'datasets'
OUTPUT_DIR = 'output'
DATASETS = ['easy', 'hard', 'bio', 'finance', 'nlp', 'speech', 'vision']


if not os.path.exists(DATA_DIR):
    raise Exception('Data directory specified by DATA_DIR does not exist.')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for dataset in DATASETS:

    print('Training algorithm %s on dataset %s...' % (ALGORITHM, dataset))
    data = os.path.join(DATA_DIR, '%s.train' % (dataset))
    model_file = os.path.join(OUTPUT_DIR, '%s.train.%s.pkl' % (dataset, ALGORITHM))
    unformatted_cmd = 'python3 classify.py --data %s --mode train --model-file %s --algorithm %s'
    cmd = unformatted_cmd % (data, model_file, ALGORITHM)
    os.system(cmd)

    for subset in ['train', 'dev', 'test']:
        data = os.path.join(DATA_DIR, '%s.%s' % (dataset, subset))
        # Some datasets might not contain full train, dev, test splits.
        # In this case we should continue without error.
        if not os.path.exists(data):
            continue
        print('Generating %s predictions on dataset %s (%s)...' % (ALGORITHM, dataset, subset))
        model_file = os.path.join(OUTPUT_DIR, '%s.train.%s.pkl' % (dataset, ALGORITHM))
        predictions_file = os.path.join(OUTPUT_DIR, '%s.%s.%s.predictions' % (dataset, subset, ALGORITHM))
        unformatted_cmd = 'python3 classify.py --data %s --mode test --model-file %s --predictions-file %s'
        cmd = unformatted_cmd % (data, model_file, predictions_file)
        os.system(cmd)
        if subset != 'test':
            print('Computing accuracy obtained by %s on dataset %s (%s)...' % (ALGORITHM, dataset, subset))
            cmd = 'python3 compute_accuracy.py %s %s' % (data, predictions_file)
            os.system(cmd)
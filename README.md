# Final Project

## How to reproduce the best result?

Download the training/test data from the [website](https://tbrain.trendmicro.com.tw/Competitions/Details/26).
Copy the test dataset to the `data` directory.

```bash
mkdir data
unzip $TEST_DATASET.zip
mv $TEST_DATASET.csv data/test.csv
```

Run with:

```bash
python main.py --config config/bart.yml --test
```

## How to run other algorithms?
### Cosine Similarity
```bash
python main.py --config config/baseline.yml --test
```
### Classification (NLI)
```bash
python main.py --config config/nli.yml --test
```

### Classification (Binary)
```bash
python main.py --config config/binary_classifier.yml --test
```

### GPL
```bash
python main.py --config config/gpl.yml --test
```

## Team members
Coming out the ideas and solutions with @Jacky-15.

Jie-Jyun Liu, Kuang-Heng Ching

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

## Team members
Coming out the ideas and solutions with @Jacky-15.
Jie-Jyun Liu, Kuang-Heng Ching

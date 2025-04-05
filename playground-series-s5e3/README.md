# Binary Prediction with a Rainfall Dataset | Kaggle

Playground Series - Season 5, Episode 3

- https://www.kaggle.com/competitions/playground-series-s5e3/

## Prepare data

```console
$ . ~/.venv/kaggle/bin/activate
$ kaggle competitions download -c playground-series-s5e3
$ unzip playground-series-s5e3.zip -d data
Archive:  playground-series-s5e3.zip
  inflating: data/sample_submission.csv
  inflating: data/test.csv
  inflating: data/train.csv
```

## Train and predict

```console
# cross validation
$ python main.py --validate

# train and predict
$ python main.py

# check submissions
$ kaggle c submissions playground-series-s5e3

# submit
$ kaggle c submit playground-series-s5e3 -f submission.csv -m "Submission message"
```

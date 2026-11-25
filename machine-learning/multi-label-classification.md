# Multi-Label Classification

Multi-label classification assigns **zero, one, or several labels to each example**. A news article might be tagged with both technology and business; a photo may contain a dog, a person, and a bicycle. This differs from multiclass classification, where exactly one class is selected from a mutually exclusive set.

## Representing targets

For K possible labels, represent an example with a binary vector:

    y = [1, 0, 1, 0, 0]

A value of 1 means the label is present. Labels are not required to sum to one, so a softmax output is usually inappropriate. A common neural-network setup uses K sigmoid outputs and binary cross-entropy per label.

## Modeling strategies

- **Binary relevance:** Train one binary classifier per label. It is simple and parallelizable, but each model ignores label dependencies.
- **Classifier chains:** Feed earlier label predictions or targets into later classifiers. This can capture dependencies, but order matters and training/inference must be handled consistently.
- **Label powerset:** Treat each observed label combination as a single class. It models combinations directly, but the number of possible combinations can grow rapidly and unseen combinations are difficult.
- **Native multilabel models:** Some tree ensembles and neural architectures predict several labels jointly.

## Scikit-learn baseline

~~~python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Y_train and Y_test are binary indicator matrices with shape
# (number_of_examples, number_of_labels).
model = OneVsRestClassifier(
    LogisticRegression(max_iter=1000, class_weight="balanced")
)
model.fit(X_train, Y_train)

scores = model.predict_proba(X_test)
threshold = 0.35  # tune on validation data, not on the test set
predictions = (scores >= threshold).astype(int)

print(classification_report(Y_test, predictions, zero_division=0))
~~~

The example assumes each base estimator exposes probabilities. For estimators that do not, use decision scores and define a suitable thresholding procedure.

## Evaluation: choose metrics that match the task

- **Micro precision/recall/F1:** Aggregate all label decisions first; frequent labels have more influence.
- **Macro F1:** Compute F1 per label and average equally; useful for exposing weak rare labels.
- **Hamming loss:** Fraction of individual label decisions that are wrong.
- **Subset accuracy:** Fraction of examples for which the entire predicted label set is exactly correct. It is strict.
- **Label ranking average precision (LRAP):** Rewards ranking true labels above false labels.
- **Precision@k or Recall@k:** Useful when a system displays a fixed-size shortlist.

Report per-label support and metrics as well as an aggregate. A high micro score can hide poor performance on rare labels.

## Thresholds and imbalance

A default threshold of 0.5 is not automatically optimal. Tune thresholds on a validation set according to the cost of false positives and false negatives. Per-label thresholds can help when label prevalence and error costs differ, but require enough validation examples. Class weighting or resampling may help; verify that probabilities remain meaningful if they are used downstream.

## Common pitfalls

1. Applying softmax when labels can co-occur.
2. Randomly splitting correlated documents or users across train and test.
3. Selecting thresholds using test labels.
4. Reporting only subset accuracy, which can be harsh when there are many labels.
5. Ignoring label imbalance and annotation ambiguity.

## Practice

Use a dataset with at least five labels. Compare binary relevance with a classifier chain, tune a global threshold on validation data, and report micro-F1, macro-F1, Hamming loss, and per-label recall.

## References

- [Scikit-learn: Multiclass and multilabel classification](https://scikit-learn.org/stable/modules/multiclass.html)
- [Scikit-learn: Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)

**Key takeaway:** Multilabel systems predict independent label decisions or label rankings—not a single mutually exclusive class.
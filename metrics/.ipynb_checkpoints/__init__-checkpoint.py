from .functional import (
    get_stats,
    fbeta_score,
    f1_score,
    iou_score,
    accuracy,
    precision,
    recall,
    sensitivity,
    specificity,
    balanced_accuracy,
    positive_predictive_value,
    negative_predictive_value,
    false_negative_rate,
    false_positive_rate,
    false_discovery_rate,
    false_omission_rate,
    positive_likelihood_ratio,
    negative_likelihood_ratio,
)

from .metrics import Evaluator



def get_metrics_function(labels_clf, output_clf,num_class=3):
    evaluator = Evaluator(num_class)
    evaluator.add_batch(labels_clf, output_clf)
    miou = evaluator.Mean_Intersection_over_Union()
    return miou
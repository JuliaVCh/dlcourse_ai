def multiclass_accuracy(prediction, ground_truth):
    return ((prediction == ground_truth) * 1).sum() / prediction.size

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    # Write code here
    if len(y_true) == 0:
        return 0.0
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if (yt == yp):
            tp += 1

    total = len(y_true)

    precision = tp/total
    recall = tp/total

    if precision + recall == 0:
        return 0.0

    return 2*(precision*recall)/(precision + recall)
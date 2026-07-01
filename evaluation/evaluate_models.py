import pandas as pd

from evaluation.metrics import (
    pehe,
    bias,
    rmse,
    mean_cate_true,
    mean_cate_pred
)


def evaluate(
    cate_true,
    cate_pred,
    learner,
    dataset
):

    results = {

        "dataset": dataset,

        "learner": learner,

        "PEHE": pehe(
            cate_true,
            cate_pred
        ),

        "BIAS": bias(
            cate_true,
            cate_pred
        ),

        "RMSE": rmse(
            cate_true,
            cate_pred
        ),

        "MEAN_CATE_TRUE": mean_cate_true(
            cate_true
        ),

        "MEAN_CATE_PRED": mean_cate_pred(
            cate_pred
        )
    }

    return pd.DataFrame([results])
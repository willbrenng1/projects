from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from mlxtend.feature_selection import SequentialFeatureSelector
import pandas as pd
import numpy as np
from scipy import stats
import warnings
from collections import defaultdict


def split_scale_convert_data(df, num_of_splits, test_percentage, cols_to_scale):

    scaler = preprocessing.StandardScaler()
    df = df.drop("id", axis=1)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)
    training_data, temp_testing_data = train_test_split(
        df, test_size=test_percentage, random_state=42
    )
    training_data_x = training_data.drop("price", axis=1)
    training_data_y = training_data["price"]
    training_data_x[cols_to_scale] = scaler.fit_transform(
        training_data_x[cols_to_scale]
    ).astype(float)
    if num_of_splits == 2:
        testing_data_x = temp_testing_data.drop("price", axis=1)
        testing_data_y = temp_testing_data["price"]
        testing_data_x[cols_to_scale] = scaler.transform(
            testing_data_x[cols_to_scale]
        ).astype(float)
        return training_data_x, training_data_y, testing_data_x, testing_data_y
    if num_of_splits == 3:
        testing_data, validation_data = train_test_split(
            temp_testing_data, test_size=0.5, random_state=42
        )
        testing_data_x = testing_data.drop("price", axis=1)
        validation_data_x = validation_data.drop("price", axis=1)
        testing_data_y = testing_data["price"]
        validation_data_y = validation_data["price"]

        testing_data_x[cols_to_scale] = scaler.transform(
            testing_data_x[cols_to_scale]
        ).astype(float)

        validation_data_x[cols_to_scale] = scaler.transform(
            validation_data_x[cols_to_scale]
        ).astype(float)

        return (
            training_data_x,
            training_data_y,
            validation_data_x,
            validation_data_y,
            testing_data_x,
            testing_data_y,
        )
    else:
        None


def compute_inference_and_goodness_of_fit_statistics(model, x, y, confidence_interval):

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    def compute_model_matrix(x):
        return np.column_stack((np.ones(len(x)), x.values))

    def compute_model_betas(model):
        return np.insert(model.coef_, 0, model.intercept_)

    def get_model_predictions(model):
        return model.predict(x)

    def compute_adj_r_sq(preds, x, y):
        r_2 = r2_score(y, preds)
        n, p = x.shape[0], x.shape[1]
        adjusted_r2 = 1 - ((1 - r_2) * (n - 1)) / (n - p - 1)
        return adjusted_r2

    def compute_model_residual(preds, y):
        return preds - y

    def compute_model_df(x):
        n, p = x.shape[0], x.shape[1]
        return n - p - 1

    def compute_sigma_sq_hat(residual, df):
        rss = np.sum(residual**2)
        return rss / df

    def compute_var_cov_matrix(X_design, sigma_sq_hat):
        XtX_inv = np.linalg.inv(X_design.T @ X_design)
        return sigma_sq_hat * XtX_inv

    def compute_model_standard_error(var_cov_matrix):
        return np.sqrt(np.diag(var_cov_matrix))

    def compute_model_t_stat(betas, standard_error):
        return betas / standard_error

    def compute_model_p_val(t_stat, df):
        return (2 * (1 - stats.t.cdf(np.abs(t_stat), df))).round(5)

    def compute_model_ci(ci, df, betas, standard_error):
        pd.options.display.float_format = "{:.4e}".format
        two_tail_ci = 1 - (1 - ci) / 2
        t_crit = stats.t.ppf(two_tail_ci, df)

        ci_upper = betas + t_crit * standard_error
        ci_lower = betas - t_crit * standard_error

        return ci_upper, ci_lower

    def compute_aic_bic(y, x, predictions):
        n = len(y)
        k = x.shape[1] + 1
        rss = np.sum((y - predictions) ** 2)

        aic = n * np.log(rss / n) + 2 * k
        bic = n * np.log(rss / n) + k * np.log(n)

        return aic, bic

    x_d = compute_model_matrix(x)
    betas = compute_model_betas(model)
    predictions = get_model_predictions(model)
    adj_r2 = compute_adj_r_sq(predictions, x, y)
    residuals = compute_model_residual(predictions, y)
    degree_freedom = compute_model_df(x)
    ssh = compute_sigma_sq_hat(residuals, degree_freedom)
    var_covar_matrix = compute_var_cov_matrix(x_d, ssh)
    standard_error = compute_model_standard_error(var_covar_matrix)
    t_stat = compute_model_t_stat(betas, standard_error)
    p_vals = compute_model_p_val(t_stat, degree_freedom)
    upper_ci, lower_ci = compute_model_ci(
        confidence_interval, degree_freedom, betas, standard_error
    )
    aic, bic = compute_aic_bic(y, x, predictions)
    mse = mean_squared_error(
        y,
        predictions,
    )
    rmse = root_mean_squared_error(
        y,
        predictions,
    )
    mae = mean_absolute_error(
        y,
        predictions,
    )

    return {
        "inference_stats": pd.DataFrame(
            {
                "coef": betas,
                "standard_error": standard_error,
                "t_stat": t_stat,
                "p_vals": p_vals,
                "ci_lower": lower_ci,
                "ci_upper": upper_ci,
            },
            index=["intercept"] + list(x.columns),
        ),
        "precision_stats": {
            "adj_r2": adj_r2,
            "aic": aic,
            "bic": bic,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
        },
    }


def adj_r2(estimator, x, y):

    n = len(y)
    p = x.shape[1]
    pred = estimator.predict(x)
    r2 = r2_score(y, pred)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def produce_forward_selection_lms(x, y, max_features, cv_folds, ci_level):

    def adj_r2(estimator, x, y):
        n = len(y)
        p = x.shape[1]
        pred = estimator.predict(x)
        r2 = r2_score(y, pred)
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    mod_eval_stats_dict = defaultdict(dict)
    for i in range(1, max_features + 1):

        forward_selection_model = SequentialFeatureSelector(
            LinearRegression(), k_features=i, forward=True, scoring=adj_r2, cv=cv_folds
        )

        forward_selected_features = forward_selection_model.fit(x, y)
        forward_selected_indices = forward_selected_features.k_feature_idx_
        selected_cols = x.columns[list(forward_selected_indices)]

        x_select = x[selected_cols]
        mod = LinearRegression(fit_intercept=True)
        mod.fit(x_select, y)

        model_metrics = compute_inference_and_goodness_of_fit_statistics(
            mod, x_select, y, ci_level
        )

        mod_eval_stats_dict[f"{i} - forward"]["model"] = mod
        mod_eval_stats_dict[f"{i} - forward"]["model_metrics"] = model_metrics

    return mod_eval_stats_dict


def produce_ridge_regression_model(x, y, cv_folds, alpha_list, ci_level, result_dict):

    ridge_model = RidgeCV(alphas=alpha_list, cv=cv_folds)
    ridge_model.fit(x, y)
    model_metrics = compute_inference_and_goodness_of_fit_statistics(
        ridge_model, x, y, ci_level
    )

    model_index = len(result_dict.keys()) + 1

    result_dict[f"{model_index} - ridgeCV"]["model"] = ridge_model
    result_dict[f"{model_index} - ridgeCV"]["model_metrics"] = model_metrics
    result_dict[f"{model_index} - ridgeCV"]["best_alpha"] = ridge_model.alpha_


def produce_lasso_regression_model(x, y, cv_folds, alpha_list, ci_level, result_dict):

    lasso_model = LassoCV(alphas=alpha_list, cv=cv_folds)
    lasso_model.fit(x, y)
    model_metrics = compute_inference_and_goodness_of_fit_statistics(
        lasso_model, x, y, ci_level
    )

    model_index = len(result_dict.keys()) + 1

    result_dict[f"{model_index} - lassoCV"]["model"] = lasso_model
    result_dict[f"{model_index} - lassoCV"]["model_metrics"] = model_metrics
    result_dict[f"{model_index} - lassoCV"]["best_alpha"] = lasso_model.alpha_
    # result_dict[model_index]["selection_method"] = "lassoCV"


# def return_best_models(df, dicts):
#     model_index_list = [int(model.split(" - ")[0]) for model in df["model"].to_list()]

#     return {i: dicts[i] for i in model_index_list}


def return_best_models(df, dicts):
    model_index_list = df["model"].to_list()

    return {mod: dicts[mod]["model"] for mod in model_index_list}


def best_model_validation(dicts, x_val, y_val):

    preds_dict = {}
    res_dict = defaultdict(dict)

    def compute_adj_r_sq(preds, x, y):
        r_2 = r2_score(y, preds)
        n, p = x.shape[0], x.shape[1]
        adjusted_r2 = 1 - ((1 - r_2) * (n - 1)) / (n - p - 1)
        return adjusted_r2

    def compute_aic_bic(y, x, predictions):
        n = len(y)
        k = x.shape[1] + 1
        rss = np.sum((y - predictions) ** 2)

        aic = n * np.log(rss / n) + 2 * k
        bic = n * np.log(rss / n) + k * np.log(n)

        return aic, bic

    for k, v in dicts.items():

        model_features = v.feature_names_in_
        x = x_val[model_features]
        y_pred = v.predict(x)
        preds_dict[k] = y_pred
        res_dict[k]["mse"] = mean_squared_error(
            y_val,
            y_pred,
        )
        res_dict[k]["adj_r2"] = compute_adj_r_sq(y_pred, x, y_val)
        res_dict[k]["rmse"] = root_mean_squared_error(y_val, y_pred)
        res_dict[k]["mae"] = mean_absolute_error(y_val, y_pred)
        res_dict[k]["aic"], res_dict[k]["bic"] = compute_aic_bic(y_val, x, y_pred)
    preds_dict["y_true"] = np.array(y_val)
    return preds_dict, res_dict


def final_model_validation(best_model, dicts, x_test, y_test):

    preds_dict = {}
    res_dict = defaultdict(dict)

    def compute_adj_r_sq(preds, x, y):
        r_2 = r2_score(y, preds)
        n, p = x.shape[0], x.shape[1]
        adjusted_r2 = 1 - ((1 - r_2) * (n - 1)) / (n - p - 1)
        return adjusted_r2

    def compute_aic_bic(y, x, predictions):
        n = len(y)
        k = x.shape[1] + 1
        rss = np.sum((y - predictions) ** 2)

        aic = n * np.log(rss / n) + 2 * k
        bic = n * np.log(rss / n) + k * np.log(n)

        return aic, bic

    final_model_dict = {best_model: dicts[best_model]}
    for k, v in final_model_dict.items():

        model_features = v.feature_names_in_
        x = x_test[model_features]
        y_pred = v.predict(x)
        preds_dict[k] = y_pred
        preds_dict["residual"] = y_test - y_pred
        res_dict[k]["mse"] = mean_squared_error(
            y_test,
            y_pred,
        )
        res_dict[k]["adj_r2"] = compute_adj_r_sq(y_pred, x, y_test)
        res_dict[k]["rmse"] = root_mean_squared_error(y_test, y_pred)
        res_dict[k]["mae"] = mean_absolute_error(y_test, y_pred)
        res_dict[k]["aic"], res_dict[k]["bic"] = compute_aic_bic(y_test, x, y_pred)

    preds_dict["y_true"] = np.array(y_test)
    return preds_dict, res_dict


def get_important_features(best_model, best_models_dict, top_n):
    model = best_models_dict[best_model]

    coefficients = model.coef_

    feature_names = best_models_dict[best_model].feature_names_in_

    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})

    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values(by="abs_coefficient", ascending=False)

    top_features = coef_df[["feature", "coefficient"]].head(top_n)
    return top_features

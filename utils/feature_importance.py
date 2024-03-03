import numpy as np
import pandas as pd


def get_shap_global(explainer, data, return_local=False):
    shap_values = explainer(data).values
    importance_mean = np.mean(np.abs(shap_values), axis=0)
    importance_std = np.std(np.abs(shap_values), axis=0, ddof=1) / np.sqrt(data.shape[0])
    if return_local:
        return importance_mean, importance_std, shap_values
    return importance_mean, importance_std


def permute_all_features_obs(model, X, y, num_perms, is_reg, seed, return_perms=False):
    y_pred = model.predict(X) if is_reg else model.predict_proba(X)[:, 1]
    orig_loss = np.power(y - y_pred, 2)
    def permute_feature(model, X, y, feature, num_perms, is_reg, seed, return_perms):
        perms_losses = []
        for i in range(num_perms):
            X_perm = X.copy()
            X_perm[feature] = X[feature].sample(frac=1, random_state=seed + i).values
            y_pred_perm = model.predict(X_perm) if is_reg else model.predict_proba(X_perm)[:, 1]
            perm_loss = np.power(y - y_pred_perm, 2)  # difference for each row, for a single permutation of 1 feature
            perms_losses.append(perm_loss - orig_loss)
        perms_losses = np.array(perms_losses)  # each column is a single permutation of the feature
        if return_perms:
            return perms_losses
        return np.mean(perms_losses, axis=1)  # mean over num_perms permutations
    return {feature : permute_feature(model, X, y, feature, num_perms, is_reg, seed, return_perms) for feature in X.columns}


def get_pfi_obs(model, X, y, num_perms, is_reg, seed, return_local=False):
    importance_res = permute_all_features_obs(model, X, y, num_perms, is_reg, seed)
    importance_res = pd.DataFrame(importance_res).values
    importance_mean = np.mean(importance_res, axis=0)
    importance_std = np.std(importance_res, axis=0, ddof=1) / np.sqrt(X.shape[0])
    if return_local:
        return importance_mean, importance_std, importance_res
    return importance_mean, importance_std

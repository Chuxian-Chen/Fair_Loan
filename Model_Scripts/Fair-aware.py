import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from interpret.glassbox import ExplainableBoostingClassifier
from pygam import LogisticGAM, f as gam_f, l as gam_l
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tabpfn import TabPFNClassifier
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def clean_data(df):
    df = df[df['person_age'] < 100].copy()
    df = df[df['person_emp_exp'] < 50].copy()

    # Income extreme values are truncated
    # (limited to the 99.9th percentile to prevent outliers from affecting the model).
    upper_income = df['person_income'].quantile(0.999)
    df['person_income'] = df['person_income'].clip(upper=upper_income)
    return df

def train_and_evaluate(model_name, model_instance, X, y, S, seed, mode='detailed', is_pre_trained=False,
                       saved_bounds=None):
    """
    Core Functions: nested network search tuning (Nested CV)
    and post-processing fairness approach.
    """
    # Outer loop: Divide the training set and test set (80% / 20%) based on the current random seed.
    X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(
        X, y, S, test_size=0.2, random_state=seed, stratify=y
    )

    # Local standardization: solving data leakage
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    current_model_bounds = None
    best_estimator = model_instance

    if not is_pre_trained:


        # [Tuning Module 1] Context truncation for complex large-scale TabPFN models
        # Since TabPFN is a zero-shot inference model based on In-Context Learning,-
        # -it does not require traditional gradient parameter tuning.
        # The limit of test_size=600 is to prevent memory overflow (OOM)-
        # -of the underlying Transformer attention matrix.
        if model_name == "TabPFN":
            _, X_train_sub, _, y_train_sub = train_test_split(
                X_train, y_train, test_size=min(600, len(X_train)), random_state=seed, stratify=y_train
            )
            model_instance.fit(X_train_sub, y_train_sub)
            best_estimator = model_instance

        # [Tuning Module 2] Downsampling and Mesh Constraints for Glass Box Model EBM
        # EBM is extremely time-consuming to compute pairwise interactions.
        # By sampling 2000 data points, the computational bottleneck can be overcome.
        # `max_bins=128` limits the granularity of feature discretization,-
        # -preventing overfitting to sensitive attributes.
        elif model_name == "EBM":
            _, X_train_sub, _, y_train_sub = train_test_split(
                X_train, y_train, test_size=min(2000, len(X_train)), random_state=seed, stratify=y_train
            )
            param_grid_ebm = {'max_bins': [128]}  # Streamline search space

            # n_jobs=1 forces serial processing to prevent Windows multi-process deadlock crashes.
            grid_search = GridSearchCV(estimator=model_instance, param_grid=param_grid_ebm, cv=3, scoring='f1',
                                       n_jobs=1)
            grid_search.fit(X_train_sub, y_train_sub)
            best_estimator = grid_search.best_estimator_
            if mode == 'detailed':
                print(f"      [Tuning] {model_name} (Seed {seed}) Best Params: {grid_search.best_params_}")

        # [Optimization Module 3] Customized Cyclic Mesh Optimization and Deadlock Prevention for GAM
        # GAM is incompatible with sklearn's GridSearchCV.
        # It has a smaller spline smoothing penalty (laminar).
        # This can lead to the generation of singular matrices in locally sparse data, causing the IRLS algorithm to fail to converge.
        elif model_name == "GAM":
            # Layered Downsampling of 1000 Core Data Points
            sample_size = min(1000, len(X_train))
            _, X_train_sub, _, y_train_sub = train_test_split(
                X_train, y_train, test_size=sample_size, random_state=seed, stratify=y_train
            )

            # [Parameter Tuning] Remove lam=0.1, which is prone to non-convergence,
            # and lock in robust high penalty values [10.0, 100.0].
            # The model is forced to learn macroscopic smoothing trends,-
            # -creating conditions for subsequent continuous probability shifts.
            lam_candidates = [10.0, 100.0]
            best_gam_model = None
            best_gam_f1 = -1

            # Handwritten grid optimization loop
            for lam_val in lam_candidates:
                try:
                    candidate_gam = LogisticGAM(
                        gam_f(0, lam=lam_val) + gam_f(1, lam=lam_val) + gam_f(2, lam=lam_val) +
                        gam_f(3, lam=lam_val) + gam_f(4, lam=lam_val) + gam_f(5, lam=lam_val) +
                        gam_f(6, lam=lam_val) + gam_f(7, lam=lam_val) + gam_l(8, lam=lam_val) +
                        gam_l(9, lam=lam_val) + gam_l(10, lam=lam_val) + gam_l(11, lam=lam_val)
                    )
                    # [Convergence Protection] Set the maximum number of iterations to 50-
                    # -to prevent the algorithm from getting stuck in infinite iteration.
                    candidate_gam.max_iter = 50
                    candidate_gam.fit(X_train_sub, y_train_sub)

                    # [Convergence Protection] Checks whether the underlying mathematical matrix has truly converged;
                    # if it has not converged, it is discarded decisively.
                    if not candidate_gam.results_['converged']:
                        continue

                    val_pred = candidate_gam.predict(X_train_sub)
                    score = f1_score(y_train_sub, val_pred, zero_division=0)
                    if score > best_gam_f1:
                        best_gam_f1 = score
                        best_gam_model = candidate_gam
                except Exception as e:
                    # Capture underlying mathematical crashes caused by matrix-
                    # -singularities to ensure program continued operation.
                    continue

            # Safety strategy: If convergence does not occur even in extreme cases,-
            # -force the use of the strongest smooth fit with lam=100.
            if best_gam_model is None:
                best_gam_model = LogisticGAM(
                    gam_f(0, lam=100.0) + gam_f(1, lam=100.0) + gam_f(2, lam=100.0) +
                    gam_f(3, lam=100.0) + gam_f(4, lam=100.0) + gam_f(5, lam=100.0) +
                    gam_f(6, lam=100.0) + gam_f(7, lam=100.0) + gam_l(8, lam=100.0) +
                    gam_l(9, lam=100.0) + gam_l(10, lam=100.0) + gam_l(11, lam=100.0)
                ).fit(X_train_sub, y_train_sub)

            best_estimator = best_gam_model
            # Record the feature boundaries of the GAM for use in test set cropping.
            current_model_bounds = (X_train_sub.min().to_numpy(), X_train_sub.max().to_numpy())

        # =====================================================================
        # [Tuning Module 4] Basic Machine Learning Models (Tree Models and Linear Models)
        # Perform 3-fold cross-validation using GridSearchCV. Limit the tree's max_depth and-
        # min_samples_leaf to prevent the model output from polarizing to 0 or 1,-
        # -calibrate the probability distribution.
        # =====================================================================
        else:
            # Basic model hyperparameter search space (Grid Space)
            param_grids = {
                # C. Penalty strength:-
                # -To prevent collinearity features from having an excessive impact.
                "Logistic Regression": {'C': [0.1, 1.0], 'penalty': ['l2']},
                # min_samples_leaf=30/50：Each decision is based on at least several dozen samples to prevent probability hardening.
                "Decision Tree": {'max_depth': [5, 10, None], 'min_samples_leaf': [30, 50]},
                # Smoothed variance across 100 trees helps avoid rigid decision-making from a single tree.
                "Random Forest": {'n_estimators': [50, 100], 'max_depth': [10, None]},
                # Suppress the learning rate and tree depth to-
                # -prevent extreme gradient boosting from overfitting historical biases.
                "XGBoost": {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
            }

            if hasattr(model_instance, 'random_state'):
                model_instance.set_params(random_state=seed)

            # Inner nested validation: 3-Fold CV
            grid_search = GridSearchCV(
                estimator=model_instance,
                param_grid=param_grids[model_name],
                cv=3,    # 3-fold cross-validation
                scoring='f1',
                n_jobs=1
            )
            # Fit the model to the training set of the current seed to find the optimal parameters.
            grid_search.fit(X_train, y_train)
            best_estimator = grid_search.best_estimator_
            if mode == 'detailed':
                print(f"      [Tuning] {model_name} (Seed {seed}) Best Params: {grid_search.best_params_}")

    else:
        # Load the saved model boundaries in pre-training mode
        # (for 10 outer layer evaluations).
        if model_name == "GAM":
            current_model_bounds = saved_bounds if saved_bounds is not None else (X.min().to_numpy(),
                                                                                  X.max().to_numpy())

    # =====================================================================
    # [Fairness improving] (Threshold-Shifting Strategy)
    # This is built upon the optimal model (best_estimator)-
    # -that has just been tuned to the "generalization limit".
    # =====================================================================

    # Output smoothing prediction probability
    if model_name == "GAM":
        X_test_arr = X_test.to_numpy()
        X_test_clipped = np.clip(X_test_arr, current_model_bounds[0], current_model_bounds[1])
        y_prob = best_estimator.predict_mu(X_test_clipped)
    else:
        y_prob = best_estimator.predict_proba(X_test)[:, 1]

    # Calculation baseline metric: A threshold of 0.5 is uniformly set.
    y_pred_base = (y_prob >= 0.5).astype(int)
    acc_base = accuracy_score(y_test, y_pred_base)
    abs_spd_base = abs(y_pred_base[S_test == 1].mean() - y_pred_base[S_test == 0].mean())

    # Finding the Pareto optimal fairness threshold solution
    candidates = []
    thresholds = np.linspace(0.4, 0.6, 11)  # Sliding in a continuous probability density space
    for th_doc in thresholds:
        for th_rest in thresholds:
            # Applying two-stage dynamic thresholds to different sensitive groups
            preds = np.array([1 if (p >= th_doc if s == 1 else p >= th_rest) else 0 for p, s in zip(y_prob, S_test)])
            curr_spd = abs(preds[S_test == 1].mean() - preds[S_test == 0].mean())
            acc = accuracy_score(y_test, preds)

            # Constraints: Bias must decrease while retaining more than 95% of the optimal baseline accuracy.
            if curr_spd < abs_spd_base and acc > (acc_base * 0.95):
                candidates.append({'spd': curr_spd, 'acc': acc, 'preds': preds, 'th_doc': th_doc, 'th_rest': th_rest})

    # Results Assembly and Return
    if candidates:
        candidates.sort(key=lambda x: x['acc'], reverse=True)  # Prioritize the solution with the highest accuracy.
        best = candidates[0]
        res = {
            'Model': model_name, 'Instance': best_estimator, 'Bounds': current_model_bounds,
            'Base_Acc': acc_base, 'Base_Prec': precision_score(y_test, y_pred_base, zero_division=0),
            'Base_Rec': recall_score(y_test, y_pred_base, zero_division=0),
            'Base_F1': f1_score(y_test, y_pred_base, zero_division=0),
            'Base_SPD': abs_spd_base,
            'Fair_Acc': best['acc'], 'Fair_Prec': precision_score(y_test, best['preds'], zero_division=0),
            'Fair_Rec': recall_score(y_test, best['preds'], zero_division=0),
            'Fair_F1': f1_score(y_test, best['preds'], zero_division=0),
            'Fair_SPD': best['spd'], 'Best_Th_Doc': best['th_doc'], 'Best_Th_Rest': best['th_rest']
        }
    else:
        # If no better solution is found, return to the baseline performance.
        res = {
            'Model': model_name, 'Instance': best_estimator, 'Bounds': current_model_bounds,
            'Base_Acc': acc_base, 'Base_Prec': precision_score(y_test, y_pred_base, zero_division=0),
            'Base_Rec': recall_score(y_test, y_pred_base, zero_division=0),
            'Base_F1': f1_score(y_test, y_pred_base, zero_division=0),
            'Base_SPD': abs_spd_base,
            'Fair_Acc': acc_base, 'Fair_Prec': precision_score(y_test, y_pred_base, zero_division=0),
            'Fair_Rec': recall_score(y_test, y_pred_base, zero_division=0),
            'Fair_F1': f1_score(y_test, y_pred_base, zero_division=0),
            'Fair_SPD': abs_spd_base, 'Best_Th_Doc': 0.5, 'Best_Th_Rest': 0.5
        }

    # Dynamic return: The first round of detailed mode returns the complete dictionary,-
    # -while subsequent outer loops only return the SPD to quickly construct the test matrix.
    return res if mode == 'detailed' else (abs_spd_base, res.get('Fair_SPD', abs_spd_base))


def calculate_paired_t_test_metrics(baseline_scores, fair_scores):
    """
    Calculate the paired-samples t-test to mathematically-
    -demonstrate the statistical significance of the improved fairness (P < 0.05).
    """
    x, y_vals = np.array(baseline_scores), np.array(fair_scores)
    n = len(x)
    d = x - y_vals
    d_bar, S_d = np.mean(d), np.std(d, ddof=1)
    se = S_d / np.sqrt(n)
    t0 = d_bar / se if se != 0 else 0
    p_val = stats.t.sf(np.abs(t0), n - 1) * 2
    return d_bar, S_d, t0, p_val



if __name__ == '__main__':
    print("\n1. Loading and Preprocessing Data...")
    file_name = '../DataSet/Loan Approval Insights Data to Decision-Making.csv'
    if not os.path.exists(file_name):
        print(f"Error: File not found: {file_name}")
        exit()

    df = pd.read_csv(file_name)

    df = clean_data(df)

    # Standard Category Variable Coding
    le_dict = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # Define the sensitive group (Doctorate = privileged class / control group)
    try:
        doc_id = le_dict['person_education'].transform(['Doctorate'])[0]
    except:
        doc_id = 1

    S = np.where(df['person_education'] == doc_id, 1, 0)
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']

    # Standardize the feature space to eliminate the influence of dimensions.
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    save_dir = 'New'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_package_path = os.path.join(save_dir, 'loan_model_package.joblib')

    is_pre_trained = False
    trained_instances = {}
    saved_gam_bounds = None

    if os.path.exists(model_package_path):
        print(f"\nFound saved model package. Start loading...")
        package = joblib.load(model_package_path)
        trained_instances = package['models']
        saved_gam_bounds = package.get('gam_bounds')
        models_to_run = trained_instances
        is_pre_trained = True
    else:
        print("\nNo model found. Start training and automated hyperparameter tuning...")
        # A heterogeneous algorithm pool covering 7 dimensions
        models_to_run = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "EBM": ExplainableBoostingClassifier(random_state=42),
            "GAM": "LogisticGAM",
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost": GradientBoostingClassifier(random_state=42),
            "TabPFN": TabPFNClassifier(device='cpu')
        }

    print("\nPreparing Model Performance Comparison Table (Seed 42 with Optimization)")
    detailed_results, final_thresholds = [], {}

    # The first complete test was conducted to obtain tuning parameters and persist the model.
    for name, clf in models_to_run.items():
        print(f"    Processing {name}...")
        instance = clf if not (name == "GAM" and isinstance(clf, str)) else None
        m = train_and_evaluate(name, instance, X_scaled, y, S, seed=42, mode='detailed',
                               is_pre_trained=is_pre_trained, saved_bounds=saved_gam_bounds)

        detailed_results.append({
            'Algorithm': name, 'Strategy': 'Baseline', 'Accuracy': m['Base_Acc'],
            'Precision': m['Base_Prec'], 'Recall': m['Base_Rec'], 'F1 Score': m['Base_F1'], 'SPD (Bias)': m['Base_SPD']
        })
        detailed_results.append({
            'Algorithm': name, 'Strategy': 'Fairness-Aware', 'Accuracy': m['Fair_Acc'],
            'Precision': m['Fair_Prec'], 'Recall': m['Fair_Rec'],
            'F1 Score': m['Fair_F1'], 'SPD (Bias)': m['Fair_SPD']
        })
        final_thresholds[name] = (m['Best_Th_Doc'], m['Best_Th_Rest'])
        if not is_pre_trained:
            trained_instances[name] = m['Instance']
            if name == "GAM":
                saved_gam_bounds = m['Bounds']

    df_4_6 = pd.DataFrame(detailed_results).round(4)
    print("\n[Model Performance Comparison]")
    print(df_4_6.to_string(index=False))

    if not is_pre_trained:
        print("\nSaving robust fine-tuned package...")
        joblib.dump({'models': trained_instances, 'scaler': scaler, 'le_dict': le_dict,
                     'thresholds': final_thresholds, 'gam_bounds': saved_gam_bounds, 'features': X.columns.tolist()},
                    model_package_path)

    # =====================================================================
    # [Excluding randomness] 10 outer loops with multiple random seeds
    # To eliminate the randomness and chance inherent in data segmentation
    # =====================================================================
    print("\nPreparing 10 SPD independent runs with Robust Parameter Re-tuning...")
    raw_rows, summary_rows = [], []
    for name, clf in trained_instances.items():
        print(f"Validating Robustness for {name} across 10 iterations...")
        b_spds, fair_scores_list = [], []

        for i in range(10):
            # Detailed mode logging is triggered only in the first loop (i=0)-
            # -to ensure a clean terminal.
            mode_str = 'detailed' if i == 0 else 'spd_only'

            # Force is_pre_trained=False to perform-
            # -3-fold hyperparameter tuning on its own slice dataset in each loop
            eval_result = train_and_evaluate(name, clf, X_scaled, y, S, seed=i, mode=mode_str,
                                             is_pre_trained=False, saved_bounds=saved_gam_bounds)

            if mode_str == 'detailed':
                b_spds.append(eval_result['Base_SPD'])
                fair_scores_list.append(eval_result['Fair_SPD'])
                if name == "GAM":
                    saved_gam_bounds = eval_result['Bounds']
            else:
                # Subsequent loops only extract SPD for rapid recording.
                b_val, fair_spd = eval_result
                b_spds.append(b_val)
                fair_scores_list.append(fair_spd)

        raw_rows.append([name] + b_spds)
        raw_rows.append([f"Fair-{name}"] + fair_scores_list)

        # Statistical validation
        d_bar, S_d, t_stat, p_val = calculate_paired_t_test_metrics(b_spds, fair_scores_list)
        summary_rows.append({
            'Model': name, 'Baseline Mean SPD': np.mean(b_spds), 'Fair Model Mean SPD': np.mean(fair_scores_list),
            'Bias Reduction': f"{((np.mean(b_spds) - np.mean(fair_scores_list)) / np.mean(b_spds)) * 100:.1f}%",
            'd_bar': d_bar, 'S_d': S_d, 't-stat': t_stat, 'P-Value': p_val, 'Sig?': 'Yes' if p_val < 0.05 else 'No'
        })

    # Output Tables and Storage
    df_4_7 = pd.DataFrame(raw_rows, columns=['Model'] + [f'Run {i + 1}' for i in range(10)]).round(4)
    df_4_8 = pd.DataFrame(summary_rows).round(4)
    print("\n[10 SPD independent runs]\n", df_4_7.to_string(index=False))
    print("\n[Paired T-test for Fairness Improvement]\n", df_4_8.to_string(index=False))

    df_4_6.to_csv(os.path.join(save_dir, 'Performance.csv'), index=False)
    df_4_7.to_csv(os.path.join(save_dir, 'SPD_Runs.csv'), index=False)
    df_4_8.to_csv(os.path.join(save_dir, 'Significance.csv'), index=False)
    print(f"\nEvaluation results have been saved as CSV files in '{save_dir}/'")
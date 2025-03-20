from eqm.cross_validation import filter_data_to_fold
from eqm.paths import *
from scripts.flipped_training import *
import dill
from scripts.reporting import *
from eqm.classifier_helper import get_classifier_from_coefficients
from eqm.debug import ipsh
from eqm.data import remove_variable
import matplotlib.pyplot as plt
import numpy as np

# extra for explainability example
import lime
import shap
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

#############################################
data_name = "compas_arrest_small"
# data_name = "compas_violent_small"
fold_id = "K05N01"

# settings just for disparity checks
epsilon = 0.02
epsilons = [0.005, 0.02]
# epsilons = np.linspace(0.001, 0.03, num=15).tolist()

#############################################


# given a results df row with a classifier, and a feature matrix, returns the vector of predictions
def df_predict_handle(row, X):
    return row['clf'].predict(x)[0]


def get_metrics(clf, X, Y, X_test, Y_test, metrics=None):
    train_preds = clf.predict(X)
    test_preds = clf.predict(X_test)

    train_error = np.mean(train_preds != Y)
    test_error = np.mean(test_preds != Y_test)
    train_TPR = np.mean((train_preds == Y)[Y == 1])
    test_TPR = np.mean((test_preds == Y_test)[Y_test == 1])
    train_FPR = np.mean((train_preds != Y)[Y == -1])
    test_FPR = np.mean((test_preds != Y_test)[Y_test == -1])

    results = {"train_error": train_error,
               "test_error": test_error}

    assert all(k in results for k in metrics)
    if metrics is not None:
        results = {k: results[k] for k in metrics}

    return results

def get_multiplicity(baseline_clf, level_set_clfs, X, feature_names):
    baseline_preds = baseline_clf.predict(X)
    num_instances = X.shape[0]

    ambiguous_set = set()
    num_max_discrepancy = 0

    shap_value_arr = []
    # following line works with epsilon set based models!
    for i, clf in enumerate(level_set_clfs):
        preds = clf.predict(X)
        conflicts = np.not_equal(preds, baseline_preds)
        conflict_indices = np.where(conflicts)[0]
        conf_ind_len = len(conflict_indices)
        # 1. python code to output X corresponding to the wrong Y
        # if (conf_ind_len>1):
            # print(i, conf_ind_len, conflict_indices[0], X[conflict_indices[0]], len(X[conflict_indices[0]]))
        
        # 2. then, need to identify what are the feature that are used as an input (22 is the length of the inp array)
        # ['(Intercept)', 'race_is_causasian', 'race_is_african_american', 'race_is_hispanic', 'race_is_other', 'age_leq_25', 
        #  'age_25_to_45', 'age_geq_46', 'female', 'n_priors_eq_0', 'n_priors_geq_1', 'n_priors_geq_2', 'n_priors_geq_5', 
        #  'n_juvenile_misdemeanors_eq_0', 'n_juvenile_misdemeanors_geq_1', 'n_juvenile_misdemeanors_geq_2', 
        #  'n_juvenile_misdemeanors_geq_5', 'n_juvenile_felonies_eq_0', 'n_juvenile_felonies_geq_1', 
        #  'n_juvenile_felonies_geq_2', 'n_juvenile_felonies_geq_5', 'charge_degree_eq_M']
        
        # 3. then, need to collect all these conflicting inputs, analyze with lime, then to combine the results with some clustering

        ambiguous_set.update(np.where(conflicts)[0])
        discrepancy = np.sum(conflicts)
        num_max_discrepancy = max(discrepancy, num_max_discrepancy)

        if (conf_ind_len>1):
            # explain_w_lime(clf, X, feature_names, conflict_indices)
            shap_value_arr.append(explain_w_shap(clf, X, feature_names, conflict_indices))
        
        if (len(shap_value_arr) == 1):
            break

    num_ambiguous = len(ambiguous_set)
    ambiguity = num_ambiguous / num_instances
    max_discrepancy = num_max_discrepancy / num_instances
    results = {"n": X.shape[0],
            "ambiguity": ambiguity,
            "num_ambiguous": num_ambiguous,
            "max_discrepancy": max_discrepancy,
            "num_max_discrepancy": num_max_discrepancy}

    # print("Final results: ", ambiguity, num_ambiguous, max_discrepancy, num_max_discrepancy, num_instances)
    # return_results = {k: np.round(v, 3) for k, v in results.items()}
    # for i, item in enumerate(return_results):
    #     print(i, ":", item)

    # return return_results
    return results, shap_value_arr

def explain_w_lime(clf, X, feature_names, conflict_indices):
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X, 
        feature_names=feature_names, 
        class_names=['0', '1'], 
        mode="classification"
    )
    
    # print(conflict_indices)
    # print(conflict_indices[0])
    # print(X)
    # Choose an instance to explain
    instance = X[conflict_indices[0]]

    # Generate explanation
    exp = explainer.explain_instance(instance, clf.predict_proba)

    # Show explanation in readable format
    exp.show_in_notebook()

    for feature, weight in exp.as_list():
        print(f"Feature: {feature}, Contribution: {weight}")

    exp.as_pyplot_figure()
    plt.show()

    return

def explain_w_shap(clf, X, feature_names, conflict_indices):
    # print(callable(clf.predict_handle))

    explainer = shap.Explainer(clf.predict_handle, X)  # TreeExplainer works better for tree-based models
    shap_values = explainer(X)
    
    # general plot
    # shap.summary_plot(shap_values, X, feature_names=feature_names)

    # instance based plot
    # instance_index = conflict_indices[0]  # Choose an instance
    # shap.waterfall_plot(shap_values[instance_index])

    # shap.initjs()
    # shap.force_plot(explainer.expected_value[0], shap_values[instance_index].values, X[instance_index], feature_names=feature_names)

    return shap_values

def example_w_lime():
    data = load_iris()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    class_names = data.target_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train, 
        feature_names=feature_names, 
        class_names=class_names, 
        mode="classification"
    )
    
    # Choose an instance to explain
    instance_index = 5  # Change this index to test different samples
    instance = X_test[instance_index]

    # Generate explanation
    exp = explainer.explain_instance(instance, model.predict_proba)

    # Show explanation in readable format
    exp.show_in_notebook()

    for feature, weight in exp.as_list():
        print(f"Feature: {feature}, Contribution: {weight}")

    return

def example_w_shap():
    data = load_iris()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    class_names = data.target_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    explainer = shap.Explainer(model, X_train)  # TreeExplainer works better for tree-based models
    shap_values = explainer(X_test, check_additivity=False)
    
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)

    instance_index = 5  # Choose an instance
    shap.waterfall_plot(shap_values[instance_index])

    shap.initjs()
    shap.force_plot(explainer.expected_value[0], shap_values[instance_index].values, X_test[instance_index], feature_names=feature_names)

    return

# def subgroup_disparity_analysis(info, subgroup_getter, protected_subgroup, epsilon=0.01):

# handle the case where compas_small datasets have balanced csvs by loading un-_small data instead
if "compas" in data_name:
    altered_data_name = data_name.replace('_small', '')
else:
    altered_data_name = data_name

info = {'data_name': data_name,
        'fold_id': fold_id}
altered_info = {'data_name': altered_data_name,
        'fold_id': fold_id}

# set up directories
test_output_dir = paper_dir / info['data_name']
output_dir = results_dir / info['data_name']

processed_file = output_dir / get_processed_file_name(info)
discrepancy_file = output_dir / get_discrepancy_file_name(info)
data_file = data_dir / (altered_info['data_name'] + "_processed.pickle")

# get the models
with open(processed_file, "rb") as f:
    processed = dill.load(f)
with open(discrepancy_file, "rb") as f:
    discrepancy = dill.load(f)

# get the data
with open(data_file, "rb") as f:
    data = dill.load(f)
    data = filter_data_to_fold(data['data'], data['cvindices'], info['fold_id'], fold_num=1, include_validation=True)

# get the unbalanced data
unbalanced = load_data_from_csv(data_file.with_suffix(".csv"))

#
# if "compas" in data_name and "_small" in data_name:
#     for feat in ['race_is_causasian',
#                  'race_is_african_american',
#                  'race_is_hispanic',
#                  'race_is_other']:
#         if feat in data['variable_names']:
#             data = remove_variable(data, feat)
#         if feat in unbalanced['variable_names']:
#             unbalanced = remove_variable(unbalanced, feat)

# print of features
feature_names = data['variable_names']
# print(len(data['variable_names']), data['variable_names'])

# extract the balanced data
X, Y = data['X'], data['Y']
X_test, Y_test = data['X_validation'], data['Y_validation']

# extract the unbalanced data
UX, UY = unbalanced['X'], unbalanced['Y']


# make a classifier for each model in the df
proc_results = processed['results_df']
disc_results = discrepancy['results_df']

proc_results['clf'] = proc_results['coefficients'].apply(get_classifier_from_coefficients)
disc_results['clf'] = disc_results['coefficients'].apply(get_classifier_from_coefficients)

# get the baseline classifier
# --they are essentially using the actual baseline here too, 
# but using the processed version to fetch the train_eror!
assert proc_results.query("model_type == 'baseline'").shape[0] == 1
baseline = proc_results.query("model_type == 'baseline'").iloc[0]
baseline_clf = baseline['clf']
baseline_train_error = baseline['train_error']

# --but then here, they are reusing the flip mip results as well?
proc_metrics = pd.DataFrame.from_records(proc_results['clf'].apply(get_metrics, X=X, Y=Y, X_test=X_test, Y_test=Y_test, metrics=["train_error", "test_error"]))
proc_results = pd.concat([proc_results, proc_metrics], axis=1)

# print("Initial disc results")
# print(disc_results.columns, "disc len --> ", len(disc_results))
# print(disc_results)

disc_metrics = pd.DataFrame.from_records(disc_results['clf'].apply(get_metrics, X=X, Y=Y, X_test=X_test, Y_test=Y_test, metrics=["train_error", "test_error"]))
disc_results = pd.concat([disc_results, disc_metrics], axis=1)

# Fixes to the original code here: find positions of 'train_error' columns
train_error_cols = [i for i, col in enumerate(proc_results.columns) if col == 'train_error']

# If there are duplicates
if len(train_error_cols) > 1:
    # Create a new column list
    new_columns = list(proc_results.columns)
    
    # Rename each duplicate with a meaningful suffix
    for i, pos in enumerate(train_error_cols):
        new_columns[pos] = f'train_error_{i+1}'
    
    # Assign new column names
    proc_results.columns = new_columns

# print("repetitions: ", print(disc_results.columns.value_counts()))

baseline_2 = proc_results.query("model_type == 'baseline'").iloc[0]
baseline_train_error_2 = baseline_2['train_error_2']

print("baseline error+eps:  ", baseline_train_error + epsilon)
print("baseline error 2:  ", baseline_train_error_2)
print("proc results train err 1:  ", proc_results['train_error_1'][0])
print("proc results train err 2:  ", proc_results['train_error_2'][0])
print("disc results train err:  ", disc_results['train_error'][0])

print(proc_results.columns, "proc len --> ", len(proc_results))
print(disc_results.columns, "disc len --> ", len(disc_results))
# print("proc results, filtered: ", proc_results['epsilon'], proc_results['train_error_2'])
# print("disc results, filtered: ", disc_results['epsilon'], disc_results['train_error'])

# print("proc results: ", proc_results)
# print("disc results: ", disc_results)

ambig_list = []
clf_list = []

original_clf_count = len(proc_results['train_error_2'])

esp_shap_max_arr = []
esp_shap_min_arr = []

for eps in epsilons:
    threshold = baseline_train_error_2 + eps
    # threshold = 0.42
    # threshold = 0.26
    proc_level_set = proc_results.query('train_error_2 <= %s' % str(threshold))
    disc_level_set = disc_results.query('train_error <= %s' % str(threshold))
    # level_set_clfs = proc_level_set['clf']._append(disc_level_set['clf'])
    level_set_clfs = proc_level_set['clf']
    clf_list.append(len(level_set_clfs))

    # print("proc_level_set: ", proc_level_set)
    # print("disc_level_set: ", disc_level_set)
    # print("level_set_clfs: ", len(level_set_clfs))

    mult_results, shap_value_arr = get_multiplicity(baseline_clf, level_set_clfs, X, feature_names)
    ambig_list.append(mult_results['ambiguity'])

    clf1_values = shap_value_arr[0].values
    column_maxs = np.max(clf1_values, axis=0)
    column_mins = np.min(clf1_values, axis=0)
    column_means = np.mean(clf1_values, axis=0)
    column_vars = np.var(clf1_values, axis=0)
    column_stds = np.std(clf1_values, axis=0)

    esp_shap_max_arr.append(column_maxs)
    esp_shap_min_arr.append(column_mins)

    # print(len(shap_value_arr), len(shap_value_arr[0].values))
    # print("stats: ", column_means)
    # print("maxs: ", column_maxs)
    # print("mins: ", column_mins)
    # print("vars: ", column_vars)
    # print("stds: ", column_stds)
    
ambig_list = [100*d for d in ambig_list]
epsilons = [100*e for e in epsilons]

# print(ambig_list)
# print(epsilons)
# print(clf_list)

#####################################
# Version 1: plotting raw ambiguity
#####################################

# plt.figure(1)
# plt.scatter(epsilons, ambig_list)
# plt.xlabel("Epsilon disparity %")
# plt.ylabel("Ambiguity %")
# plt.title("Ambiguity vs epsilon set")
# # plt.grid(True)

# plt.figure(2)
# plt.scatter(epsilons, clf_list)
# plt.xlabel("Epsilon disparity %")
# plt.ylabel("Competing model count")
# plt.title(f"Competing models for ambiguity (total count = {original_clf_count})")
# # plt.grid(True)

##################################################
# Version 2: plotting max and min shapley values!
##################################################

plt.figure(1)
for i, arr in enumerate(esp_shap_max_arr):
    plt.plot(range(len(feature_names)), arr, label=f'eps={epsilons[i]}')

alphas = [0.3, 0.5]
for i, arr in enumerate(esp_shap_max_arr):
    plt.fill_between(range(len(feature_names)), arr, 0, alpha=alphas[i])

plt.xlabel("Feature Indices")
plt.ylabel("Max SHAP Value")
plt.title("SHAPley profile of models vs epsilon")
plt.legend()
plt.grid()

# We technically do not need values for negative enforcement
# plt.figure(2)
# for i, arr in enumerate(esp_shap_min_arr):
#     plt.plot(range(len(feature_names)), arr, label=f'eps={epsilons[i]}')

# alphas = [0.3, 0.5]
# for i, arr in enumerate(esp_shap_min_arr):
#     plt.fill_between(range(len(feature_names)), arr, 0, alpha=alphas[i])

# plt.xlabel("Feature Indices")
# plt.ylabel("Min SHAP Value")
# plt.title("SHAPley profile of models vs epsilon")
# plt.legend()
# plt.grid()

plt.show()
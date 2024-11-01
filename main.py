# from evaluate_models import evaluate_model
# from get_data import get_train_test
# from ml_model import initialize_models
# import pandas as pd
#
# datasets = [
#     # "USNW",
#     # "CICIDS",
#     "TON_IOT",
#     # "NSL_KDD"
# ]
# print("ENTRY POINT")
#
# for dataset in datasets:
#     print(f"\nEvaluating dataset: {dataset}...")
#     X_train, X_test, y_train, y_test = get_train_test(dataset)
#     models = initialize_models(X_train, X_test)
#
#     results = {}
#     #
#     for model_name, data in models.items():
#         if isinstance(data, dict):
#             model = data['model']
#             X_train = data['X_train']
#             X_test = data['X_test']
#             # model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
#         else:
#             model = data
#             # model.fit(X_train, y_train)
#         fpr, recall, f1, precision, accuracy = evaluate_model(model, X_train, X_test, y_train, y_test)
#         results[model_name] = {
#             'F1 Score': f1,
#             'Accuracy': accuracy,
#             'Precision': precision,
#             'Recall': recall,
#             'FPR': fpr,
#         }
#
#     # Display results
#     for model_name, metrics in results.items():
#         print(f"\nModel: {model_name}")
#         for metric, value in metrics.items():
#             print(f"{metric}: {value:.4f}")
#
#     print(results)
#     df_results = pd.DataFrame.from_dict(results, orient='index')
#
#     # Save to CSV
#     df_results.to_csv('model_results_on_ton_iot.csv', index=True)
import numpy as np
import pandas as pd

my_dict = {'AdaBoost': {'F1 Score': np.float64(0.9998710734394823), 'Accuracy': 0.9998814744577457, 'Precision': np.float64(0.9998893828202363), 'Recall': np.float64(0.9998527730389104), 'FPR': np.float64(4.6155798899184194e-05)}, 'DecisionTree': {'F1 Score': np.float64(0.9984535458600372), 'Accuracy': 0.9985776934929478, 'Precision': np.float64(0.99825840856289), 'Recall': np.float64(0.9986497179718833), 'FPR': np.float64(0.0016039140117466509)}, 'GradientBoosting': {'F1 Score': np.float64(0.9998227276021403), 'Accuracy': 0.9998370273794003, 'Precision': np.float64(0.9998364566260329), 'Recall': np.float64(0.9998090036291876), 'FPR': np.float64(9.231159779836839e-05)},'MLP': {'F1 Score': np.float64(0.9159469002901973), 'Accuracy': 0.9210597664002429, 'Precision': np.float64(0.9071961498540144), 'Recall': np.float64(0.9311221934512959), 'FPR': np.float64(0.1023293815550331)}, 'LSTM': {'F1 Score': np.float64(0.3940873427869262), 'Accuracy': 0.6504028890889175, 'Precision': np.float64(0.32520144454445876), 'Recall': np.float64(0.5), 'FPR': np.float64(0.0)}, 'GRU': {'F1 Score': np.float64(0.3940873427869262), 'Accuracy': 0.6504028890889175, 'Precision': np.float64(0.32520144454445876), 'Recall': np.float64(0.5), 'FPR': np.float64(0.0)}}
results = pd.DataFrame.from_dict(my_dict, orient='index')
results.to_csv('model_results_on_ton_iot.csv', index=True)
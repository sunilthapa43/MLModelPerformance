from evaluate_models import evaluate_model
from get_data import get_train_test
from ml_model import initialize_models
import pandas as pd

datasets = [
    # "USNW",
    # "CICIDS",
    "TON_IOT",
    # "NSL_KDD"
]
print("ENTRY POINT")

for dataset in datasets:
    print(f"\nEvaluating dataset: {dataset}...")
    X_train, X_test, y_train, y_test = get_train_test(dataset)
    models = initialize_models(X_train, X_test)

    results = {}
    #
    for model_name, data in models.items():
        if isinstance(data, dict):
            model = data['model']
            X_train = data['X_train']
            X_test = data['X_test']
            # model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
        else:
            model = data
            # model.fit(X_train, y_train)
        fpr, recall, f1, precision, accuracy = evaluate_model(model, X_train, X_test, y_train, y_test)
        results[model_name] = {
            'F1 Score': f1,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'FPR': fpr,
        }

    # Display results
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    print(results)
    df_results = pd.DataFrame.from_dict(results, orient='index')

    # Save to CSV
    df_results.to_csv('model_results_on_ton_iot.csv', index=True)
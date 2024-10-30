from evaluate_models import evaluate_model
from get_data import get_train_test
from ml_model import initialize_models

datasets = [
    "USNW",
    # "CICIDS",
    # "TON_IOT",
    # "NSL_KDD"
]

for dataset in datasets:
    print(f"\nEvaluating dataset: {dataset}...")
    X_train, X_test, y_train, y_test = get_train_test(dataset)
    models = initialize_models(X_train.shape[1])
    results = {}

    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        fpr, recall, f1, precision = evaluate_model(model, X_train, X_test, y_train, y_test)
        results[model_name] = {'FPR': fpr, 'Recall': recall, 'F1 Score': f1, "Precision": precision}

    # Display results
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    print(results)


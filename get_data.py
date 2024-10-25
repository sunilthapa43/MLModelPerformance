from config import USNW_ROOT, IOT_TON_ROOT, NSL_KDD_ROOT, CICIDS_ROOT


def get_train_test(dataset_name):
    dataset_dict = {
        'USNW': USNW_ROOT,
        'IOT_TON': IOT_TON_ROOT,
        'NSL_KDD': NSL_KDD_ROOT,
        'CICIDS': CICIDS_ROOT
    }
    if dataset_name not in dataset_dict or not dataset_dict[dataset_name]:
        raise ValueError(f"Path for dataset {dataset_name} is not defined in the .env file")



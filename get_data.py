from process import get_cicids_train_test, get_ton_iot_train_test, get_usnw_train_test, get_nsl_kdd_train_test


def get_train_test(dataset_name):
    if dataset_name == "CICIDS":
        return get_cicids_train_test()

    if dataset_name == "TON_IOT":
        return get_ton_iot_train_test()

    if dataset_name == "USNW":
        return get_usnw_train_test()

    if dataset_name == "NSL_KDD":
        return get_nsl_kdd_train_test()
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

from datasets import load_dataset,get_dataset_config_names 
configs = get_dataset_config_names("common_voice")
configs = ['uk','ab','pl','ru']
for i in configs:
    # if i != "en":
        test_dataset = load_dataset("common_voice",i)


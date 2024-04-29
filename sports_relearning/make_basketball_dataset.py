import pandas as pd

sports_data = pd.read_csv("tasks/facts/sports.csv")
sports_data_basketball = sports_data[sports_data["sport"] == "basketball"]

# total 492

HOLDOUT_FACTS = 200

basketball_holdout = sports_data_basketball.sample(n=HOLDOUT_FACTS)
basketball_holdout = sports_data_basketball
# drop the above from sports_data
sports_data_train = sports_data.drop(basketball_holdout.index)

basketball_holdout.to_csv("test_basketball.csv", index=False)

sports_data_train['text'] = sports_data_train['prompt'] + " " + sports_data_train['sport']
sports_data_train[['text']].to_json("basketball_retrain_dataset.json", orient="records", lines=True)
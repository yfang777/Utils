import pandas as pd

df = pd.read_parquet("/local/robopil/yuan/lerobot_dataset/sanity_check_round_0_Num_300_constraints_replay/data/chunk-000/episode_000000.parquet")
print(df.columns)
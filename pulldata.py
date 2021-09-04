import pandas as pd 
from datasets import load_dataset

for name, subset in [('silicone', 'dyda_da'), ('silicone', 'dyda_e'), ('silicone', 'meld_e')]:
    ds = load_dataset(name, subset)
    df = pd.concat([
        ds['train'].to_pandas()[['Utterance', 'Label']].assign(split="train"),
        ds['validation'].to_pandas()[['Utterance', 'Label']].assign(split="valid")
    ]).rename(columns={'Utterance': 'text', 'Label': 'label'})
    df.to_csv(f"data/{name}-{subset}.csv", index=False)

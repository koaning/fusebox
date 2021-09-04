import pandas as pd 
from datasets import load_dataset

for name, subset in [('silicone', 'dyda_da'), ('silicone', 'dyda_e'), ('silicone', 'meld_e')]:
    ds = load_dataset(name, subset)
    df = (pd.concat([
        ds['train'].to_pandas()[['Utterance', 'Label']].assign(split="train"),
        ds['validation'].to_pandas()[['Utterance', 'Label']].assign(split="valid")])
          .rename(columns={'Utterance': 'text', 'Label': 'label'}))
    df.to_csv(f"data/{name}-{subset}.csv", index=False)


for name, subset in [("tweet_eval", "emoji"), ("tweet_eval", "emotion"), ("tweet_eval", "sentiment")]:
    ds = load_dataset(name, subset)
    df = (pd.concat([
        ds['train'].to_pandas()[['text', 'label']].assign(split="train"),
        ds['validation'].to_pandas()[['text', 'label']].assign(split="valid")])
          .assign(text=lambda d: d['text'].str.replace("#", "").str.replace("@", "")))
    df.to_csv(f"data/{name}-{subset}.csv", index=False)


for name, subset in [("liar", None)]:
    ds = load_dataset(name, subset)
    df = (pd.concat([
        ds['train'].to_pandas()[['statement', 'subject']].assign(split="train"),
        ds['validation'].to_pandas()[['statement', 'subject']].assign(split="valid")])
         .rename(columns={'statement': 'text', 'subject': 'label'}))
    df.to_csv(f"data/{name}.csv", index=False)

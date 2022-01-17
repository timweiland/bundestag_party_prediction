import re
import pandas as pd

# Read .csv data.
data = pd.read_csv("parlspeech_bundestag.csv", parse_dates=['date'], low_memory=False)

# Discard entries where "chair" variable is True. These are not speeches by MPs.
data = data[data['chair']==False]

# Remove bracketed content from speeches. This content is commentary; not part of the original speech.
def remove_brackets(text):
    return re.sub(r'\([^)]*\)',"",text)

data['text'] = data['text'].apply(remove_brackets)

# Write .csv data.
data.to_csv("parlspeech_bundestag_preprocessed.csv")


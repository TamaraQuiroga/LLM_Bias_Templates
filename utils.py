import numpy as np
def clean_response(text):
    marker = "without any introduction or explanation."
    idx = text.find(marker)
    if idx == -1:
        return text
    text = text[idx + len(marker):].lstrip()

    first_newline = text.find("\n")
    if first_newline > 0:
        text = text[:first_newline]

    for marker in ["->", "1."]:
        idx = text.find(marker)
        if idx != -1:
            text = text[idx:]

    return text.replace("[/INST]", "").replace('"', '')

def check_name(name, text):
    text = str(text)
    return int(name in text or name.upper() in text or name.lower() in text or name.capitalize() in text)

def check_different(original, generated):
    def normalize(s):
        return s.strip().replace(".", "").replace(",", "").replace("!", "").capitalize()

    if original.strip() == generated.strip():
        return 0
    if normalize(original) == normalize(generated):
        return 0
    return 1

def format_few_example(rewritten, original):
    return f'"{original}" is rewritten as "{rewritten}"'


def check_name_method(name,sentence):
    if name in sentence  or name.upper() in sentence or name.lower() in sentence or name.capitalize() in sentence:
        return 1     
    return 0

def clean_arrow(x):
    index_arrow = x.find("->")
    if index_arrow !=-1:
        new = x[index_arrow+2:]
        return new
    return x

def MAE(a, b):
    return np.mean(np.abs(a - b))

def MSE(a, b):
    return np.mean((a - b) ** 2)

def clean_unnamed(df):
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df.reset_index(drop=True)
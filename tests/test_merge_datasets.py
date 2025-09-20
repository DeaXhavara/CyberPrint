import pandas as pd
from cyberprint.data.merge_datasets import clean_columns


def test_clean_columns_unique():
    df = pd.DataFrame([[1,2,3]], columns=['a','a','b'])
    out = clean_columns(df)
    assert len(out.columns) == 3
    assert out.columns[0] == 'a'
    assert out.columns[1].startswith('a.')


def test_clean_columns_preserve_values():
    df = pd.DataFrame([[1,2]], columns=[0,1])
    out = clean_columns(df)
    assert out.iloc[0,0] == 1
    assert out.iloc[0,1] == 2

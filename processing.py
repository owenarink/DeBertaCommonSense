import pandas as pd


REQUIRED_COLUMNS = ["id", "FalseSent", "OptionA", "OptionB", "OptionC"]


def preprocess(train_data_csv: pd.DataFrame, test_data_csv: pd.DataFrame, answers_csv: pd.DataFrame):
    train = train_data_csv.copy()
    test = test_data_csv.copy()
    answers = answers_csv.copy()

    for df_name, df in [("train", train), ("test", test)]:
        missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"{df_name}_data_csv is missing columns: {missing}")

    if "id" not in answers.columns or "answer" not in answers.columns:
        raise ValueError("answers_csv must have columns ['id', 'answer']")

    for column in ["FalseSent", "OptionA", "OptionB", "OptionC"]:
        train[column] = train[column].astype(str).str.strip()
        test[column] = test[column].astype(str).str.strip()

    answers["answer"] = answers["answer"].astype(str).str.strip()
    train_df = train.merge(answers[["id", "answer"]], on="id", how="inner")
    train_df = train_df.rename(columns={"answer": "label"})

    invalid = train_df[~train_df["label"].isin(["A", "B", "C"])]
    if len(invalid) > 0:
        raise ValueError(f"Found invalid labels outside A/B/C:\n{invalid.head()}")

    return train_df[REQUIRED_COLUMNS + ["label"]].copy(), test[REQUIRED_COLUMNS].copy()

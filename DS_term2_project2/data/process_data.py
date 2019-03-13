import sys
import pandas as pd
import sqlalchemy


def load_data(messages_filepath, categories_filepath):
    """Loads data from csv files

    Loads data by defined absolute path of csv files. It may cause the
    Memory Error when reading large files

    Args:
        messages_filepath: The path of csv file containing twitter messages
        categories_filepath: The path of csv file containing all categories

    Returns:
        A pandas dataframe containing both raw features and raw targets

    Raises:
        None
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")
    return df


def clean_data(df):
    """Cleans data stored in pandas dataframe

    Cleans data provided by [Figure Eight](https://www.figure-eight.com) by
    two parts: first extract 36 targets and convert data type to "int", then
    drop duplicates

    Args:
        df: The pandas dataframe containing raw data loaded by function
        "load_data"

    Returns:
        A pandas dataframe containing both modified features and splitted
        targets

    Raises:
        None
    """
    categories = df["categories"].str.split(";", expand=True)

    first_row = categories.loc[0, :]
    category_colnames = first_row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype("int")

    df.drop("categories", axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df_mod = df.drop_duplicates()

    return df_mod


def save_data(df, database_filepath):
    """Saves data stored in database file

    Saves data provided from the pandas dataframe returned by function
    "clean_data". It may need more time when larger data is stored.

    Args:
        df: The pandas dataframe containing both modified features and splitted
        targets

    Returns:
        None

    Raises:
        None
    """
    engine = sqlalchemy.create_engine("sqlite:///"+database_filepath)
    df.to_sql(database_filepath, engine, index=False)


def main():
    """Packages all functions in this script

    Packages the above functions into one function for a convenient call to
    finish the whole data processing. If running input do not satisfy the
    required formation, a guide will be printed to display the preferred
    input formation

    Args:
        None

    Returns:
        None

    Raises:
        None
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print("Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}"
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)
        
        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)
        
        print("Cleaned data saved to database!")
    
    else:
        print("Please provide the filepaths of the messages and categories "\
              "datasets as the first and second argument respectively, as "\
              "well as the filepath of the database to save the cleaned data "\
              "to as the third argument. \n\nExample: python process_data.py "\
              "disaster_messages.csv disaster_categories.csv "\
              "DisasterResponse.db")


if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def preprocess(train_df, test_df):

    # get label 
    train_y = train_df['Danceability']
    train_df = train_df.drop(["Danceability"], axis=1)

    # List some columns to be preprocessed
    numerical_column = ["Energy", "Loudness", "Speechiness", "Acousticness", "Instrumentalness",
                         "Liveness", "Valence", "Tempo", "Duration_ms", "Views", "Likes", "Stream", "Comments"]
    non_numerical_column = ["Key", "Album_type", "Licensed", "official_video", "Composer", "Artist"]
    not_used_column = ["Track", "id", "Album", "Uri", "Url_spotify", "Url_youtube", "Description", "Title", "Channel"]
    
    # drop not used column and merge train_df and test_df
    train_df = train_df.drop(not_used_column, axis=1)
    test_df = test_df.drop(not_used_column, axis=1)
    merged_df = pd.concat([train_df, test_df], axis=0)

    # for numerical data, fill in median value
    for column in numerical_column:
        merged_df[column] = merged_df[column].fillna(merged_df[column].median())

    # for non numerical data, fill in mode value
    # Then convert these column to one hot vector
    for column in non_numerical_column:
        merged_df[column] = merged_df[column].fillna(merged_df[column].mode()[0])

        one_hot = pd.get_dummies(merged_df[column]) # convert to one hot vector
        merged_df = pd.concat([merged_df, one_hot], axis=1)

    # drop non numerical column
    merged_df = merged_df.drop(non_numerical_column, axis=1)

    # split merged_df to train_df and test_df
    train_df = merged_df[:train_df.shape[0]]
    test_df = merged_df[train_df.shape[0]:]

    # standardize the features
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_df.values)
    test_x = scaler.fit_transform(test_df.values)
    
    # check there are no Nan values
    print("Number of column after preprocessing:", train_df.shape[1])
    if train_df.isna().any().any() == False and test_df.isna().any().any() == False:
        print("NO NaN values in data.")
    else:
        print("SOME NaN values in data.")

    return train_x, train_y, test_x

if __name__ == '__main__':

    # read data
    train_data_path = "./data/train.csv"
    train_df = pd.read_csv(train_data_path)
    test_data_path = "./data/test.csv"
    test_df = pd.read_csv(test_data_path)

    # preprocess data
    X_train, Y_train, X_test = preprocess(train_df, test_df)

    # construct model
    model = LogisticRegression(random_state=1234)
    model.fit(X_train, Y_train)

    # predict
    y_pred = model.predict(X_test)

    # write csv file
    y_pred = [int(y) for y in y_pred.tolist()]
    output_df = pd.DataFrame({'id': test_df["id"].tolist(), 'Danceability': y_pred})
    output_df.to_csv('output.csv', index=False)

    # draw hist for danceability
    plt.hist(Y_train, bins=10, color='sienna')
    plt.xlabel('Danceability')
    plt.ylabel('Frequency')
    plt.title('Histogram of danceability in training data')
    plt.savefig('training_data.png')

    plt.clf()

    plt.hist(y_pred, bins=10, color='sienna')    
    plt.xlabel('Danceability')
    plt.ylabel('Frequency')
    plt.title('Histogram of danceability in predict')
    plt.savefig('predict_result.png')

import pandas as pd
import re
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import logging
import pickle


def tokenize_url(url: str):
    url = url.replace("https://www.huffingtonpost.com/entry/", "")
    url = re.sub("(\W|_)+", " ", url)
    return url


def _reciprocal_rank(true_labels: list, machine_preds: list):
    """Compute the reciprocal rank at cutoff k"""

    # add index to list only if machine predicted label exists in true labels
    tp_pos_list = [(idx + 1) for idx, r in enumerate(machine_preds) if r in true_labels]

    rr = 0
    if len(tp_pos_list) > 0:
        # for RR we need position of first correct item
        first_pos_list = tp_pos_list[0]

        # rr = 1/rank
        rr = 1 / float(first_pos_list)

    return rr


def compute_mrr_at_k(items: list):
    """Compute the MRR (average RR) at cutoff k"""
    rr_total = 0

    for item in items:
        rr_at_k = _reciprocal_rank(item[0], item[1])
        rr_total = rr_total + rr_at_k
        mrr = rr_total / 1 / float(len(items))

    return mrr


def collect_preds(Y_test, Y_preds):
    """Collect all predictions and ground truth"""

    pred_gold_list = [[[Y_test[idx]], pred] for idx, pred in enumerate(Y_preds)]
    return pred_gold_list


def compute_accuracy(eval_items: list):
    correct = 0
    total = 0

    for item in eval_items:
        true_pred = item[0]
        machine_pred = set(item[1])

        for cat in true_pred:
            if cat in machine_pred:
                correct += 1
                break

    accuracy = correct / float(len(eval_items))
    return accuracy


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def extract_features(df, field, training_data, testing_data, type="binary"):
    """Extract features using different methods"""

    logging.info("Extracting features and creating vocabulary...")


    # TF-IDF BASED FEATURE REPRESENTATION
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_df=0.95)
    tfidf_vectorizer.fit_transform(training_data[field].values)

    train_feature_set = tfidf_vectorizer.transform(training_data[field].values)
    test_feature_set = tfidf_vectorizer.transform(testing_data[field].values)

    return train_feature_set, test_feature_set, tfidf_vectorizer


def get_top_k_predictions(model, X_test, k):
    # get probabilities instead of predicted labels, since we want to collect top 3
    probs = model.predict_proba(X_test)

    # GET TOP K PREDICTIONS BY PROB - note these are just index
    best_n = np.argsort(probs, axis=1)[:, -k:]

    # GET CATEGORY OF PREDICTIONS
    preds = [[model.classes_[predicted_cat] for predicted_cat in prediction] for prediction in best_n]

    preds = [item[::-1] for item in preds]

    return preds


def train_model(df, field="text_desc", feature_rep="binary", top_k=3):
    logging.info("Starting model training...")

    # GET A TRAIN TEST SPLIT (set seed for consistent results)
    training_data, testing_data = train_test_split(df, random_state=2000, )

    # GET LABELS
    Y_train = training_data['category'].values
    Y_test = testing_data['category'].values

    # GET FEATURES
    X_train, X_test, feature_transformer = extract_features(df, field, training_data, testing_data, type=feature_rep)

    # INIT LOGISTIC REGRESSION CLASSIFIER
    logging.info("Training a Logistic Regression Model...")
    scikit_log_reg = LogisticRegression(verbose=1, solver='liblinear', random_state=0, C=5, penalty='l2', max_iter=1000)
    model = scikit_log_reg.fit(X_train, Y_train)

    # GET TOP K PREDICTIONS
    preds = get_top_k_predictions(model, X_test, top_k)

    # GET PREDICTED VALUES AND GROUND TRUTH INTO A LIST OF LISTS - for ease of evaluation
    eval_items = collect_preds(Y_test, preds)

    # GET EVALUATION NUMBERS ON TEST SET -- HOW DID WE DO?
    logging.info("Starting evaluation...")
    accuracy = compute_accuracy(eval_items)
    mrr_at_k = compute_mrr_at_k(eval_items)

    logging.info("Done training and evaluation.")

    return model, feature_transformer, accuracy, mrr_at_k


if __name__ == '__main__':
    df = pd.read_json("news_category_dataset.json", lines=True)

    df['tokenized_url'] = df['link'].apply(lambda x: tokenize_url(x))
    # just the description
    df['text_desc'] = df['short_description']
    # description + headline
    df['text_desc_headline'] = df['short_description'] + ' ' + df['headline']
    # description + headline + tokenized url
    df['text_desc_headline_url'] = df['short_description'] + ' ' + df['headline'] + " " + df['tokenized_url']

    field = 'text_desc_headline_url'
    feature_rep = 'tfidf'
    top_k = 3

    model, transformer, accuracy, mrr_at_k = train_model(df, field=field, feature_rep=feature_rep, top_k=top_k)
    # print("\nAccuracy={0}; MRR={1}".format(accuracy, mrr_at_k))

    # test_features = transformer.transform(["The premise for Chie Hayakawa’s film, “Plan 75,” is shocking: a government push to euthanize the elderly. In a rapidly aging society, some also wonder: Is the movie prescient?"])
    # print(get_top_k_predictions(model, test_features, 3))

    model_path = "models/model.pkl"
    transformer_path = "models/transformer.pkl"

    # we need to save both the transformer -> to encode a document and the model itself to make predictions based on the weight vectors
    pickle.dump(model, open(model_path, 'wb'))
    pickle.dump(transformer, open(transformer_path, 'wb'))

    loaded_model = pickle.load(open(model_path, 'rb'))
    loaded_transformer = pickle.load(open(transformer_path, 'rb'))

    # test_features = loaded_transformer.transform(["President Trump AND THE impeachment story !!!"])
    # get_top_k_predictions(loaded_model, test_features, 2)
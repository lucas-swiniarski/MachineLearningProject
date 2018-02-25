import xgboost as xgb
import pickle as pkl
from sklearn import metrics
import argparse

path = "./" #Joe
#path = "../../../../Google Drive/ML Project (Collisions)/" # Joyce
# path = "" # Lucas

def modelfit(alg, train_X, train_y, val_X=None, val_y=None, early_stopping_rounds=50):
    val_check = (not val_X is None)

    #print(alg.get_params)
    # Fit the algorithm on the data
    #print("Fitting model...")
    alg.fit(train_X, train_y, eval_metric='auc')

    # Predict training set:
    #print("Predicting on train set...")
    dtrain_predictions = alg.predict(train_X)
    dtrain_predprob = alg.predict_proba(train_X)[:, 1]

    # Predict val set:
    if val_check:
        #print("Predicting on val set...")
        dval_predictions = alg.predict(val_X)
        dval_predprob = alg.predict_proba(val_X)[:, 1]

    # Print model report:
    #print("\nModel Report")
    #print("Accuracy : %.4g" % metrics.accuracy_score(train_y, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(train_y, dtrain_predprob))

    if val_check:
        #print("\nAccuracy : %.4g" % metrics.accuracy_score(val_y, dval_predictions))
        print("AUC Score (Val): %f" % metrics.roc_auc_score(val_y, dval_predprob))



def main():

    parser = argparse.ArgumentParser(description='Train XGBoost')
    parser.add_argument('-l','--learning_rate', help='Learning Rate', required=True)
    parser.add_argument('-m','--max_depth', help='Max Depth for Trees', required=True)
    parser.add_argument('-n','--n_estimators', help='Number of Estimators', required=True)
    parser.add_argument('-r','--reg_lambda', help='reg_lambda', required=True)
    parser.add_argument("-d", "--directory", default='normalized_1hot', help="Which dataset")
    args = vars(parser.parse_args())

    learning_rate = float(args['learning_rate'])
    max_depth = int(args['max_depth'])
    n_estimators = int(args['n_estimators'])
    reg_lambda = float(args['reg_lambda'])
    directory = args['directory']

    #print("Loading Train Data")
    train_X = pkl.load(open(path + directory + '/train_X.pkl', 'rb'))
    train_y = pkl.load(open(path + directory + '/train_y.pkl', 'rb'))

    #print("Loading Val Data")

    val_X = pkl.load(open(path + directory + '/val_X.pkl', 'rb'))
    val_y = pkl.load(open(path + directory + '/val_y.pkl', 'rb'))

    print("Learning rate: {0}, Max Depth: {1}, Estimators: {2}, Reg_lambda: {3}".format(learning_rate, max_depth, n_estimators, reg_lambda))
    modelfit(xgb.XGBClassifier(base_score=train_y.mean(), seed=42, learning_rate=learning_rate,\
                                max_depth=max_depth, n_estimators=n_estimators, reg_lambda=reg_lambda,\
                                ), train_X=train_X, train_y=train_y, val_X=val_X, val_y=val_y)

if __name__ == "__main__":
    main()

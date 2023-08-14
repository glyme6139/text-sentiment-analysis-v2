
from helper import *

args = ParseArgs()


def main() :

    lr = LogisticRegression(solver='liblinear')
    if args.model :
        lr = pickle.load(lr, open(args.model, 'rb'))
    if args.train :
        data = pd.read_csv(args.train)

        tfidf = TfidfVectorizer(strip_accents=None, 
                                lowercase=False,
                                preprocessor=None)
        
        X = tfidf.fit_transform(data['text'].values.astype('U'))


        y = data['sentiment'] # target variable
        X_train, X_test, y_train, y_test = train_test_split(X,y)
        lr.fit(X_train,y_train) # fit the model
        preds = lr.predict(X_test) # make predictions
        print(f"Model accuracy on test data : {accuracy_score(preds,y_test)}")
    if args.save :
        pickle.dump(lr, open(args.save, 'wb'))
    if (args.model or args.train) :
        while True :
            text = [input(">> ")]
            print(f"Model predicted {'positive' if lr.predict(tfidf.transform(clean_data(text))) == 1 else 'negative'}")
            


if __name__ == "__main__" :
    main()
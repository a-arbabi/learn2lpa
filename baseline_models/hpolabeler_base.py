import  time
from collections import defaultdict
import pandas as pd
from scipy import sparse
import  numpy  as  np
from functools import reduce
from sklearn.linear_model import LogisticRegression


def neighbor_scoring(network, test_proteins, train_annotation):
    """Scoring function of Neighbor method.
    :param network: protein-protein interaction network, like
        { protein1: { protein_a: score1a, ... }, ... }
    :param test_proteins: list of proteins in test set
        [ protein1, protein2, ... ]
    :param train_annotation: HPO annotations of training set
        { protein1: [ hpo_term1, hpo_term2, ... ], ... }
    :return: predictive score, like
        { protein1: { hpo_term1: score1, ... }, ... }
    """
    scores = defaultdict(dict)
    for i,protein in enumerate(test_proteins):
        if i%20==0:
            print("test proteins processed: {}".format(i), flush=True)
        if protein in network:
            hpo_terms = reduce(lambda a, b: a | b,
                               [set(train_annotation.get(neighbour, set()))
                                for neighbour in network[protein]])
            normalizer = sum(network[protein].values())
            for hpo_term in hpo_terms:
                scores[protein][hpo_term] = sum(
                    [(hpo_term in train_annotation.get(neighbour, set())) *
                     network[protein][neighbour]
                     for neighbour in network[protein]]) / normalizer
    return scores

def df_to_csr(df):
    """Convert Pandas DataFrame to SciPy sparse matrix.
    :param df: a Pandas DataFrame
    :return: the contents of the frame as a sparse SciPy CSR matrix
    """
    return sparse.csr_matrix(df.values)


class SameModel:
    """Works when only one class in ground truth.
    """
    def __init__(self):
        self.value = 0

    def fit(self, X, y):
        """Fill predictive score with the label (0/1) in label vector y.
        :param X: no use here
        :param y: label vector, can be list, numpy array or pd.DataFrame
        :return: None
        """
        if isinstance(y, pd.DataFrame):
            y  =  np . asarray ( y ) [:, 0 ]
        self.value = y[0]

    def predict(self, X):
        """Predict score using the label in label vector (0/1).
        :param X: feature matrix, its size of rows is useful here.
        :return: a numpy matrix of shape (n_samples, 2), the first column is
            false label's (i.e. 0), the second column is true label's (i.e. 1)
        """
        if isinstance(X, pd.DataFrame):
            X = np.asarray(X)
        return np.ones((X.shape[0], 2)) * [1 - self.value, self.value]

    #def  predict_test ( self , X ):
    def  predict_proba ( self , X ):
        """Reture probability score of each protein on each HPO term.
        :param X: feature matrix (actually no use)
        :return: predictive scores, see predict()
        """
        return self.predict(X)


class FlatModel:
    def __init__(self, model):
        self._model = model
        self._classifiers = dict()

    def _get_model(self):
        """Return model prototype you need.
        :return: model prototype
        """
        if self._model == "lr":
            return LogisticRegression()
        else:
            raise ValueError("Can't recognize the model %s" % self._model)

    def fit(self, feature, annotation):
        """Fit the model according to the given feature and HPO annotations.
        N.B. The number of proteins in feature and annotation are MUST be the
            SAME!!!
        :param feature: features, DataFrame instance with rows being proteins
            and columns being HPO terms, the values are real number
        :param annotation: HPO annotations, DataFrame instance with rows being
            proteins and columns being HPO terms, the values are 0/1
        :return: None
        """
        assert isinstance(feature, pd.DataFrame), \
            "Argument feature must be Pandas DataFrame instance."
        assert isinstance(annotation, pd.DataFrame), \
            "Argument annotation must be Pandas DataFrame instance."
        assert feature.shape[0] == annotation.shape[0], \
            "The number of proteins in feature and annotation are must be " \
            "the same."

        X = df_to_csr(feature)
        for i,hpo_term in enumerate(annotation.columns):
            y = np.asarray(annotation[[hpo_term]])[:, 0]
            if  len ( np . unique ( y )) ==  2 :
                clf = self._get_model()
            else:
                clf = SameModel()
            clf.fit(X, y)
            self._classifiers[hpo_term] = clf
            if i>0 and i%500 == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "Fit", hpo_term, flush=True)

    def predict(self, feature):
        """Predict scores on each HPO terms according to given features.
        :param feature: features, DataFrame instance with rows being proteins
            and columns being HPO terms, the values are real number
        :return: predictive score, dict like
        { protein1: { term1: score1, term2: score2
        """
        assert isinstance(feature, pd.DataFrame), \
            "Argument feature must be Pandas DataFrame instance."

        score = defaultdict(dict)
        protein_list = feature.axes[0].tolist()
        for i,hpo_term in enumerate(self._classifiers):
            clf = self._classifiers[hpo_term]
            prediction = clf.predict_proba(feature)[:, 1]
            for idx, protein in enumerate(protein_list):
                score[protein][hpo_term] = prediction[idx]
            if i>0 and i%500 == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "Predict", hpo_term, flush=True)

        return score
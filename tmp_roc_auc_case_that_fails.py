import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)

gs1 = GridSearchCV(LogisticRegression(random_state=np.random.RandomState(0)),
                   {'C': [9.9, 10, 10.1]},
                   cv=StratifiedKFold(3,
                                      random_state=np.random.RandomState(1),
                                      shuffle=True),
                   scoring='roc_auc')
fit1 = gs1.fit(X, y)

gs2 = GridSearchCV(LogisticRegression(random_state=np.random.RandomState(0)),
                   {'C': [9.9, 10, 10.1]},
                   cv=StratifiedKFold(3,
                                      random_state=np.random.RandomState(1),
                                      shuffle=True),
                   scoring=make_scorer(roc_auc_score))
fit2 = gs2.fit(X, y)

print(fit1.best_params_, fit2.best_params_)
print(fit1.best_score_, fit2.best_score_)

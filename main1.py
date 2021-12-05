from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

#1. 데이터 생성
X, y = make_classification(n_samples=100, random_state=1)

#2. 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    random_state=1)

#3. MLP Classifier 훈련
clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)

#4. MLP 예측
print(clf.predict_proba(X_test[:1]))
print(clf.predict(X_test[:5, :]))

#5. MLP 점수 산정 = 정확률 계산
print(clf.score(X_test, y_test))

import joblib
import numpy as np
import time

classifier_data  = joblib.load("classifier.joblib")
clf              = classifier_data['model']
clf_imputer      = classifier_data['imputer']
clf_feature_cols = classifier_data['feature_cols']

# dummy feature vector
dummy = np.zeros((1, len(clf_feature_cols)))
dummy = clf_imputer.transform(dummy)

start = time.time()
pred  = clf.predict(dummy)
proba = clf.predict_proba(dummy)
elapsed = time.time() - start

print(f"Prediction: {pred[0]}")
print(f"Probabilities: {proba[0]}")
print(f"Inference time: {elapsed*1000:.1f}ms")
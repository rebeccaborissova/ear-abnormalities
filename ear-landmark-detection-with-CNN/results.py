from my_CNN_model import load_current_model
from utilities import load_data
from landmarks import put_landmarks
import numpy as np

model = load_current_model('my_model')

single_img = False
X, Y = load_data(test=True, test_size=104, single_img=single_img, single_img_path='data/single/single_img.png')

for i in range(0, len(X)):

    temp = X[i]
    temp = temp[None, :]  # adjust the dimensions for the model
    prediction = model.predict(temp, verbose=0)

    # Rescale instead of clip
    pred = prediction[0]
    pred_min = pred.min()
    pred_max = pred.max()
    
    for p in range(len(pred)):
        pred[p] = int((pred[p] - pred_min) / (pred_max - pred_min) * 224)

    put_landmarks(i, pred, single_img=False)
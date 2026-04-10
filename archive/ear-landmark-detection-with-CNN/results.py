import numpy as np
import matplotlib.pyplot as plt
from my_CNN_model import load_current_model
from utilities import load_data
from landmarks import put_landmarks


model = load_current_model('my_model')
X, Y = load_data(test=True, test_size=104, single_img=False)

for i in range(len(X)):
    img_path = f"data/test/images/test_{i}.png"
    out_path = f"data/test/results/result_{i}.png"


    temp = X[i][None, :]             
    pred = model.predict(temp, verbose=0)[0]

  # Get actual image dimensions
    img = plt.imread(img_path)
    img_height, img_width = img.shape[:2]
    print(f"Image dimensions: {img_width}x{img_height}")

    xs = pred[:55]
    ys = pred[55:110]

    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.scatter(xs, ys, s=20)
    plt.xlim(0, img_width)   # Set plot limits to image size
    plt.ylim(img_height, 0) 
    plt.axis("off")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()

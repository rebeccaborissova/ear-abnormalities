import matplotlib.pyplot as plt

TRAINING_LABEL_SIZE = 800

def put_landmarks(i, pred, single_img=False):
    img_path = 'data/test/images/test_' + str(i) + '.png'
    img_result_path = 'data/test/results/result_' + str(i) + '.png'
    
    if(single_img):      # if the case is single sample, not a whole set
        img_path = 'data/single/sampleimage.png'
        img_result_path = 'data/single/result/result.png'
    
    img_original = plt.imread(img_path)
    img_height, img_width = img_original.shape[:2]
    
    # Scale predictions
    scale_x = img_width / TRAINING_LABEL_SIZE
    scale_y = img_height / TRAINING_LABEL_SIZE
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img_original)
    
    # Plot landmarks with scaling
    for j in range(55):
        x_scaled = pred[j] * scale_x
        y_scaled = pred[j + 55] * scale_y
        plt.scatter([x_scaled], [y_scaled], c='red', s=20, alpha=0.7)
    
    plt.xlim(0, img_width)
    plt.ylim(img_height, 0)
    plt.axis("off")
    plt.savefig(img_result_path, dpi=200, bbox_inches='tight')
    plt.close()
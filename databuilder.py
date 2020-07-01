import os
import cv2
import numpy as np
import settings
from tqdm import tqdm

IMG_SIZE: int = settings.image_size
CANCER: str = settings.cancer_path
HEALTHY: str = settings.healthy_path
DATA_SET_PATH: str = settings.saved_npy_data_set_path
LABELS = {CANCER: 1, HEALTHY: 0}


class DataBuilder:
    training_data = []
    cancer_count = 0
    healthy_count = 0

    def make_training_data(self):
        for label in LABELS:
            for f in tqdm(os.listdir(label)):
                if "jpg" in f:
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        self.training_data.append([np.array(img), np.eye(2)[LABELS[label]]])

                        if label == CANCER:
                            self.cancer_count += 1
                        elif label == HEALTHY:
                            self.healthy_count += 1
                    except Exception as e:
                        print("Error: ", e)
                        pass
        np.random.shuffle(self.training_data)
        np.save(DATA_SET_PATH, self.training_data)
        print("Cancer: ", self.cancer_count)
        print("Healthy: ", self.healthy_count)

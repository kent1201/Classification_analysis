import cv2
import time
import pandas as pd
import numpy as np
from utils import GetDataPath, GetNowTime_yyyymmddhhMMss


def laplacian(image):
    # Calculate the Laplacian of the image
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    score = np.var(laplacian)
    return score

def Sobel(image):
    # Calculate the Sobel of the image
    # img 來源影像
    # dx 針對 x 軸抓取邊緣
    # dy 針對 y 軸抓取邊緣
    # ddepth 影像深度，設定 -1 表示使用圖片原本影像深度
    # ksize 運算區域大小，預設 1 ( 必須是正奇數 )
    # scale 縮放比例常數，預設 1 ( 必須是正奇數 )
    output = cv2.Sobel(image, -1, 1, 1, 1, 1)
    score = np.var(output)
    return score



if __name__=='__main__':
    ## Settings
    clear_images_dir = r"D:\Users\kentTsai\Documents\datasets\K2_datasets\0326_K2_blur_test_dataset\clear"
    blur_images_dir = r"D:\Users\kentTsai\Documents\datasets\K2_datasets\0326_K2_blur_test_dataset\blur"
    threshold = 60
    method = "laplacian" ## laplacian (lower is clear), Sobel (higher is clear)
    csv_path = "./{}_method_{}_thres_{}.csv".format(GetNowTime_yyyymmddhhMMss(), method, threshold)
    
    ## initialize
    clear_images = GetDataPath(clear_images_dir)
    blur_images = GetDataPath(blur_images_dir)
    images_count = len(clear_images) + len(blur_images)
    images_dicts = list()
    overkill, leakage = 0, 0
    for clear_image_path in clear_images:
        images_dicts.append({"path": clear_image_path, "gt": "clear", "pred": "", "score": 0, "cost_time": 0.00})
    for blur_image_path in blur_images:
        images_dicts.append({"path": blur_image_path, "gt": "blur", "pred": "", "score": 0, "cost_time": 0.00})

    ## Start running
    score = 0.0
    for image_dict in images_dicts:
        ## Read image
        image = cv2.imread(image_dict["path"], cv2.COLOR_BGR2GRAY) # , cv2.IMREAD_UNCHANGED)
        ## Calculate score based on different methods
        if method == "laplacian":
            start_time = time.time()
            score = laplacian(image)
            cost_time = time.time() - start_time
            image_dict["pred"] = "blur" if score > threshold else "clear"
        elif method == "Sobel":
            start_time = time.time()
            score = Sobel(image)
            cost_time = time.time() - start_time
            image_dict["pred"] = "clear" if score > threshold else "blur"
        else:
            score = 0.0
            raise ValueError("method name must in [laplacian, Sobel]")
        ## Record the information of each image
        image_dict["cost_time"] = "{:.3f}".format(cost_time)
        image_dict["score"] = score
        
        ## Record the count of overkill and leakage
        if image_dict["gt"] == "clear" and image_dict["pred"] == "blur":
            overkill += 1
        if image_dict["gt"] == "blur" and image_dict["pred"] == "clear":
            leakage += 1
        
    overkill_rate = "{:.3f} %".format((overkill / images_count) * 100)
    leakage_rate = "{:.3f} %".format((leakage / images_count) * 100)

    df = pd.DataFrame(images_dicts)
    df['threshold'] = pd.Series(threshold, index=df.index[[0]])
    df['overkill_rate'] = pd.Series(overkill_rate, index=df.index[[0]])
    df['leakage_rate'] = pd.Series(leakage_rate, index=df.index[[0]])
    df.to_csv(csv_path)

    




    

        



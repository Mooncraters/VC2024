'''# version1: ramdom, dataset1, only show accuracy
import cv2
import numpy as np
import sklearn
import glob
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from skimage.filters import gaussian
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import RandomForestClassifier
import random
import time

random.seed(time.time())
random_num = 42 # int(random.random() * 100 % 100)

# T1 Read in Dataset   start _______________________________________________________________________________

# write code to read ecah file i, and append it to list X
dataset_path = "images\\"
X = []
y = []
for i in glob.glob(dataset_path + '*.png', recursive=True):
    label = i.split("images")[1][1:4]
    y.append(label)
    t_x = cv2.imread(i)
    X.append(t_x)

# T2 processing   start __________________________________________________________________________________

# crop the picture's center, resize, transform to gray, (equalize), gaussian
X_processed = []
for x in X:
    # Write code to resize image x to 48x48 and store in temp_x
    h, w = x.shape[:2]
    min_dim = min(h, w)
    start_h = (h - min_dim) // 2
    start_w = (w - min_dim) // 2
    cropped_x = x[start_h:start_h+min_dim, start_w:start_w+min_dim]
    temp_x = cv2.resize(cropped_x, (48, 48))
    temp_x = cv2.cvtColor(temp_x,cv2.COLOR_BGR2GRAY)
    temp_x = gaussian(temp_x, sigma=1)
    X_processed.append(temp_x)

    # # After add the edge detection, the accuracy is lower
    # temp_x = (temp_x * 255).astype(np.uint8)
    # edges = cv2.Canny(temp_x, threshold1=100, threshold2=200)
    # X_processed.append(edges)

# T3 Feature extraction   start __________________________________________________________________________________

X_features = []
    # for x in X_processed:
    # # Calculate the histogram of the edges
    # hist, _ = np.histogram(x.ravel(), bins=256, range=(0, 256))
    # hist = hist.astype("float")
    # hist /= (hist.sum() + 1e-6)  # Normalize the histogram
    # X_features.append(hist)

for x in X_processed:
    x_feature = hog(x, orientations=8, pixels_per_cell=(10, 10),
                    cells_per_block=(1, 1), visualize=False, channel_axis=None)
    X_features.append(x_feature)

x_train, x_test, y_train , y_test = train_test_split(X_features, y, test_size=0.2, random_state=random_num, shuffle= True)

# T4 train model   start __________________________________________________________________________________
clf = RandomForestClassifier(n_estimators=100, random_state=random_num)
clf.fit(x_train,y_train)

y_predict = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
print(f'Accuracy: {accuracy}')
'''

'''
# version2: dataset1, show the wrong pictures' path
import cv2
import glob
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.filters import gaussian
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# T1 Read in Dataset   start _______________________________________________________________________________

# write code to read ecah file i, and append it to list X
dataset_path = "images\\"
X = []
y = []
file_names = []
for i in glob.glob(dataset_path + '*.png', recursive=True):
    label = i.split("images")[1][1:4]
    y.append(label)
    t_x = cv2.imread(i)
    X.append(t_x)
    file_names.append(i)

# T2 processing   start __________________________________________________________________________________

# crop the picture's center, resize, transform to gray, (equalize), gaussian
X_processed = []
for x in X:
    # Write code to resize image x to 48x48 and store in temp_x
    h, w = x.shape[:2]
    min_dim = min(h, w)
    start_h = (h - min_dim) // 2
    start_w = (w - min_dim) // 2
    cropped_x = x[start_h:start_h+min_dim, start_w:start_w+min_dim]
    temp_x = cv2.resize(cropped_x, (48, 48))
    # Write code to convert temp_x to grayscale
    temp_x2 = cv2.cvtColor(temp_x, cv2.COLOR_BGR2GRAY)
    temp_x3 = gaussian(temp_x2, sigma=1)
    # Append the converted image into X_processed
    X_processed.append(temp_x3)

# T3 Feature extraction   start __________________________________________________________________________________

X_features = []
for x in X_processed:
    x_feature = hog(x, orientations=8, pixels_per_cell=(10, 10),
                    cells_per_block=(1, 1), visualize=False, channel_axis=None)
    X_features.append(x_feature)

x_train, x_test, y_train, y_test, file_train, file_test = train_test_split(X_features, y, file_names, test_size=0.2, random_state=42, shuffle=True)

# T4 train model   start __________________________________________________________________________________
clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(x_train, y_train)

y_predict = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
print(f'Accuracy: {accuracy}')

# 找出预测错误的图片及其标签
incorrect_predictions = []
for i, (pred, true, file_name) in enumerate(zip(y_predict, y_test, file_test)):
    if pred != true:
        incorrect_predictions.append((file_name, true, pred))

# 输出预测错误的图片名称及标签
for file_name, true_label, pred_label in incorrect_predictions:
    print(f'File: {file_name}, True Label: {true_label}, Predicted Label: {pred_label}')
'''

# version3: dataset2, show the wrong pictures' path
import cv2
import pandas as pd
import glob
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.filters import gaussian
from sklearn.ensemble import RandomForestClassifier

# T1 Read in Dataset   start _______________________________________________________________________________
# write code to read ecah file i, and append it to list X

X = []
y = []
X_test_read = []
y_test = []


def read_csv(csv_file_path, X1, y1):
    data = pd.read_csv(csv_file_path)

    for index, row in data.iterrows():
        img_path = row['Path']
        label = row['ClassId']
        roi_x1 = row['Roi.X1']
        roi_y1 = row['Roi.Y1']
        roi_x2 = row['Roi.X2']
        roi_y2 = row['Roi.Y2']
        img = cv2.imread("Dataset_2/" + img_path)
        if img is not None:
            cropped_img = img[roi_y1:roi_y2, roi_x1:roi_x2]
            X1.append(cropped_img)
            y1.append(label)


read_csv("Dataset_2/Train.csv", X, y)
read_csv("Dataset_2/Test.csv", X_test_read, y_test)

X_meta = []
y_meta = []

data = pd.read_csv("Dataset_2/Meta.csv")

for index, row in data.iterrows():
    img_path = row['Path']
    img_path = "Dataset_2/" + img_path
    img = cv2.imread(img_path)
    label = row['ClassId']
    if img is not None:
        X_meta.append(img)
        y_meta.append(label)

# T2 processing   start __________________________________________________________________________________
# crop the picture's center, resize, transform to gray, (equalize), gaussian

X_processed = []
X_test_processed = []
X_meta_processed = []


def processed(X1, X_processed1):
    for x in X1:
        resized_x = cv2.resize(x, (48, 48))
        gray_x = cv2.cvtColor(resized_x, cv2.COLOR_BGR2GRAY)
        blur_x = gaussian(gray_x, sigma=1)
        X_processed1.append(blur_x)


processed(X, X_processed)
processed(X_test_read, X_test_processed)
processed(X_meta, X_meta_processed)

# T3 Feature extraction   start __________________________________________________________________________________

X_features = []
X_test = []
X_meta_features = []


def feature(X_processed1, X_features1):
    for x in X_processed1:
        x_feature = hog(x, orientations=8, pixels_per_cell=(10, 10),
                        cells_per_block=(1, 1), visualize=False, channel_axis=None)
        X_features1.append(x_feature)


feature(X_processed, X_features)
feature(X_test_processed, X_test)
feature(X_meta_processed, X_meta_features)
X_train = X_features + X_meta_features
y_train = y + y_meta

# T4 train model   start __________________________________________________________________________________

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_predict)
print(f'Accuracy: {accuracy}')

'''
#  debug
import cv2
import numpy as np
import sklearn
import glob
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from skimage.filters import gaussian
from scipy.ndimage import gaussian_filter
for i in range(43):
    path = "Dataset_2/Meta/" + str(i) + ".png"
    x = cv2.imread(path)
    blue_channel, green_channel, red_channel = cv2.split(x)

    # 计算每个颜色通道的总和
    blue_sum = np.sum(blue_channel)
    green_sum = np.sum(green_channel)
    red_sum = np.sum(red_channel)
    color_sum = blue_sum + green_sum + red_sum
    blue_sum = blue_sum / color_sum
    green_sum = green_sum / color_sum
    red_sum = red_sum / color_sum

    print(i , ":")
    print(f"Sum of Blue channel: {blue_sum}")
    print(f"Sum of Green channel: {green_sum}")
    print(f"Sum of Red channel: {red_sum}")
    print()
# h, w = x.shape[:2]
# min_dim = min(h, w)
# start_h = (h - min_dim) // 2
# start_w = (w - min_dim) // 2
# cropped_x = x[start_h:start_h+min_dim, start_w:start_w+min_dim]
#
# lab = cv2.cvtColor(cropped_x, cv2.COLOR_BGR2LAB)
# l, a, b = cv2.split(lab)
# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
# cl = clahe.apply(l)
# limg = cv2.merge((cl, a, b))
# temp_x = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
#
# temp_x = cv2.resize(temp_x, (48, 48))
# temp_x = cv2.cvtColor(temp_x,cv2.COLOR_BGR2GRAY)
# # temp_x = cv2.equalizeHist(temp_x)
# temp_x = gaussian_filter(temp_x, sigma=1)
# # temp_x = cv2.Canny((temp_x * 255).astype('uint8'), threshold1=100, threshold2=200)
# #cv2.imshow("", temp_x)
# cv2.imwrite('resaved.png', temp_x)
# # After add the edge detection, the accuracy is lower
'''

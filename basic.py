'''import cv2
import sklearn
import glob
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from skimage.filters import gaussian
from sklearn.ensemble import RandomForestClassifier

# T1  start _______________________________________________________________________________
# Read in Dataset

# change the dataset path here according to your folder structure
dataset_path = "images\\"

X = []
y = []
for i in glob.glob(dataset_path + '*.png', recursive=True):

    label = i.split("images")[1][1:4]
    y.append(label)
    t_x = cv2.imread(i)
    X.append(t_x)

    # write code to read ecah file i, and append it to list X

# you should have X, y with 5998 entries on each.
# T1 end ____________________________________________________________________________________


# T2 start __________________________________________________________________________________
# Preprocessing
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
    temp_x2 = cv2.cvtColor(temp_x,cv2.COLOR_BGR2GRAY)
    temp_x3 = gaussian(temp_x2, sigma=1)
    # After add the edge detection, the accuracy is lower
    # edges = cv2.Canny((temp_x3 * 255).astype('uint8'), threshold1=100, threshold2=200)
    # Append the converted image into X_processed
    X_processed.append(temp_x3)

# T2 end ____________________________________________________________________________________

# T3 start __________________________________________________________________________________
# Feature extraction
X_features = []
for x in X_processed:
    x_feature = hog(x, orientations=8, pixels_per_cell=(10, 10),
                    cells_per_block=(1, 1), visualize=False, channel_axis=None)
    X_features.append(x_feature)

# write code to Split training & testing sets using sklearn.model_selection.train_test_split
x_train, x_test, y_train , y_test = train_test_split(X_features, y, test_size=0.2,random_state = 42,shuffle= True)
#T3 end ____________________________________________________________________________________



#T4 start __________________________________________________________________________________
# train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train,y_train)

y_predict = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
print(f'Accuracy: {accuracy}')
'''


'''import cv2
import glob
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.filters import gaussian
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# T1  start _______________________________________________________________________________
# Read in Dataset

# change the dataset path here according to your folder structure
dataset_path = "images\\"

X = []
y = []
file_names = []
for i in glob.glob(dataset_path + '*.png', recursive=True):
    label = i.split("images")[1][1:4]
    y.append(label)
    t_x = cv2.imread(i)
    X.append(t_x)
    file_names.append(i)  # 记录文件名称

# you should have X, y with 5998 entries on each.
# T1 end ____________________________________________________________________________________


# T2 start __________________________________________________________________________________
# Preprocessing
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

# T2 end ____________________________________________________________________________________

# T3 start __________________________________________________________________________________
# Feature extraction
X_features = []
for x in X_processed:
    x_feature = hog(x, orientations=8, pixels_per_cell=(10, 10),
                    cells_per_block=(1, 1), visualize=False, channel_axis=None)
    X_features.append(x_feature)

# write code to Split training & testing sets using sklearn.model_selection.train_test_split
x_train, x_test, y_train, y_test, file_train, file_test = train_test_split(X_features, y, file_names, test_size=0.2, random_state=42, shuffle=True)
# T3 end ____________________________________________________________________________________

# T4 start __________________________________________________________________________________
# train model
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

# T4 end ____________________________________________________________________________________
'''



'''import cv2
import glob
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.filters import gaussian
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# T1  start _______________________________________________________________________________
# Read in Dataset

# change the dataset path here according to your folder structure
dataset_path = "Dataset_2\\Train\\"

X = []
y = []
file_names = []
for i in glob.glob(dataset_path + '*.png', recursive=True):
    label = i.split("images")[1][1:4]
    y.append(label)
    t_x = cv2.imread(i)
    X.append(t_x)
    file_names.append(i)  # 记录文件名称

# you should have X, y with 5998 entries on each.
# T1 end ____________________________________________________________________________________


# T2 start __________________________________________________________________________________
# Preprocessing
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

# T2 end ____________________________________________________________________________________

# T3 start __________________________________________________________________________________
# Feature extraction
X_features = []
for x in X_processed:
    x_feature = hog(x, orientations=8, pixels_per_cell=(10, 10),
                    cells_per_block=(1, 1), visualize=False, channel_axis=None)
    X_features.append(x_feature)

# write code to Split training & testing sets using sklearn.model_selection.train_test_split
x_train, x_test, y_train, y_test, file_train, file_test = train_test_split(X_features, y, file_names, test_size=0.2, random_state=42, shuffle=True)
# T3 end ____________________________________________________________________________________

# T4 start __________________________________________________________________________________
# train model
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

# T4 end ____________________________________________________________________________________
'''
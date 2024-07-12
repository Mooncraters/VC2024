import os
import cv2

def load_images_from_folder(folder):
    '''
    for each image in the folder, load it and append to the list
    '''
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images

def load_train_images(base_path):
    '''
    load all the images from the folder by class
    input: base_path
    return: all_images, which is a list of list;
            all_images[i] is a list of images in the i-th class
            all_labels is a list of labels
    '''
    all_images = []
    all_labels = []
    for class_folder in sorted(os.listdir(base_path), key=int):
        all_labels.append(int(class_folder))
        class_folder_path = os.path.join(base_path, class_folder)
        if os.path.isdir(class_folder_path):
            class_images = load_images_from_folder(class_folder_path)
            all_images.append(class_images)
    return all_images, all_labels
def load_test_images():
    '''
    load the test images
    '''
    #to be modified
    return

if __name__ == '__main__':
    import constant
    # image folder path
    base_path = constant.PATH_TO_TRAIN_DATA

    # load all the images
    all_images,all_labels = load_train_images(base_path)

    # print the information of loading
    for i, class_images in enumerate(all_images):
        print(f"Class {i}: {len(class_images)} images")
        print(f"all_labels: {all_labels[i]}")

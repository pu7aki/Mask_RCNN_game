import os
import random
import sys
import matplotlib.pyplot as plt
import coco
import model as modellib
import skimage.io
import visualize

# %matplotlib inline

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = "mask_rcnn_coco.h5"

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
# config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

game = 0;
while game < 100:

    # Load a random image from the images folder
    file_names = next(os.walk(IMAGE_DIR))[2]
    image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    level = 0
    while level==0:
        level = input("The level you want (1/2/3):")
        if level=='1':
            level=1
        elif level =='2':
            level=2
        elif level =='3':
            level=3
        else:
            print("Wrong input!")
            level=0

    print("Close the window to continue")
    label_list,masked_image= visualize.display_instances(level,image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    #print(label_list)

    time = 1;
    while time < 10:
        if level == 1:
            classid = input("Input your guess (lower case):")
            if classid == label_list[0]:
                print('Well done!')
                plt.imshow(image)
                plt.axis('off')  # 关掉坐标轴为 off
                plt.show()
                time = 10
            else:
                print('One more time!')
                plt.imshow(masked_image)
                plt.axis('off')  # 关掉坐标轴为 off
                plt.show()
                time = time + 1
                if time == 9:
                    print('Last chance!')
        elif level == 2:
            check = 0
            first = second = 0
            while check == 0:
                if first == second == 1:
                    check = 1
                else:
                    plt.imshow(masked_image)
                    plt.axis('off')  # 关掉坐标轴为 off
                    plt.show()
                    classid = input("Input your guess (lower case):")
                    if classid == label_list[0]:
                        print('Well done!')
                        first = 1
                        if(label_list[0]==label_list[1]):
                            second=1
                    elif classid == label_list[1]:
                        print('Well done!')
                        second = 1
                        if (label_list[0] == label_list[1]):
                            first = 1
                    else:
                        print('One more time!')
                        time = time + 1
                        if time == 9:
                            print('Last chance!')
            plt.imshow(image)
            plt.axis('off')  # 关掉坐标轴为 off
            plt.show()
            time=10
        else:
            check = 0
            first = second = third = 0
            while check == 0:
                #print(first,second,third)
                if first == second == third == 1:
                    check = 1
                else:
                    plt.imshow(masked_image)
                    plt.axis('off')  # 关掉坐标轴为 off
                    plt.show()
                    classid = input("Input your guess (lower case):")
                    if classid == label_list[0]:
                        print('Well done!')
                        first = 1
                        if (label_list[0] == label_list[2] == label_list[1]):
                            second = third = 1
                        elif(label_list[0] == label_list[2]):
                            third = 1
                        elif(label_list[0] == label_list[1]):
                            second = 1
                    elif classid == label_list[1]:
                        print('Well done!')
                        second = 1
                        if (label_list[0] == label_list[2] == label_list[1]):
                            first = third = 1
                        elif(label_list[1] == label_list[0]):
                            first = 1
                        elif (label_list[1] == label_list[2]):
                            third = 1
                    elif classid == label_list[2]:
                        print('Well done!')
                        third = 1
                        if (label_list[0] == label_list[2] == label_list[1]):
                            second = first = 1
                        elif(label_list[2] == label_list[0]):
                            first = 1
                        elif (label_list[2] == label_list[1]):
                            second = 1
                    else:
                        print('One more time!')
                        time = time + 1
                        if time == 9:
                            print('Last chance!')
            plt.imshow(image)
            plt.axis('off')  # 关掉坐标轴为 off
            plt.show()
            time = 10


    over = input("Do you wanna play again (y/n):")
    if over == 'y':
        game = game + 1
    else:
        sys.exit()

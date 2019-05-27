import os
import shutil
crd = os.getcwd() + "/21_class_new/train/"
det = os.getcwd() + "/21_class_new/train_all_classes/"
print(crd)
for folder in sorted(os.listdir(crd)):
    count = 0
    for img in sorted(os.listdir(crd + folder)):
        print(img)
        name = folder + "_" + str(count) + ".jpg"
        shutil.copyfile(crd + folder + "/" + img, det + name)
        count += 1
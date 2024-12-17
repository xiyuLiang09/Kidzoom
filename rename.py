import os

def rename(path, label):
    # 后缀名，用于筛选特定文件
    suffix_1 = ".jpg"
    suffix_2 = ".JPG"
    i = 1;
    for file in os.listdir(path):
        if file.endswith(suffix_1) or file.endswith(suffix_2):
            if os.path.isfile(os.path.join(path, file)):
                new_name = file.replace(file, label + "_%d" % i + suffix_1)
                os.rename(os.path.join(path, file), os.path.join(path, new_name))
                i += 1


path = "D:\\JetBrains\\PyCharm_2023.1.1\\Project\\yolov7-main\\datasets\\window\\open\\images"
label = "open"
rename(path, label)
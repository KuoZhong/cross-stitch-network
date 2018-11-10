import os
import random
seed = 0
random.seed(seed)

class_names = ['back_pack',
 'bike',
 'bike_helmet',
 'bookcase',
 'bottle',
 'calculator',
 'desk_chair',
 'desk_lamp',
 'desktop_computer',
 'file_cabinet',
 'headphones',
 'keyboard',
 'laptop_computer',
 'letter_tray',
 'mobile_phone',
 'monitor',
 'mouse',
 'mug',
 'paper_notebook',
 'pen',
 'phone',
 'printer',
 'projector',
 'punchers',
 'ring_binder',
 'ruler',
 'scissors',
 'speaker',
 'stapler',
 'tape_dispenser',
 'trash_can']

def train_test_split(img_path, link_file_path, portion=70):
    train = []
    test = []
    dirs = os.listdir(img_path)
    for m_dir in dirs:
        cur_path = os.path.join(img_path, m_dir)
        class_index = class_names.index(m_dir)
        files = os.listdir(cur_path)
        files = sorted(files)
        file_num_per_class = len(files)
        random.shuffle(files)
        train.extend([os.path.join(cur_path, file)+ ' '+str(class_index)+'\n' for file in files[:file_num_per_class*portion//100]])
        test.extend([os.path.join(cur_path, file)+ ' '+str(class_index)+'\n' for file in files[file_num_per_class * portion // 100:]])
    f_train = open(os.path.join(link_file_path, 'train.txt'), 'w')
    f_test = open(os.path.join(link_file_path, 'test.txt'), 'w')
    f_train.writelines(train)
    f_test.writelines(test)
    f_train.close()
    f_test.close()

# test_path = '/home/neon/dataset/office/amazon/images'
# link_path = '/home/neon/dataset/office/amazon/'
# train_test_split(test_path, link_file_path=link_path)
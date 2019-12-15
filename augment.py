from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import backend as K
import os
 
#K.set_image_dim_ordering('th')
 
path = 'C:/logo/data/image/'

# augmentation을 진행할 설정값 설정
augmentation = ImageDataGenerator(
        rotation_range = 30,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        vertical_flip = False,
        fill_mode='nearest')
# 최대 30도 회전, 상하 이동은 최대 1%
 
i = 0
#directory = 'The_directory_of_each_class/'
files = os.listdir(path)
 
for file in files :
    # 이미지를 불러와서 numpy array (1, 3, 128, 128)로 reshaping 해준다.
    # (batch_size, color_space, height, width)
    img = load_img(path + file)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    # flow(data.x_train, data.y_train, batch_size=batch_size) 반복자는
    # x, y를 가지고 한번에 batch_size만큼의 랜덤하게 변형된 학습 데이터를 만들어줌
    for batch in augmentation.flow(x, batch_size = 1,
                              save_to_dir = 'imageGenerate',
                              save_prefix = '%d'%i,
                              save_format = 'jpg'):
        i+=1
        
        if i%20 == 0:
            break 

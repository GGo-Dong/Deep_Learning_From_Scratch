import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img)) # numpy array로 표현된 이미지를 PIL 전용 객체로 변환함
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0] # 테스트로 출력해 볼 이미지
label = t_train[0] # 테스트로 출력해 볼 레이블
print(label) # 5

print(img.shape)
img = img.reshape(28, 28) # flatten으로 인해 1차원으로 변한 이미지를 원래 모양인 28x28 로 되돌림
print(img.shape)

img_show(img)
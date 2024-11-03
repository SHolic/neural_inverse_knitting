import tensorflow as tf
import numpy as np
import scipy
from skimage.util import crop as imcrop
from PIL import Image
import pdb

palette = np.array([
    # front
    [255, 0, 16],  # FK
    [43, 206, 72],  # BK
    [255, 255, 128],  # T
    [0, 255, 127],  # H
    [94, 241, 242],  # M
    [34, 139, 34],  # E
    [0, 191, 255],  # V
    [0, 129, 69],  # V(R)
    [255, 0, 190],  # V(L)
    [255, 164, 4],  # X(R)
    [0, 117, 220],  # X(L)
    [255, 109, 21],  # O(5)
    [122, 0, 136],  # Y
    [162, 0, 101]  # FO(2)
    # # complete
    # [255, 0, 16],       # FK
    # [30, 206, 30],      # BK
    # [255, 255, 128],    # T
    # [0, 255, 127],      # H|M
    # [130, 210, 210],    # M
    # [34, 139, 34],      # E|V(L)
    # [0, 191, 255],      # V|HM
    # [0, 129, 69],       # V(R)
    # [255, 0, 190],      # V(L)
    # [255, 164, 4],      # X(R)
    # [0, 117, 220],      # X(L)
    # [117, 59, 59],      # S
    # [179, 179, 179],    # T(F)
    # [255, 215, 0],      # V|M
    # [255, 105, 180],    # T(B)
    # [160, 32, 240],     # M|H(B)
    # [139, 69, 19],      # E|V(R)
    # [0, 164, 255],      # V|FK
    # [255, 30, 30],      # FK, MAK
    # [230, 230, 110],    # FT, FKMAK
    # [220, 200, 100],    # FT, MBK
    # [100, 230, 230],    # M, BK
    # [110, 220, 220],    # M, FK
    # [20, 200, 255],     # V, BK
    # [10, 140, 80],      # VR, FKMAK
    # [10, 250, 110],     # H, BK
    # [240, 20, 170],     # VL, FKMAK
    # [240, 100, 30],     # AO(2)
    # [250, 110, 40],     # O(5), AK
    # [200, 100, 70],     # O(5), FKBK
    # [220, 80, 60],      # BO(2)
    # [230, 90, 50],      # O(5), BK
    # [130, 10, 120],     # Y, MATBK
    # [170, 10, 90]       # FO(2)

])

mirror_mapping = np.array([
    0, 1, 2, 3, # KPTM
    6, 7,   # FR -> FL
    4, 5,   # FL -> FR
    10, 11, # BR -> BL
    8, 9,   # BL -> BR
    14, 15, # XR -> XL
    12, 13, # XL -> XR
    16      # S
]).astype(np.int32)

def tf_ind_to_rgb(t_ind):
    # HSV palette
    # t_rgb = tf.image.hsv_to_rgb(
    #     tf.concat(
    #         axis=3,
    #         values=[
    #             tf.cast(t_ind, dtype=tf.float32) / 34.,
    #             tf.ones(tf.shape(t_ind)),
    #             tf.ones(tf.shape(t_ind))
    #         ]))
    t_rgb = tf.cast(tf.concat(axis = 3,
        values = [
            tf.gather(palette[:, 0], t_ind),
            tf.gather(palette[:, 1], t_ind),
            tf.gather(palette[:, 2], t_ind)
        ]), dtype = tf.uint8)
    return t_rgb

def tf_mirror_image(t_img):
    return tf.image.flip_left_right(t_img)

def tf_mirror_instr(t_inst):
    return tf_mirror_image(tf.gather(mirror_mapping, t_inst))

def tf_mirror(t_img, t_inst):
    t_img = tf_mirror_image(t_img)
    t_inst = tf_mirror_instr(t_inst)
    return t_img, t_inst

def save_instr(fname, img):
    img = img[:,:,0].astype(np.uint8)
    img = Image.fromarray(img, mode = 'P')
    img.putpalette([
        # front
        255, 0, 16,  # FK
        43, 206, 72,  # BK
        255, 255, 128,  # T
        0, 255, 127,  # H
        94, 241, 242,  # M
        34, 139, 34,  # E
        0, 191, 255,  # V
        0, 129, 69,  # V(R)
        255, 0, 190,  # V(L)
        255, 164, 4,  # X(R)
        0, 117, 220,  # X(L)
        255, 109, 21,  # O(5)
        122, 0, 136,  # Y
        162, 0, 101  # FO(2)
        # # complete
        # 255, 0, 16,       # FK
        # 30, 206, 30,      # BK
        # 255, 255, 128,    # T
        # 0, 255, 127,      # H|M
        # 130, 210, 210,    # M
        # 34, 139, 34,      # E|V(L)
        # 0, 191, 255,      # V|HM
        # 0, 129, 69,       # V(R)
        # 255, 0, 190,      # V(L)
        # 255, 164, 4,      # X(R)
        # 0, 117, 220,      # X(L)
        # 117, 59, 59,      # S
        # 179, 179, 179,    # T(F)
        # 255, 215, 0,      # V|M
        # 255, 105, 180,    # T(B)
        # 160, 32, 240,     # M|H(B)
        # 139, 69, 19,      # E|V(R)
        # 0, 164, 255,      # V|FK
        # 255, 30, 30,      # FK, MAK
        # 230, 230, 110,    # FT, FKMAK
        # 220, 200, 100,    # FT, MBK
        # 100, 230, 230,    # M, BK
        # 110, 220, 220,    # M, FK
        # 20, 200, 255,     # V, BK
        # 10, 140, 80,      # VR, FKMAK
        # 10, 250, 110,     # H, BK
        # 240, 20, 170,     # VL, FKMAK
        # 240, 100, 30,     # AO(2)
        # 250, 110, 40,     # O(5), AK
        # 200, 100, 70,     # O(5), FKBK
        # 220, 80, 60,      # BO(2)
        # 230, 90, 50,      # O(5), BK
        # 130, 10, 120,     # Y, MATBK
        # 170, 10, 90       # FO(2)

    ])
    img.save(fname)

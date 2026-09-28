import pickle
import json
import cv2
import numpy as np
import seaborn as sns
import bz2
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import io
import datetime
import nibabel as nib
import os

def dump_pickle_file(file, path):
    r"""Save an object as a pickle file."""
    with open(path, 'wb') as output_file:
        pickle.dump(file, output_file)

def load_pickle_file(path):
    r"""Load a pickle file."""
    with open(path, 'rb') as input_file:
        file = pickle.load(input_file)
    return file

def dump_compressed_pickle_file(file, path):
    r"""Save an object as a compressed pickle file."""
    with bz2.BZ2File(path, 'w') as output_file:
        pickle.dump(file, output_file)

def load_compressed_pickle_file(path):
    r"""Load a compressed pickle file."""
    with bz2.BZ2File(path, 'rb') as input_file:
        file = pickle.load(input_file)
    return file
    
def dump_json_file(file, path):
    r"""Save an object as JSON."""
    with open(path, 'w') as output_file:
        json.dump(file, output_file)

def load_json_file(path):
    r"""Load a JSON file."""
    with open(path, 'r') as input_file:
        file =  json.load(input_file)
    return file

def get_img_from_fig(fig, dpi=400, size = (1700, 1700)):
    r"""Convert a matplotlib figure to an image."""
    buf = io.BytesIO()

    size_inch = fig.get_size_inches()
    size_inch = size / size_inch
    dpi = int(min(size_inch))
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img = buf
    return img.read()

def visualize_perceptive_range(img, cell_fire_rate=0.5):
    r"""Visualise the current perceptive range."""
    if np.max(img) == 0:
        img[int(img.shape[0] / 2), int(img.shape[1] / 2), :] = 1
    else:
        img = img[:, :, 0]

        x_roll = np.roll(img, 1, axis= 0) + np.roll(img, -1, axis= 0)
        y_roll = np.roll(x_roll, 1, axis= 1) + np.roll(x_roll, -1, axis= 1)
        img_new = (img + np.clip(x_roll + y_roll, 0, 1)) 

        random_array = np.random.rand(img.shape[0], img.shape[1])
        img_new[random_array < cell_fire_rate] = 0
        img_new = img + img_new

        img = np.dstack((img_new, img_new, img_new))

    return img


def visualize_all_channels_fast(img, replace_firstImage = None, min=1, max=100, labels = None):
    r"""Visualise all cell channels (simple view)."""
    if img.shape[0] == 1:
        img = img[0]
    if labels is not None and labels.shape[0] == 1:
        labels = labels[0]

    tiles = int(math.ceil(math.sqrt(img.shape[2])))
    img_x = img.shape[0]
    img_y = img.shape[1]

    img_all_channels = np.zeros((img_x*tiles, img_y*tiles))
    for tile_pos in range(img.shape[2]):
        tile = img[:,:,tile_pos]
        x = tile_pos%tiles
        y = int(math.floor(tile_pos/tiles))
        if tile_pos < 3:
            tile = tile

        img_all_channels[x*img_x:(x+1)*img_x, y*img_y:(y+1)*img_y] = tile

        tile_pos_lab = tile_pos -3
        if labels is not None and labels.shape[2] > tile_pos_lab and tile_pos_lab > 0:
            tile_label = labels[:,:,tile_pos_lab]

            gx_m1, gy_m1 = np.gradient(tile_label)
            tile_label = gy_m1 * gy_m1 + gx_m1 * gx_m1
            tile_label[tile_label != 0.0] = 1
            img_all_channels[x*img_x:(x+1)*img_x, y*img_y:(y+1)*img_y][tile_label == 1] = 1000

    img_all_channels_blue = img_all_channels.copy()
    img_all_channels_blue[img_all_channels_blue!=0] = 0

    img_all_channels_red = img_all_channels.copy()
    img_all_channels_red[img_all_channels_red > 0] = 0

    img_all_channels_green = img_all_channels.copy()
    img_all_channels_green[img_all_channels_green < 0] = 0


    img_all_channels_red = img_all_channels_red * -1
    img_all_channels_red[img_all_channels_red <= min] = (img_all_channels_red[img_all_channels_red <= min] / min) * 0.5
    img_all_channels_red[img_all_channels_red > min] = np.log(img_all_channels_red[img_all_channels_red > min]) / np.log(max) + 0.5

    img_all_channels_green[img_all_channels_green <= min] = (img_all_channels_green[img_all_channels_green <= min] / min) * 0.5
    img_all_channels_green[img_all_channels_green > min] = np.log(img_all_channels_green[img_all_channels_green > min]) / np.log(max) + 0.5


    img_all_channels = np.stack([img_all_channels_blue, img_all_channels_green, img_all_channels_red], axis=2)

    max = np.max(img_all_channels)   
    min = np.min(img_all_channels)

    if replace_firstImage is not None:
        img_all_channels[0:img_x, 0:img_y, :] = replace_firstImage
 
    return img_all_channels



def visualize_all_channels(img, replace_firstImage = None, divide_by=3, labels = None, color_map="nipy_spectral", size = (1700, 1700)):
    r"""Visualise all cell channels (detailed, slower view)."""
    if img.shape[0] == 1:
        img = img[0]
    if labels is not None and labels.shape[0] == 1:
        labels = labels[0]

    tiles = int(math.ceil(math.sqrt(img.shape[2])))
    img_x = img.shape[0]
    img_y = img.shape[1]

    print("MEASURE TIME")
    time_a = datetime.datetime.now()

    img_all_channels = np.zeros((img_x*tiles, img_y*tiles))
    for tile_pos in range(img.shape[2]):
        tile = img[:,:,tile_pos]
        x = tile_pos%tiles
        y = int(math.floor(tile_pos/tiles))
        if tile_pos < 3:
            tile = tile

        img_all_channels[x*img_x:(x+1)*img_x, y*img_y:(y+1)*img_y] = tile

        tile_pos_lab = tile_pos -3
        if labels is not None and labels.shape[2] > tile_pos_lab and tile_pos_lab > 0:
            tile_label = labels[:,:,tile_pos_lab]

            gx_m1, gy_m1 = np.gradient(tile_label)
            tile_label = gy_m1 * gy_m1 + gx_m1 * gx_m1
            tile_label[tile_label != 0.0] = 1
            img_all_channels[x*img_x:(x+1)*img_x, y*img_y:(y+1)*img_y][tile_label == 1] = 1000

    time_b = datetime.datetime.now()
    print((time_b - time_a).microseconds)
    
    if np.min(size) != 0:
        figsize_def = (10, int(10*size[1]/size[0]))
        print(figsize_def)
    else: 
        figsize_Def = (2,1)
    fig, axes = plt.subplots(figsize=figsize_def)
    pos = axes.imshow(img_all_channels, norm=colors.SymLogNorm(linthresh=0.3, linscale=0.3,
                                              vmin=-10.0, vmax=10.0), cmap=color_map)
    
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad = 0.05)

    axes.margins(x= 0, y=0)

    fig.colorbar(pos, cax=cax)
    
    fig.canvas.draw()

    time_c = datetime.datetime.now()

    print((time_c - time_b).microseconds)
    print(fig)

    return fig 

def convert_image(img, prediction, label=None, encode_image=True):
    r"""Combine an image and optional label into one displayable image."""
    img_rgb = img 
    img_rgb = img_rgb - np.amin(img_rgb)
    img_rgb = img_rgb * img_rgb 
    img_rgb = img_rgb / np.amax(img_rgb)
    label_pred = prediction

    img_rgb, label, label_pred = [orderArray(v.squeeze()) for v in [img_rgb, label, label_pred]]

    
    label = np.amax(label, axis=-1)
    label_pred = np.amax(label_pred, axis=-1)
    label_pred = np.stack((label_pred, label_pred, label_pred), axis=-1)
    

    # Overlay the label on the image.
    if label is not None:
        sobel_x = cv2.Sobel(src=label, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
        sobel_y = cv2.Sobel(src=label, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
        sobel = sobel_x + sobel_y
        if len(sobel.shape) < 3:
            sobel = np.stack((sobel, sobel, sobel), axis=-1)

        sobel[:,:,2] = sobel[:,:,0]
        sobel[:,:,0] = 0
        sobel = np.abs(sobel)
        img_rgb[img_rgb < 0] = 0
        label_pred[label_pred < 0] = 0

        sobel = cv2.resize(sobel, dsize=(label_pred.shape[0], label_pred.shape[1])) 
        img_rgb = cv2.resize(img_rgb, dsize=(label_pred.shape[0], label_pred.shape[1]), interpolation=cv2.INTER_NEAREST) 

        img_rgb = np.clip((sobel  * 0.8 + img_rgb + 0.5 * label_pred), 0, 1)

    if sum(img_rgb.shape) > 2000:
        size = (int(img_rgb.shape[0]/6), int(img_rgb.shape[1]/6))
        img_rgb = cv2.resize(img_rgb, dsize=size, interpolation=cv2.INTER_CUBIC) 

    if encode_image:
        img_rgb = encode(img_rgb)
    return img_rgb 

def orderArray(array):

    if len(array.shape) < 3:
        array = np.stack((array, array, array), axis=-1)

    if array.shape[0] < array.shape[2]:
        return np.transpose(array, (1, 2, 0))
    if array.shape[1] < array.shape[2]:
        return np.transpose(array, (0, 2, 1))
    else:
         return array


def encode(img_rgb, size=(150, 100)):
    r"""Encode an image."""
    size_img = img_rgb.shape
    size_img = [1, size_img[0]/ size_img[1]]

    size_img_scaledX = [int(x * size[0] * 0.95) for x in size_img] 
    size_img_scaledY = [int(x * size[1] * 0.95) for x in size_img] 

    scale = (10, 10)

    for s in [size_img_scaledX, size_img_scaledY]:
        if s[0] <= size[0] and s[1] <= size[1] and s[0] > scale[0]:
            scale = s

    img_rgb = img_rgb * 255
    img_rgb[img_rgb > 255] = 255
    factor_y = img_rgb.shape[0] / img_rgb.shape[1] 
    img_rgb = cv2.resize(img_rgb, dsize=scale, interpolation=cv2.INTER_NEAREST)
    img_rgb = cv2.imencode(".png", img_rgb)[1].tobytes()
    return img_rgb

def saveNiiGz(output, label, patient_id, path):
    r"""Save a NIfTI file."""
    output = np.round(output.cpu().detach().numpy())
    output[output < 0] = 0
    output[output > 0] = 1
    nib_image = nib.Nifti1Image(output, np.eye(4))
    nib_label = nib.Nifti1Image(label.cpu().detach().numpy(), np.eye(4))
    nib.save(nib_image, os.path.join(path, patient_id + "_image.nii.gz"))  
    nib.save(nib_label, os.path.join(path, patient_id + "_label.nii.gz"))  
    

# Plot individual patient scores.
def loss_log_to_image(loss_log):
    sns.scatterplot(data=loss_log, x="id", y="Dice")

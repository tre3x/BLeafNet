import numpy as np
import cv2
import os
import glob
import random

def get_input(path):
    im = cv2.imread(path)
    return(im)

def get_files(path, ext):
    files = []
    label_files= []
    for x in os.walk(path):
        for y in glob.glob(os.path.join(x[0], '*.{}'.format(ext))):
            files.append(y)
    label_files = os.listdir(path)
    label_files = sorted(label_files)
    return files, label_files

def get_output(path, label_file):
    img_id = path.split('/')[-2]
    laba = []
    for label in label_file:
      if label == img_id:
        laba.append(1)
      else:
        laba.append(0)
    return laba

#RGB MODEL DATA GENERATOR
def rgbimage_generator(files, label_files, batch_size, dim):

      while True:
          batch_paths  = np.random.choice(a  = files, 
                                          size = batch_size)
          batch_input_rgb = []
          batch_output = [] 
          
          for input_path in batch_paths:
              input = get_input(input_path)
              input = cv2.resize(input, dim)
              output = get_output(input_path, label_files)
              batch_input_rgb.append(input)
              batch_output.append(output)
          batch_input_rgb = np.array(batch_input_rgb)
          batch_y = np.array(batch_output)
          yield batch_input_rgb, batch_y

#BW Model Data Generator
def bwimage_generator(files, label_files, batch_size, dim):

      while True:
          batch_paths  = np.random.choice(a  = files, 
                                          size = batch_size)
          batch_input_bw = [] 
          batch_output = [] 
          
          for input_path in batch_paths:
              input = get_input(input_path)
              input = cv2.resize(input, dim)
              #GrayScale
              img_gray = cv2.cvtColor(np.asarray(input).astype('uint8'), cv2.COLOR_BGR2GRAY)
              img_gray = np.stack((img_gray,)*3, axis=-1)
              output = get_output(input_path, label_files)
              batch_input_bw.append(img_gray)
              batch_output.append(output)
          batch_input_bw = np.array(batch_input_bw)
          batch_y = np.array(batch_output)
          yield batch_input_bw, batch_y

#R Model Data Generator
def rimage_generator(files, label_files, batch_size, dim):

      while True:
          batch_paths  = np.random.choice(a  = files, 
                                          size = batch_size)
          batch_input_r = []
          batch_output = [] 
          
          for input_path in batch_paths:
              input = get_input(input_path)
              input = cv2.resize(input, dim)
              im_red = np.empty_like(input)
              im_red[:] = input
              #R
              im_red[:, :, 0] = 0
              im_red[:, :, 1] = 0
              output = get_output(input_path, label_files)
              batch_input_r.append(im_red)
              batch_output.append(output)
          batch_input_r = np.array(batch_input_r)
          output = np.array(output)
          batch_y = np.array(batch_output)
          yield batch_input_r, batch_y

#G Model Data Generator
def gimage_generator(files, label_files, batch_size, dim):

      while True:
          batch_paths  = np.random.choice(a  = files, 
                                          size = batch_size)
          batch_input_g = []
          batch_output = [] 
          
          for input_path in batch_paths:
              input = get_input(input_path)
              input = cv2.resize(input, dim)
              im_green = np.empty_like(input)
              im_green[:] = input
              #G
              im_green[:, :, 0] = 0
              im_green[:, :, 2] = 0
              output = get_output(input_path, label_files)
              batch_input_g.append(im_green)
              batch_output.append(output)
          batch_input_g = np.array(batch_input_g)
          batch_y = np.array(batch_output)
          yield batch_input_g, batch_y

#B Model Data Generator
def bimage_generator(files, label_files, batch_size, dim):

      while True:
          batch_paths  = np.random.choice(a  = files, 
                                          size = batch_size)
          batch_input_b = []
          batch_output = [] 
          
          for input_path in batch_paths:
              input = get_input(input_path)
              input = cv2.resize(input, dim)
              im_blue = np.empty_like(input)
              im_blue[:] = input
              #B
              im_blue[:, :, 1] = 0
              im_blue[:, :, 2] = 0
              output = get_output(input_path, label_files)
              batch_input_b.append(im_blue)
              batch_output.append(output)
          batch_input_b = np.array(batch_input_b)
          batch_y = np.array(batch_output)
          yield batch_input_b, batch_y

#Ensembled Model Data Generator
def image_generator(files, label_files, batch_size, dim):

      while True:
          random.shuffle(files)
          batch_paths  = np.random.choice(a  = files, 
                                          size = batch_size)
          batch_input_rgb = []
          batch_input_bw = []
          batch_input_r = []
          batch_input_g = []
          batch_input_b = []
          batch_output = [] 
          
          for input_path in batch_paths:
              input = get_input(input_path)
              input = cv2.resize(input, dim)
              im_red = np.empty_like(input)
              im_red[:] = input

              im_blue = np.empty_like(input)
              im_blue[:] = input

              im_green = np.empty_like(input)
              im_green[:] = input
              #GrayScale
              img_gray = cv2.cvtColor(np.asarray(input).astype('uint8'), cv2.COLOR_BGR2GRAY)
              img_gray = np.stack((img_gray,)*3, axis=-1)
              #G
              im_green[:, :, 0] = 0
              im_green[:, :, 2] = 0
              #R
              im_red[:, :, 0] = 0
              im_red[:, :, 1] = 0
              #B
              im_blue[:, :, 1] = 0
              im_blue[:, :, 2] = 0
              output = get_output(input_path, label_files)
              batch_input_rgb.append(input)
              batch_input_bw.append(img_gray)
              batch_input_r.append(im_red)
              batch_input_b.append(im_blue)
              batch_input_g.append(im_green)
              batch_output.append(output)
          batch_input_rgb = np.array(batch_input_rgb)
          batch_input_bw = np.array(batch_input_bw)
          batch_input_b = np.array(batch_input_b)
          batch_input_g = np.array(batch_input_g)
          batch_input_r = np.array(batch_input_r)
          batch_y = np.array(batch_output)
          yield [batch_input_rgb, batch_input_bw, batch_input_r, batch_input_g, batch_input_b], batch_y
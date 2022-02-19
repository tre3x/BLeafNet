import os
import cv2
import numpy as np
import tensorflow as tf
import utils
import model

class train():
    def __init__(self, train_path, val_path):
        self.train_path = train_path
        self.val_path = val_path
        here = os.path.dirname(os.path.abspath(__file__))
        self.paths = [os.path.join(here, "models", "RGB"), os.path.join(here, "models", "GRAY"), os.path.join(here, "models", "R"), 
                        os.path.join(here, "models", "G"), os.path.join(here, "models", "B"), os.path.join(here, "models", "fused")]
    
    def get_files(self):
        self.train_files, self.label_files = utils.get_files(self.train_path, 'jpg')
        self.val_files, self.label_files = utils.get_files(self.val_path, 'jpg')

    def train(self, model, path, epochs, batch_size, steps, dim, tag):
        model.compile('Adam', loss = tf.keras.losses.CategoricalCrossentropy(), metrics = ['accuracy'])
        
        checkpoint_filepath = path
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, 
                                                                        save_weights_only=False, monitor='val_accuracy', mode='max')

        if tag=='RGB':
            model.fit(utils.rgbimage_generator(self.train_files, self.label_files, batch_size, dim), epochs = epochs, steps_per_epoch = steps,
                        validation_data =utils.rgbimage_generator(self.val_files, self.label_files, batch_size, dim), 
                        validation_steps=15, callbacks=[model_checkpoint_callback])
        if tag=='GRAY':
            model.fit(utils.bwimage_generator(self.train_files, self.label_files, batch_size, dim), epochs = epochs, steps_per_epoch = steps,
                        validation_data =utils.bwimage_generator(self.val_files, self.label_files, batch_size, dim), 
                        validation_steps=15, callbacks=[model_checkpoint_callback])
        if tag=='R':
            model.fit(utils.rimage_generator(self.train_files, self.label_files, batch_size, dim), epochs = epochs, steps_per_epoch = steps,
                        validation_data =utils.rimage_generator(self.val_files, self.label_files, batch_size, dim), 
                        validation_steps=15, callbacks=[model_checkpoint_callback])
        if tag=='G':
            model.fit(utils.gimage_generator(self.train_files, self.label_files, batch_size, dim), epochs = epochs, steps_per_epoch = steps,
                        validation_data =utils.gimage_generator(self.val_files, self.label_files, batch_size, dim), 
                        validation_steps=15, callbacks=[model_checkpoint_callback])
        if tag=='B':
            model.fit(utils.bimage_generator(self.train_files, self.label_files, batch_size, dim), epochs = epochs, steps_per_epoch = steps,
                        validation_data =utils.bimage_generator(self.val_files, self.label_files, batch_size, dim), 
                        validation_steps=15, callbacks=[model_checkpoint_callback])
        if tag=='all':
            model.fit(utils.image_generator(self.train_files, self.label_files, batch_size, dim), epochs = epochs, steps_per_epoch = steps,
                        validation_data =utils.image_generator(self.val_files, self.label_files, batch_size, dim), 
                        validation_steps=15, callbacks=[model_checkpoint_callback])
    
    def phase_1(self, classes, epochs, batch_size, steps, dim):
        print("************TRAINING RGB CHANNEL BASE MODEL************")
        mod = model.base_cnn(classes)
        self.train(mod, self.paths[0], epochs, batch_size, steps, dim,  'RGB')
        print("************TRAINING GRAYSCALE CHANNEL BASE MODEL************")
        mod = model.base_cnn(classes)
        self.train(mod, self.paths[1], epochs, batch_size, steps, dim, 'GRAY')
        print("************TRAINING RED CHANNEL BASE MODEL************")
        mod = model.base_cnn(classes)
        self.train(mod, self.paths[2], epochs, batch_size, steps, dim, 'R')
        print("************TRAINING GREEN CHANNEL BASE MODEL************")
        mod = model.base_cnn(classes)
        self.train(mod, self.paths[3], epochs, batch_size, steps, dim, 'G')
        print("************TRAINING BLUE CHANNEL BASE MODEL************")
        mod = model.base_cnn(classes)
        self.train(mod, self.paths[4], epochs, batch_size, steps, dim, 'B')

    def phase_2(self, classes, epochs, batch_size, steps, dim):
        print("************TRAINING FUSION BASED MODEL************")
        mod = model.fin_model(classes, 5, self.paths[:-1], True)
        self.train(mod, self.paths[5], epochs, batch_size, steps, dim, 'all')
    
    def run(self, epochs_base, epochs_fin, batch_size, steps, dim=(224, 224)):
        self.get_files()
        self.phase_1(len(self.label_files), epochs_base, batch_size, steps, dim)
        self.phase_2(len(self.label_files), epochs_fin, batch_size, steps, dim)
import cv2
import numpy as np
import tensorflow as tf

class TopLayer(tf.keras.layers.Layer):
    def __init__(self, classes):
        super(TopLayer, self).__init__()
        self.classes = classes

    def build(self, input_shape):
        self.avgPool = tf.keras.layers.AveragePooling2D(7,7)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1000, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.75)
        self.classify = tf.keras.layers.Dense(self.classes, activation='softmax')

    def call(self, inputs):
        top = self.avgPool(inputs)
        top = self.flatten(top)
        top = self.dense(top)
        top = self.dropout(top)
        top = self.classify(top)
        return top

class BonferroniMeanOperator(tf.keras.layers.Layer):
    def __init__(self, classes, num_models):
        super(BonferroniMeanOperator, self).__init__()
        self.classes = classes
        self.num_models = num_models
        self.arr1 = np.full((1, classes), (1/(num_models-1)))
        self.arr2 = np.full((1, classes), (1/num_models))
    
    def build(self, input_shape):
        self.add = tf.keras.layers.add
        self.multiply = tf.keras.layers.multiply

    def call(self, input):
        elems = []
        temp = []
        for outer in range(self.num_models):
            for inner in range(self.num_models):
                if(outer==inner):
                    continue
                temp.append(input[inner])
            elems.append(self.multiply([self.multiply([self.add(temp),self.arr1]), input[outer]]))
        
        fin_layer = self.multiply([self.add(elems), self.arr2])
        return fin_layer

def rename_layers(models, i=1):
    for model in models:
        for layer in model.layers:
            layer._name = layer._name + '_'*i
        i=i+1

def base_cnn(classes):
    base = tf.keras.applications.resnet50.ResNet50(weights = 'imagenet', include_top= False, input_shape=(224, 224, 3))
    out = TopLayer(classes)(base.output)
    mod = tf.keras.models.Model(inputs = base.input, outputs = out)
    return mod

def fin_model(classes, num_model, paths, prevTrain=False):
    models = []
    scores = []
    input = []
    if not prevTrain:
        for num in range(num_model):
            models.append(base_cnn(classes))
            scores.append(models[-1].output)
            input.append(models[-1].input)
        scores = BonferroniMeanOperator(classes, num_model)(scores)
        rename_layers(models)
        mod = tf.keras.models.Model(inputs = input, outputs = scores) 
    else:
        for path in paths:
            models.append(tf.keras.models.load_model(path))
            scores.append(models[-1].output)
            input.append(models[-1].input)
        scores = BonferroniMeanOperator(classes, num_model)(scores)
        rename_layers(models)
        mod = tf.keras.models.Model(inputs = input, outputs = scores)  
    return mod  

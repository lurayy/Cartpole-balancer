import tensorflow as tf 
import random
import numpy as np 
from memory import Memory

class DeepQ:
    def __init__(self, inputs, outputs, memorySize, discountFactor, learningRate, learnStart):
        self.input_size = inputs
        self.output_size = outputs
        self.memory = Memory(memorySize)
        self.discountFactor = discountFactor
        self.learnStart = learnStart
        self.learningRate = learningRate
    
    def initNetwork(self, hiddenLayers):
        model = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate)
        self.model = model
        targetModel = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate)
        self.targetModel = targetModel

    def createModel(self, inputs, outputs, hiddenLayers, activationType, learningRate):
        bias = True
        dropout = 0
        regularizationFactor = 0.01
        model = tf.keras.models.Sequential()
        if len(hiddenLayers) == 0:
            model.add(tf.keras.layers.Dense(self.input_size, input_shpae =(self.input_size,), kernel_initializer = 'lecun_uniform', use_bias = bias))
            model.add(tf.keras.layers.Activation("linear"))
        else:
            model.add(tf.keras.layers.Dense(hiddenLayers[0], input_shape=(self.input_size,), kernel_initializer = 'lecun_uniform', kernel_regularizer = tf.keras.regularizers.l2(l = regularizationFactor), use_bias = bias))
            if (activationType == "LeakyReLU"):
                model.add(tf.keras.layers.LeakyReLU(alpha = 0.01))
            else:
                model.add(tf.keras.layers.Activation(activationType))

            for index in range(1, len(hiddenLayers)-1):
                layerSize = hiddenLayers[index]
                model.add(tf.keras.layers.Dense(layerSize, kernel_initializer= 'lecun_uniform', kernel_regularizer = tf.keras.regularizers.l2(l = regularizationFactor), use_bias = bias))
                if dropout > 0:
                    model.add(tf.keras.layers.Dropout(dropout))
                if (activationType == "LeakyReLU"):
                    model.add(tf.keras.layers.LeakyReLU(aplha = 0.01))
                else:
                    model.add(tf.keras.layers.Activation(activationType))
            model.add(tf.keras.layers.Dense(self.output_size, kernel_initializer='lecun_uniform', use_bias=bias))
            model.add(tf.keras.layers.Activation("linear"))
        optimizer = tf.keras.optimizers.RMSprop(lr=learningRate, rho = 0.9, epsilon = 1e-06)
        model.compile(loss="mse", optimizer = optimizer)
        return model

    def printNetwork(self):
        i = 0
        for layer in self.model.layers:
            weight = layer.get_weights()
            print("layer ",i," : ",weight)
            i+=1
    
    # copy current network to backup (traget) model
    def backupNetwork(self, model, backup):
        weightMatrix = []
        for layer in model.layers:
            weights = layer.get_weights()
            weightMatrix.append(weights)
        i = 0
        for layer in backup.layers:
            weights = weightMatrix[i]
            layer.set_weights(weights)
            i+=1
    
    def updateTargetNetwork(self):
        self.backupNetwork(self.model, self.targetModel)
    
    def getQValues(self, state):
        predicted = self.targetModel.predict(state.reshape(1, len(state)))
        return predicted[0]
    
    def getMaxQ(self, qValues):
        return np.max(qValues)
    
    def getMaxIndex(self, qValues):
        return np.argmax(qValues)
    
    #calculate the traget fucntion
    def calculateTarget(self, qValuesNewState, reward, isFinal):
        if isFinal:
            return reward
        else:
            return reward + self.discountFactor  * self.getMaxQ(qValuesNewState)
        
    #select the action with the highest Q value
    def selectAction(self, qValues, explorationRate):
        rand = random.random()
        if rand  < explorationRate: 
            action = np.random.randint(0 , self.output_size)
        else:
            action = self.getMaxIndex(qValues)
        return action

    def selectionActionByProbability(self, qValues, bias):
        qValueSum = 0 
        shiftBy = 0 
        for value in qValues:
            if value + shiftBy < 0:
                shiftBy = - (value + shiftBy)
            shiftBy += 1e-06
        
        for value in qValues:
            qValueSum += (value + shiftBy) ** bias
    
        probabilitySum = 0
        qValueProbabilities = []
        for value in qValues:
            probability = ((value + shiftBy) ** bias) / float(qValueSum)
            qValueProbabilities.append(probability + probabilitySum)
            probabilitySum += probability
        qValueProbabilities[len(qValueProbabilities) - 1] = 1 
        rand = random.random()
        i = 0 
        for value in qValueProbabilities:
            if(rand<=value):
                return i 
            i+=1
        
    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)
    
    def learnOnLastState(self):
        if self.memory.getCurrentSize() >=1:
            return self.memory.getMemory(self.memory.getCurrentSize() - 1)
            
    def learnOnMiniBatch(self, miniBatchSize):
        if self.memory.getCurrentSize() > self.learnStart :
            miniBatch = self.memory.getMiniBatch(miniBatchSize)
            X_batch = np.empty((0,self.input_size), dtype = np.float64)
            Y_batch = np.empty((0,self.output_size), dtype = np.float64)
            for sample in miniBatch:
                isFinal = sample['isFinal']
                state = sample['state']
                action = sample['action']
                reward = sample['reward']
                newState = sample['newState']

                qValues = self.getQValues(state)
                qValuesNewState = self.getTargetQValues(newState)
                targetValue = self.calculateTarget(qValuesNewState, reward, isFinal)

                X_batch = np.append(X_batch, np.array([state.copy()]), axis=0)
                Y_sample = qValues.copy()
                Y_sample[action] = targetValue
                Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
                if isFinal:
                    X_batch = np.append(X_batch, np.array([newState.copy()]), axis=0)
                    Y_batch = np.append(Y_batch, np.array([[reward]*self.output_size]), axis=0)
            self.model.fit(X_batch, Y_batch, batch_size = len(miniBatch), nb_epoch=1, verbose = 0)

    def saveModel(self):
        model_json = self.model.to_json()
        with open("model.json","w") as json_file: 
            json_file.write(model_json)
        self.model.save_weights("model.h5")
        
        target_model_json = self.targetModel.to_json()
        with open("target_model.json","w") as json_file: 
            json_file.write(target_model_json)
        self.model.save_weights("target_model.h5")

    def loadModel(self):
        json_file = open("model.json","r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = tf.keras.models.model_from_json(loaded_model_json)
        loaded_model.load_weights("model.h5")
        self.model = loaded_model
        json_file = open("target_model.json","r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = tf.keras.models.model_from_json(loaded_model_json)
        loaded_model.load_weights("target_model.h5")
        self.targetModel = loaded_model
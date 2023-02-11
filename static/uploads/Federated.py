# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 12:35:01 2021
@author: Administrator
"""

def CNN_fun(arg1, arg2=2, arg3=2, mod=""):
    import numpy as np
    import random
    import cv2
    import os
    from imutils import paths
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import confusion_matrix
    import sys
    
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import MaxPooling2D
    from tensorflow.keras.layers import Activation
    from tensorflow.keras.layers import Flatten, Reshape
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import SGD
    import tensorflow.keras.backend as K
    import matplotlib.pyplot as plt
    from tensorflow.keras import preprocessing
    from tensorflow.keras import models
    #from tensorflow.keras.layers import InputSpec

    from keras.engine import Layer, InputSpec
    #from tensorflow.keras.layers import Layer, InputSpec
    
    tf.compat.v1.disable_eager_execution()
    
    
    #import pylab as plt
    import glob
    #from fl_mnist_implementation_tutorial_utils import *
    classes = 0
    
    def load(paths, verbose=-1):
        '''expects images for each class in seperate dir, 
        e.g all digits in 0 class in the directory named 0 '''
        data = list()
        labels = list()
        # loop over the input images
        
        #print(paths)
        for (i, imgpath) in enumerate(paths):
            # load the image and extract the class labels
            im_gray = cv2.imread(imgpath)
            im_gray = cv2.resize(im_gray, (128,128))
            image = np.array(im_gray).flatten()
            label = imgpath.split(os.path.sep)[-2]
            # scale the image to [0, 1] and add to list
            data.append(image/255.)
            labels.append(label)
            # show an update every `verbose` images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(paths)))
        # return a tuple of the data and labels
        
        onehot_encoder = OneHotEncoder(sparse=False)
    
        labels = np.array(labels)
        labels = labels.reshape(len(labels), 1)
        
        onehot_encoded = onehot_encoder.fit_transform(labels)
    
    
        return data, onehot_encoded
    
    #declear path to your mnist data folder
    ###########img_path = sys.argv[1]
    img_path = arg1
    img_path1 = ""
    
    print(arg1)
    for path0 in glob.glob(img_path + "/*"):
        img_path1 = path0
        print(img_path1)
        for path1 in glob.glob(img_path1 + "/*"):
            classes += 1
    #get the path list using the path object
    img_path = img_path1
    
    image_paths = list(paths.list_images(img_path))
    
    #apply our function
    image_list, label_list = load(image_paths, verbose=10000)
    
    #binarize the labels
    #lb = LabelBinarizer()
    #label_list = lb.fit_transform(label_list)
    
    #split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(image_list, 
                                                        label_list, 
                                                        test_size=0.1, 
                                                        random_state=42)
    
    
    X_test1 = X_test
    y_test1 = y_test
    
    def create_clients(image_list, label_list, num_clients=10, initial='clients'):
        ''' return: a dictionary with keys clients' names and value as 
                    data shards - tuple of images and label lists.
            args: 
                image_list: a list of numpy arrays of training images
                label_list:a list of binarized labels for each image
                num_client: number of fedrated members (clients)
                initials: the clients'name prefix, e.g, clients_1 
                
        '''
    
        #create a list of client names
        client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]
    
        #randomize the data
        data = list(zip(image_list, label_list))
        random.shuffle(data)
    
        #shard data and place at each client
        size = len(data)//num_clients
        shards = [data[i:i + size] for i in range(0, size*num_clients, size)]
    
        #number of clients must equal number of shards
        assert(len(shards) == len(client_names))
    
        return {client_names[i] : shards[i] for i in range(len(client_names))} 
    
    
    #create clients
    #clients_num = int(sys.argv[3])
    clients_num = int(arg3)
    clients = create_clients(X_train, y_train, num_clients=clients_num, initial='client')
    
    def batch_data(data_shard, bs=32):
        '''Takes in a clients data shard and create a tfds object off it
        args:
            shard: a data, label constituting a client's data shard
            bs:batch size
        return:
            tfds object'''
        #seperate shard into data and labels lists
        data, label = zip(*data_shard)
        dataset = tf.contrib.data.Dataset.from_tensor_slices((list(data), list(label)))
        return dataset.shuffle(len(label)).batch(bs)
    
    #process and batch the training data for each client
    clients_batched = dict()
    for (client_name, data) in clients.items():
        clients_batched[client_name] = zip(*data)
        
    #process and batch the test set  
    #test_batched = tf.contrib.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))
    test_batched = list(zip(X_test, y_test))
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Convolution2D
    from tensorflow.keras.layers import MaxPooling2D
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Dropout
    
    class SimpleMLP:
        @staticmethod
        def build(shape, classes1):
            model = Sequential()
            model.add(Reshape((128,128,3), input_shape=(128*128*3,)))
            model.add(Convolution2D(64, 3, 3)) 
            convout1 = Activation('relu')
            model.add(convout1)
            convout2 = MaxPooling2D(padding='same')
            model.add(convout2)
            
            model.add(Convolution2D(32, 3, 3)) 
            convout1 = Activation('relu')
            model.add(convout1)
            convout2 = MaxPooling2D(padding='same')
            model.add(convout2)
            
            model.add(Convolution2D(16, 3, 3)) 
            convout1 = Activation('relu')
            model.add(convout1)
            convout2 = MaxPooling2D(padding='same')
            model.add(convout2)
            
            model.add(Flatten())
            
            model.add(Dense(128))
            model.add(Activation('relu'))
            model.add(Dropout(0.2))
            model.add(Dense(classes))
            model.add(Activation('softmax'))
            
            #model.summary()
            return model
        
    lr = 0.01 
    comms_round = 100
    loss='categorical_crossentropy'
    metrics = ['accuracy']
    optimizer = SGD(lr=lr, 
                    decay=lr / comms_round, 
                    momentum=0.9
                   )  
    
    def weight_scalling_factor(clients_trn_data, client_name):
        client_names = list(clients_trn_data.keys())
        #get the bs
        bs = np.array(list(clients_trn_data[client_name])).shape[0]
        #first calculate the total training data points across clinets
        global_count = sum([tf.compat.v1.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
        # get the total number of data points held by a client
        local_count = tf.compat.v1.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
        return local_count/global_count
    
    
    def scale_model_weights(weight, scalar):
        '''function for scaling a models weights'''
        weight_final = []
        steps = len(weight)
        for i in range(steps):
            weight_final.append(scalar * weight[i])
        return weight_final
    
    
    
    def sum_scaled_weights(scaled_weight_list):
        '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
        avg_grad = list()
        #get the average grad accross all client gradients
        for grad_list_tuple in zip(*scaled_weight_list):
            layer_mean  = np.sum(grad_list_tuple, axis=0)
            #layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
            avg_grad.append(layer_mean)
            
        return avg_grad
    
    
    def test_model(X_test1, y_test1,  model, comm_round):
        return model.evaluate(X_test1, y_test1)
    
    def test_model1(X_test, Y_test,  model, comm_round):
        cce = keras.losses.CategoricalCrossentropy(from_logits=True)
        #logits = model.predict(X_test, batch_size=100)
        logits = model.predict(X_test)
        loss = cce(Y_test, logits)
        acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
        print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
        return acc
    
    #initialize global model
    smlp_global = SimpleMLP()
    global_model = smlp_global.build(784, 2)
     
    global_model.compile(loss=loss, 
                          optimizer=optimizer, 
                          metrics=metrics)
           
    #commence global training loop
    client_names= list(clients_batched.keys())
    
    idx = 1
    data[clients_num] = None
    for client in client_names:
        data[idx] = tuple(clients_batched['client_' + str(idx)])
        idx += 1
        if(idx > clients_num):
            break
            
    #comms_round = int(sys.argv[2])
    comms_round = int(arg2)
    
    for comm_round in range(comms_round):
                
        # get the global model's weights - will serve as the initial weights for all local models
        global_weights = global_model.get_weights()
        
        #initial list to collect local model weights after scalling
        scaled_local_weight_list = list()
    
        #randomize client data - using keys
        client_names= list(clients_batched.keys())
        random.shuffle(client_names)
        
        print(client_names)
        #loop through each client and create new local model
        idx = 1
        for client in client_names:
            smlp_local = SimpleMLP()
            local_model = smlp_local.build(784, 2)
            local_model.compile(loss=loss, 
                          optimizer=optimizer, 
                          metrics=metrics)
            
            #set local model weight to the weight of the global model
            local_model.set_weights(global_weights)
            
            #fit local model with client's data
            #print(clients_batched[client])
            data11 = data[idx]
            idx += 1
            """
            data11 = data1
            if(client == 'client_1'):
                data11 = data1
            if(client == 'client_2'):
                data11 = data2
            if(client == 'client_3'):
                data11 = data3
            if(client == 'client_4'):
                data11 = data4
            if(client == 'client_5'):
                data11 = data5
            if(client == 'client_6'):
                data11 = data6
            if(client == 'client_7'):
                data11 = data7
            if(client == 'client_8'):
                data11 = data8
            if(client == 'client_9'):
                data11 = data9
            if(client == 'client_10'):
                data11 = data10
            """
            try:
                data_x = np.array(data11[0])
                data_y = np.array(data11[1])
                local_model.fit(data_x, data_y, epochs=1, verbose=1)
            except:
                pass
            #scale the model weights and add to list
            #scaling_factor = weight_scalling_factor(clients_batched, client)
            scaled_weights = scale_model_weights(local_model.get_weights(), 1/comms_round)
            scaled_local_weight_list.append(scaled_weights)
            
            #clear session to free memory after each communication round
            #K.clear_session()
            #break
        
        #to get the average over all the local model, we simply take the sum of the scaled weights
        average_weights = sum_scaled_weights(scaled_local_weight_list)
        
        #update global model 
        global_model.set_weights(average_weights)
    
        print(np.array(X_test1).shape, np.array(y_test1).shape)
        print(global_model.evaluate(np.array(X_test1), np.array(y_test1), batch_size=200))
        """
        #test global model and print out metrics after each communications round
        for i in test_batched:
            data1 = tuple(zip(*i))
            X_test = np.array(data1[0])
            y_test = np.array(data1[1])
            for idx in range(200):
                global_acc, global_loss = test_model(X_test[idx], y_test[idx], global_model, comm_round)
                
                print(global_acc, global_loss)
        """  
    global_model.save_weights('Federated.h5')        
    prediction = global_model.predict(np.array(X_test1))
    predict_prob = global_model.predict(np.array(X_test1))
    y_test1 = np.argmax(y_test1, axis=1)
    prediction = np.argmax(prediction, axis=1)
    print(y_test1, prediction)     
    predict1 = []
    original = []
    index = 0
    match = 0
    while (index < len(y_test1) and index < np.size(prediction)):
        #print(train_labels[testLabelsGlobal[index]], train_labels[prediction[index]])
        if(y_test1[index] == prediction[index]):
            match += 1
        index += 1
    
    print("Accuracy: ", match/index)
    
    false_pos = 0
    true_neg = 0
    tot_pos = 0
    tot_neg = 0
    
    index = 0
    print(max(prediction))
    while (index < len(y_test1) and index < np.size(prediction)):
        #print(train_labels[testLabelsGlobal[index]], train_labels[prediction[index]])
        
        if(y_test1[index] == min(prediction)):
            if(y_test1[index] != prediction[index]):
                false_pos += 1
            tot_pos += 1
            original.append(0)
            if(prediction[index] == 0):
                predict1.append(0)
            else:
                predict1.append(1)
        elif(y_test1[index] > min(prediction)):
            if(y_test1[index] != prediction[index]):
                true_neg += 1
            tot_neg += 1
            original.append(1)
            if(prediction[index] == 0):
                predict1.append(0)
            else:
                predict1.append(1)
        index += 1
    
    print("False positive :", false_pos, tot_pos)
    print("True Negative :", true_neg, tot_neg)
    
    #fil = sys.argv[1]
    fil = arg1
    fil =  fil + '_1_ROC.png'
    print(fil)
    
    from sklearn.metrics import roc_curve
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(original, predict1)
    
    
    from sklearn.metrics import auc
    auc_keras = auc(fpr_keras, tpr_keras)
    
    fig=plt.figure()
    plt.ion()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label=' AUC (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve CNN')
    plt.legend(loc='best')
    #edit----------
    plt.savefig('static/Uploads/'+ fil)
    plt.close(fig)
    
    from sklearn.metrics import precision_recall_curve, f1_score
    
    #fil = sys.argv[1]
    fil = arg1
    fil = fil + '_PR.png'
    print(fil)
    
    lr_precision, lr_recall, _ = precision_recall_curve(original, predict1)
    # calculate scores
    lr_f1, lr_auc = f1_score(original, predict1), auc(lr_recall, lr_precision)
    # summarize scores
    print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    # plot the precision-recall curves
    positive_class = len(y_test1[y_test1==1]) / len(y_test1)
    plt.plot([0, 1], [positive_class, positive_class], linestyle='--', label='Positive Class')
    plt.plot(lr_recall, lr_precision, marker='.', label='Negative Class')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    plt.savefig('static/Uploads/'+fil)
    plt.close(fig)
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tensorflow.keras.preprocessing import image
    
    #fil = sys.argv[1]
    fil = arg1
    fil = fil + '_1_CM.png'
    print(fil)
    
    
    fig=plt.figure()
    plt.ion()
    
    confusion=confusion_matrix(y_test1,prediction)
    print(confusion)
    df_cm = pd.DataFrame(confusion, range(2), range(2))
    plt.figure(figsize=(10,7))
    sns.set(font_scale=1.4) # for label size
    
    ax = sns.heatmap(df_cm,  annot=True, fmt='d')
    ax.set_ylim([0,2])
    
    plt.title('Confusion_matrix')
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig('static/Uploads/'+fil,bbox_inches='tight')
    #plt.close(fig)
    
    #img_path = r'F:/Mohit/2021/Dec2021/Hikmat/data/Forest_Fire/Video 2 (1).jpg'
    #img_path = sys.argv[1]
    img_path = arg1
    img_path1 = ""
    
    index = 0
    
       
    for path0 in glob.glob(img_path + "/*"):
        img_path1 = path0
        print(img_path1)
        data = list()
        x = data
        idx = 0
        for path1 in glob.glob(img_path1 + "/*"):
            images_path = os.listdir(path1)
    
            for n, image1 in enumerate(images_path):
                src = os.path.join(path1, image1)
    
                im_gray = cv2.imread(src)
                im_gray = cv2.resize(im_gray, (128,128))
                x = np.array(im_gray).flatten()
                data.append(x/255.)
                x = data
                #print(x)
                
                if(idx == 0):
                
                    """
                    img = image.load_img(src, target_size=(128, 128))
                    img_tensor = image.img_to_array(img)
                    img_tensor = np.expand_dims(img_tensor, axis=0)
                    img_tensor /= 255.
                    
                    plt.imshow(img_tensor[0])
                    plt.show()
                    print(img_tensor.shape)
                    """
                    plt.rc('font', size=14, family='times new roman')
                    
                    import tensorflow.keras.backend as K
                    
                    layer_outputs = [layer.output for layer in global_model.layers[1:9]] 
                    # Extracts the outputs of the top 12 layers
                    activation_model = models.Model(inputs=global_model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
                    
                    activations = activation_model.predict(np.array(data)) 
                    
                    first_layer_activation = activations[0]
                    #print(first_layer_activation.shape)
                    plt.matshow(first_layer_activation[0, :, :, 2], cmap='viridis')
                    
                    layer_names = []
                    for layer in global_model.layers[1:9]:
                        layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
                     
                    print(layer_names)
                    images_per_row = 8
                    for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
                        #print(layer_name, layer_activation)
                        n_features = layer_activation.shape[-1] # Number of features in the feature map
                        size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
                        n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
                        display_grid = np.zeros((size * n_cols, images_per_row * size))
                        #print(n_cols)
                        for col in range(n_cols): # Tiles each filter into a big horizontal grid
                            for row in range(images_per_row):
                                channel_image = layer_activation[0,
                                                                 :, :,
                                                                 col * images_per_row + row]
                                #print(channel_image)
                                channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                                channel_image /= channel_image.std()
                                channel_image *= 64
                                channel_image += 128
                                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                                display_grid[col * size : (col + 1) * size, # Displays the grid
                                             row * size : (row + 1) * size] = channel_image
                        scale = 1. / size
                        plt.figure(figsize=(scale * display_grid.shape[1],
                                            scale * display_grid.shape[0]))
                        plt.title(layer_name)
                        plt.grid(False)
                        plt.imshow(display_grid, aspect='auto', cmap='viridis')
                        #fil = sys.argv[1] + "_"
                        fil = arg1 + "_"
                        fil += layer_name
                        fil += ".png"
                        plt.savefig('static/Uploads/'+fil,bbox_inches='tight')
                    plt.close(fig)
                break
    
    #-----------------------------------------------------------------------------------------
    
            '''
            model =global_model
            model.summary()
            #print(x)
            #x=preprocess_input(x)
            preds = model.predict(np.array(data))
            #model.summary()
            print(preds)
            #print('Predicted : ', decode_predictions(preds, top=3)[0])
            m=np.argmax(preds[0]) #386
            print (m)
            for j in range (classes):
                print (j)
                african_elepant_output = model.output[:, j]
                print (african_elepant_output)
            
                last_conv_layer = model.get_layer(index=7)
                print(last_conv_layer.output[0])
                grads= K.gradients(african_elepant_output, last_conv_layer.output)[0]
                pooled_grads= K.mean(grads, axis=(0,1,2))
                print(pooled_grads)
                iterate1 = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
                print(iterate1)
                #print("Value of x is -------------------------------------",len(x[0]))
                pooled_grads_value, conv_layer_output_value = iterate1([x])
            
                for i in range(16):
                    conv_layer_output_value[:, :, i] *=pooled_grads_value[i]
                    
                
                heatmap = np.mean(conv_layer_output_value, axis = -1)
                heatmap = np.maximum(heatmap, 0)
                heatmap /= np.max(heatmap)
                plt.matshow(heatmap)
                """
                plt.savefig("HeatMap_de"+str(j)+'.png')
                import cv2
                img = cv2.imread(img_path)
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap * 0.4 +img
                cv2.imwrite('visualize'+ str(j)+'.jpg', superimposed_img)
                """
            
                #print(src)
                import cv2
                fil = $_$sys.argv[1]
                img = cv2.imread(src)
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap * 0.4 +img
                print(j,idx)
                if(j == 0 and idx == 0):
                    fil = fil + '_HM_P.png'
                    print(fil)
                    cv2.imwrite(fil, superimposed_img)
                elif(j == 1 and idx == 3):
                    fil = fil + '_HM_N.png'
                    print(fil)
                    cv2.imwrite(fil, superimposed_img)
    
                fil = sys.argv[1]
                if(j == 0 and idx == 0):
                    fil = fil + '_HM_P_O.png'
                    cv2.imwrite(fil, img)
                elif(j == 1 and idx == 3):
                    fil = fil + '_HM_N_O.png'
                    cv2.imwrite(fil, img)
                idx += 1
    
        '''
    from skimage.feature import local_binary_pattern
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    #from sklearn.externals import joblib
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import MinMaxScaler
    import mahotas
    
    
    
    # fixed-sizes for image
    fixed_size = tuple((256, 256))
    
    # path to training data
    train_path = "train"
    #img_path = sys.argv[1]
    img_path = arg1
    img_path1 = ""
    
    for path0 in glob.glob(img_path + "/*"):
        train_path = path0
        print(train_path)
    
    # no.of.trees for Random Forests
    num_trees = 100
    
    # bins for histogram
    bins = 8
    
    # train_test_split size
    test_size = 0.10
    
    # seed for reproducing same results
    seed = 9
    
    def calculatehistogram(image1, eps=1e-7):
        lbp = local_binary_pattern(image1, 16, 2, method="uniform")
        (histogram, _) = np.histogram(lbp.ravel(),
                                      bins=np.arange(0, 16 + 3),
                                      range=(0, 16 + 2))
        #now we need to normalise the histogram so that the total sum=1
        histogram = histogram.astype("float")
        histogram /= (histogram.sum() + eps)
        return histogram 
        
    # feature-descriptor-1: Hu Moments
    def fd_hu_moments(image1):
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        feature = cv2.HuMoments(cv2.moments(image1)).flatten()
        #print(len(feature))
        return feature
    
    # feature-descriptor-2: Haralick Texture
    def fd_haralick(image1):
        # convert the image to grayscale
        gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        # compute the haralick texture feature vector
        haralick = mahotas.features.haralick(gray).mean(axis=0)
        # return the result
        #print(len(haralick))
        return haralick
    
    # feature-descriptor-3: Color Histogram
    def fd_histogram(image1, mask=None):
        # convert the image to HSV color-space
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
        # compute the color histogram
        hist  = cv2.calcHist([image1], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
        # normalize the histogram
        cv2.normalize(hist, hist)
        # return the histogram
        #print(len(hist.flatten()))
        return hist.flatten()
    
    def fd_LBP(image1):
        gray= cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
        hist = calculatehistogram(gray)
        #kp = np.array(kp)
        #print(len(pts.flatten()))
        #print(len(hist.flatten()))
        return hist.flatten()
    
    def fd_HOG(image1):
        hog = cv2.HOGDescriptor()
        #im = cv2.imread(sample)
        h = hog.compute(image1)
        return h.flatten()
    
    
    # create all the machine learning models
    models = []
    """
    if(mod=="LR"):
        models.append(('LR', LogisticRegression(random_state=9)))
        models.append(('LDA', LinearDiscriminantAnalysis()))
    elif(mod=="KNN"):
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
    elif(mod=="DT"):
        models.append(('DT', DecisionTreeClassifier(random_state=9)))
    elif(mod=="RF"):
        models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=9)))
    elif(mod=="NB"):
        models.append(('NB', GaussianNB()))
    elif(mod=="SVM"):
        models.append(('SVM', SVC(random_state=9, kernel='rbf')))
    else:
        models.append(('LR', LogisticRegression(random_state=9)))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('DT', DecisionTreeClassifier(random_state=9)))
        models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=9)))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC(random_state=9, kernel='rbf')))
    """
    models.append(('LR', LogisticRegression(random_state=9)))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('DT', DecisionTreeClassifier(random_state=9)))
    models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=9)))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(random_state=9, kernel='rbf')))
    
    # variables to hold the results and names
    results = []
    names = []
    scoring = "accuracy"
    
    
    print( "[STATUS] training started...")
    
    # split the training and testing data
    """
    (trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                              np.array(global_labels),
                                                                                              test_size=test_size,
                                                                                              random_state=seed)
    """
    train_labels = os.listdir(train_path)
    
    # sort the training labels
    train_labels.sort()
    print(train_labels)
    
    # empty lists to hold feature vectors and labels
    global_features = []
    labels = []
    
    i, j = 0, 0
    k = 0
    
    # num of images per class
    images_per_class = 80
    
    # loop over the training data sub-folders
    for training_name in train_labels:
        # join the training data path and each species training folder
        dir = os.path.join(train_path, training_name)
    
        # get the current training label
        current_label = training_name
    
        k = 1
        # loop over the images in each sub-folder
        for fileName in os.listdir(dir):
            # get the image file name
            file = dir + "/" + str(fileName)
            #print(file)
            #print(file)
            # read the image and resize it to a fixed-size
            image1 = cv2.imread(file)
            #print(image)
            #image = cv2.resize(image, fixed_size)
    
            ####################################
            # Global Feature extraction
            ####################################
            fv_hu_moments = fd_hu_moments(image1)
            fv_haralick   = fd_haralick(image1)
            fv_histogram  = fd_histogram(image1)
            #fv_LBP  = fd_LBP(image)
            #fv_HOG  = fd_HOG(image)
    
            ###################################
            # Concatenate global features
            ###################################
            global_feature = np.hstack([fv_hu_moments, fv_haralick, fv_histogram])
    	
            #print(global_feature)
            # update the list of labels and feature vectors
            #print(len(global_feature))
            labels.append(current_label)
            global_features.append(global_feature)
    
            i += 1
            k += 1
        print ("[STATUS] processed folder: {}".format(current_label))
        j += 1
    
    print ("[STATUS] completed Global Feature Extraction...")
    
    # get the overall feature vector size
    print ("[STATUS] feature vector size {}".format(np.array(global_features).shape))
    
    # get the overall training label size
    print ("[STATUS] training Labels {}".format(np.array(labels).shape))
    
    # encode the target labels
    targetNames = np.unique(labels)
    le = LabelEncoder()
    target = le.fit_transform(labels)
    print ("[STATUS] training labels encoded...")
    
    # normalize the feature vector in the range (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_features = scaler.fit_transform(global_features)
    print ("[STATUS] feature vector normalized...")
    
    # save the feature vector using HDF5
    """
    h5f_data = h5py.File('output/data.h5', 'w')
    h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))
    
    h5f_label = h5py.File('output/labels.h5', 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(target))
    
    h5f_data.close()
    h5f_label.close()
    
    # import the feature vector and trained labels
    h5f_data = h5py.File('output/data.h5', 'r')
    h5f_label = h5py.File('output/labels.h5', 'r')
    
    global_features_string = h5f_data['dataset_1']
    global_labels_string = h5f_label['dataset_1']
    
    rescaled_features = np.array(global_features_string)
    target = np.array(global_labels_string)
    
    h5f_data.close()
    h5f_label.close()
    """
    
    trainDataGlobal=np.array(rescaled_features)
    trainLabelsGlobal=np.array(target)
    
    ########################
    """
    test_labels = os.listdir("test")
    
    # sort the testing labels
    test_labels.sort()
    print(test_labels)
    
    # empty lists to hold feature vectors and labels
    global_features = []
    labels = []
    
    i, j = 0, 0
    k = 0
    
    # num of images per class
    images_per_class = 80
    
    # loop over the testing data sub-folders
    for testing_name in test_labels:
        # join the testing data path and each species testing folder
        dir = os.path.join("test", testing_name)
    
        # get the current testing label
        current_label = testing_name
    
        k = 1
        # loop over the images in each sub-folder
        for fileName in os.listdir(dir):
            # get the image file name
            file = dir + "/" + str(fileName)
    
            #print(file)
            # read the image and resize it to a fixed-size
            image = cv2.imread(file)
            #image = cv2.resize(image, fixed_size)
    
            ####################################
            # Global Feature extraction
            ####################################
            fv_hu_moments = fd_hu_moments(image)
            fv_haralick   = fd_haralick(image)
            fv_histogram  = fd_histogram(image)
            #fv_LBP  = fd_LBP(image)
            #fv_HOG  = fd_HOG(image)
    
    
            ###################################
            # Concatenate global features
            ###################################
            global_feature = np.hstack([fv_hu_moments, fv_haralick, fv_histogram])
    
            #print(len(global_feature))
            # update the list of labels and feature vectors
            labels.append(current_label)
            global_features.append(global_feature)
    
            i += 1
            k += 1
        print ("[STATUS] processed folder: {}".format(current_label))
        j += 1
    
    print ("[STATUS] completed Global Feature Extraction...")
    
    # get the overall feature vector size
    print ("[STATUS] feature vector size {}".format(np.array(global_features).shape))
    
    # get the overall testing label size
    print ("[STATUS] testing Labels {}".format(np.array(labels).shape))
    
    # encode the target labels
    targetNames = np.unique(labels)
    le = LabelEncoder()
    target = le.fit_transform(labels)
    print ("[STATUS] testing labels encoded...")
    
    # normalize the feature vector in the range (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_features = scaler.fit_transform(global_features)
    print ("[STATUS] feature vector normalized...")
    
    testDataGlobal=np.array(rescaled_features)
    testLabelsGlobal=np.array(target)
    
    print( "[STATUS] splitted train and test data...")
    print( "Train data  : {}".format(trainDataGlobal.shape))
    print( "Test data   : {}".format(testDataGlobal.shape))
    print( "Train labels: {}".format(trainLabelsGlobal.shape))
    print( "Test labels : {}".format(testLabelsGlobal.shape))
    
    # filter all the warnings
    import warnings
    warnings.filterwarnings('ignore')
    """
    ###################################
    trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal = train_test_split(trainDataGlobal, 
                                                        trainLabelsGlobal, 
                                                        test_size=0.1, 
                                                        random_state=42)
    
    
    train_labels = os.listdir(train_path)
    
    # 10-fold cross validation
    fig=plt.figure()
    plt.ion()
    
    for name, model in models:
        #print(name)
        #print(type(name))
        
        if(name==str(mod)):
        
            model.fit(trainDataGlobal, trainLabelsGlobal)
        
            # without cross-valdidation
            prediction = model.predict(testDataGlobal)
            if(name != 'SVM'):
                prediction1 = model.predict_proba(testDataGlobal)
        
            predict1 = []
            original = []
            index = 0
            match = 0
            while (index < len(testLabelsGlobal) and index < len(prediction)):
                #print(train_labels[testLabelsGlobal[index]], train_labels[prediction[index]])
                if(testLabelsGlobal[index] == prediction[index]):
                    match += 1
                index += 1
            
            print(name, "Accuracy: ", match/index)
            
            false_pos = 0
            true_neg = 0
            tot_pos = 0
            tot_neg = 0
            
            index = 0
            #print(max(prediction))
            while (index < len(testLabelsGlobal) and index < len(prediction)):
                #print(train_labels[testLabelsGlobal[index]], train_labels[prediction[index]])
                
                if(testLabelsGlobal[index] < max(prediction)):
                    if(testLabelsGlobal[index] != prediction[index]):
                        false_pos += 1
                    tot_pos += 1
                    original.append(0)
                    if(name != 'SVM'):
                        predict1.append(1-prediction1[index][testLabelsGlobal[index]])
                    else:
                        if(testLabelsGlobal[index] != prediction[index]):
                            predict1.append(1)
                        else:
                            predict1.append(0)
                elif(testLabelsGlobal[index] == max(prediction)):
                    if(testLabelsGlobal[index] != prediction[index]):
                        true_neg += 1
                    tot_neg += 1
                    original.append(1)
                    if(name != 'SVM'):
                        predict1.append(prediction1[index][testLabelsGlobal[index]])
                    else:
                        if(testLabelsGlobal[index] != prediction[index]):
                            predict1.append(0)
                        else:
                            predict1.append(1)
                index += 1
            
            print(name, "False positive :", false_pos, tot_pos)
            print(name, "True Negative :", true_neg, tot_neg)
            
        
            tem = 1
            if(tem==1): #(name == 'LR' or name == 'KNN' or name == 'DT' or name == 'SVM'):
                from sklearn.metrics import roc_curve
                fpr_keras, tpr_keras, thresholds_keras = roc_curve(original, predict1)
                
                fil = arg1
                fil =  fil + '_1_ROC_final.png'
                from sklearn.metrics import auc
                auc_keras = auc(fpr_keras, tpr_keras)
                
                #extraopen
                fig=plt.figure()
                #extraclose
                plt.plot([0, 1], [0, 1], 'k--')
                plt.plot(fpr_keras, tpr_keras, label=name + ' AUC (area = {:.3f})'.format(auc_keras))
                plt.xlabel('False positive rate')
                plt.ylabel('True positive rate')
                plt.title('ROC curve '+ name)
                plt.legend(loc='best')
                
                ##extra
        
                plt.savefig('static/Uploads/'+ fil)  
                plt.close(fig)
       #------------------------------
        """
        kfold = KFold(n_splits=10, random_state=7)
        cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
    
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
    plt.show()  
        
    #fil = sys.argv[1]
    fil = arg1
    fil = fil + '_ROC1.png'
    print(fil)
    plt.savefig('static/Uploads/'+fil)
    plt.close(fig)
    
    data = []
    #fil = sys.argv[1]
    fil = arg1
    """
    '''
    im_gray = cv2.imread(fil + "_sample.jpg")
    im_gray = cv2.resize(im_gray, (128,128))
    x = np.array(im_gray).flatten()
    data.append(x/255.)
    x = data
    #print(x)
    idx = 0
    if(idx == 0):
    #--------------------
        """
        img = image.load_img(src, target_size=(128, 128))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        
        plt.imshow(img_tensor[0])
        plt.show()
        print(img_tensor.shape)
        """
        plt.rc('font', size=14, family='times new roman')
        
        import tensorflow.keras.backend as K
        
        layer_outputs = [layer.output for layer in global_model.layers[1:9]] 
        # Extracts the outputs of the top 12 layers
        activation_model = keras.models.Model(inputs=global_model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
        
        activations = activation_model.predict(np.array(data)) 
        
        first_layer_activation = activations[0]
        #print(first_layer_activation.shape)
        plt.matshow(first_layer_activation[0, :, :, 2], cmap='viridis')
        
        layer_names = []
        for layer in global_model.layers[1:9]:
            layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
         
        print(layer_names)
        images_per_row = 8
        for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
            #print(layer_name, layer_activation)
            n_features = layer_activation.shape[-1] # Number of features in the feature map
            size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
            n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
            display_grid = np.zeros((size * n_cols, images_per_row * size))
            #print(n_cols)
            for col in range(n_cols): # Tiles each filter into a big horizontal grid
                for row in range(images_per_row):
                    channel_image = layer_activation[0,
                                                     :, :,
                                                     col * images_per_row + row]
                    #print(channel_image)
                    channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    display_grid[col * size : (col + 1) * size, # Displays the grid
                                 row * size : (row + 1) * size] = channel_image
            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            #fil = sys.argv[1] + "_"
            fil = arg1 + "_"
            fil += layer_name
            fil += "_sample.png"
            plt.savefig(fil,bbox_inches='tight')
            #plt.close(fig)
            #break
    
    
    classes = 2
    model =global_model
    #print(x)
    #x=preprocess_input(x)
    preds = model.predict(np.array(data))
    #model.summary()
    print(preds)
    #print('Predicted : ', decode_predictions(preds, top=3)[0])
    m=np.argmax(preds[0]) #386
    african_elepant_output = model.output[:, 0]
    print (african_elepant_output)
    
    last_conv_layer = model.get_layer(index=7)
    print(last_conv_layer.output[0])
    grads= K.gradients(african_elepant_output, last_conv_layer.output)[0]
    pooled_grads= K.mean(grads, axis=(0,1,2))
    print(pooled_grads)
    iterate1 = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    print(iterate1)
    pooled_grads_value, conv_layer_output_value = iterate1([x])
    
    for i in range(16):
        conv_layer_output_value[:, :, i] *=pooled_grads_value[i]
        
    
    heatmap = np.mean(conv_layer_output_value, axis = -1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap)
    #--------------------
    """
    plt.savefig("HeatMap_de"+str(j)+'.png')
    import cv2
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 +img
    cv2.imwrite('visualize'+ str(j)+'.jpg', superimposed_img)
    """
    
    #print(src)
    import cv2
    #fil = sys.argv[1]
    fil = arg1
    img = cv2.imread(fil + "_sample.jpg")
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 +img
    fil = fil + '_HM_sample.png'
    print(fil)
    cv2.imwrite(fil, superimposed_img)
    
    arr = global_model.predict(np.array(data))
    arr = np.argmax(arr, axis=1)
    
    print(arr)
    if(arr == 1):
        print('The sample image is having Covid disease')
    else:
        print('The sample image is not having Covid disease')
    
    '''
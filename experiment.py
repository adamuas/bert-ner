import os
import shutil
import pandas as pd
from glob import glob
import seaborn as sns
from matplotlib import pylab as plt
import tensorflow as tf
# depenciy handling
if tf.__version__ != '2.3.0':
    !pip install tensorflow-gpu==2.3.0
    
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization 

tf.get_logger().setLevel('ERROR')


AUTOTUNE = tf.data.experimental.AUTOTUNE
batch_size = 32
target_class = 'Class'

output_dir='./ner'
train_file_path = os.path.join(output_dir, 'train.csv')
val_file_path = os.path.join(output_dir, 'val.csv')
test_file_path = os.path.join(output_dir, 'test.csv')


# read train dataset and batch 
train_batch = tf.data.experimental.make_csv_dataset(
    train_file_path,
    batch_size=batch_size,
    select_columns=['Name', 'Class'],
    label_name=target_class).cache().prefetch(buffer_size=AUTOTUNE)

# read validation dataset and batch 
val_batch = tf.data.experimental.make_csv_dataset(
    val_file_path, 
    batch_size=batch_size,
    select_columns=['Name', 'Class'],
    label_name=target_class).cache().prefetch(buffer_size=AUTOTUNE)

# read test dataset and batch 
test_batch = tf.data.experimental.make_csv_dataset(
    test_file_path, 
    batch_size=batch_size, 
    select_columns=['Name', 'Class'],
    label_name=target_class).cache().prefetch(buffer_size=AUTOTUNE)


tf.get_logger().setLevel('ERROR')
# bert model map
model_map = {
    "small_bert/bert_en_uncased_L-4_H-512_A-8": 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'
}

# bert preprocessor map
preprocssor_map = {
    "small_bert/bert_en_uncased_L-4_H-512_A-8": 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1',
}

class MultiLayerPerceptronBlock(tf.keras.layers.Layer):
    """ 
    Multilayer Perceptron Block Using swish a modified sigmoide function 
    found to perform better than relu in terms
    """
    def __init__(self,num_units=10, layers=None):
        super(MultiLayerPerceptronBlock, self).__init__()
        if layers is None:
          self.layers = [
              tf.keras.layers.Dense(units=num_units, activation='swish', name='mlp_h_1'),
              tf.keras.layers.Dropout(rate=0.1),
              tf.keras.layers.Dense(units=num_units, activation='swish', name='mlp_h_2')
          ]
        else:
          if isinstance(layers, list):
            self.layers = layers
          else:
            raise Exeption(f"Expected layers to be of type list<tf.keras.layers.*> but got {type(layers)}")

    def call(self, inputs):
        layer_output = inputs
        for i in range(0, len(self.layers)):
          l_i = self.layers[i]
          layer_output = l_i(layer_output)
        return layer_output


def BERTEncoderBlock(bert_architecture='small_bert/bert_en_uncased_L-4_H-512_A-8', 
                    trainable=True):
    """ Generates and returns a BERT Encoder Block """
    input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, 
                                      name='text')
    processor = hub.KerasLayer(preprocssor_map.get(bert_architecture), 
                           name='preprocessing')
    bert_encoder = hub.KerasLayer(model_map.get(bert_architecture), trainable=trainable,
                          name='bert_transformer_block_encode')
  
    encoded_inputs = processor(input_layer)
    outputs = bert_encoder(encoded_inputs)
    net_output = outputs['pooled_output']

    return input_layer, net_output



def BERTMLPNerClassifierFactory(bert_architecture='small_bert/bert_en_uncased_L-4_H-512_A-8', 
                                bert_trainable=False,
                                num_classes=14):
    """
    Uses BERT for encoding inputs and outputs
    """
    # intialise 
    bert_input, bert_out = BERTEncoderBlock(bert_architecture=bert_architecture, 
                                            trainable=bert_trainable)
    # intialise MLP Block
    mlp_block = MultiLayerPerceptronBlock()
    mlp_out = mlp_block(bert_out)

    softmax_layer = tf.keras.layers.Dense(units=num_classes, activation='softmax', 
                                    name='class_probability')
    
    class_prob = softmax_layer(mlp_out)

    model = tf.keras.Model(inputs=bert_input, outputs=class_prob)

    return model




class BERTForNER(tf.keras.Model):
    """
    Define our BERForNER Model
    Buggy implementation
    """
    def __init__(self, bert_architecture='small_bert/bert_en_uncased_L-4_H-512_A-8', num_classes=14):
        super(BERTForNER, self).__init__()
        self.bert_architecture = bert_architecture
        

    def init_preprocessor(self, bert_architecture):
        tf_hub_handle = preprocssor_map.get(bert_architecture, None)
        if tf_hub_handle:
            return hub.KerasLayer(tf_hub_handle, name='BERT_Preprocessor')
        else:
            raise Exception(f"Could not find {bert_architecture} in preprocossor map")
    
    def init_bert_block(self, bert_architecture, is_training=True):
        tf_hub_handle = preprocssor_map.get(bert_architecture, None)
        if tf_hub_handle:
            return hub.KerasLayer(tf_hub_handle, trainable=is_training, name='BERT_Encoder_Block')
        else:
            raise Exception(f"Could not find {bert_architecture} in model map")

    def build(self, input_shape):
        # init input layer
        self.input_layer = tf.keras.layers.Input(shape=input_shape, 
                                                dtype=tf.string, 
                                                 name='text')
        # intialise bert preprocessor
        self.preprocessor = self.init_preprocessor(self.bert_architecture)
        prepocessed_input = self.preprocessor(self.input_layer)

        # initlaise bert encoder block
        self.bert_transformer_block = self.init_bert_block(self.bert_architecture)
        bert_output = self.bert_transformer_block(prepocessed_input)["pooled_output"]

        # intiialise mlp block
        self.mlp_block = MultiLayerPerceptronBlock()
        mlp_output = self.mlp_block(bert_output)
        
        self.softmax_layer = tf.keras.layers.Dense(num_classes, activation='softmax', name='class_prob')

        class_prob = self.softmax_layer(mlp_output)

        return tf.keras.Model(inputs=self.input_layer, outputs=class_prob)
      
    def call(self, inputs):
        inputs = self.input_layer(inputs)
        encoder_inputs = self.preprocessor(self.input_layer)
        bert_pooled_output = self.bert_transformer_block(encoder_inputs)['pooled_output']
        mlp_out = self.mlp_block(bert_pooled_output)
        class_prob = self.class_prob(mlp_out)
        
        return class_prob

    
    

if __name__ == '__main__':
    # generate instance of BERT-MLP NER Classifier
    bert_mlp_ner_classifier = BERTMLPNerClassifierFactory(num_classes=15, bert_trainable=False)
    # print network summary
    bert_mlp_ner_classifier.summary()
    
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    metrics = [tf.metrics.Precision(), tf.metrics.Recall(), tf.keras.metrics.Accuracy(), loss]
    
    epochs = 30
    steps_per_epoch = tf.data.experimental.cardinality(train_batch).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')
    
    
    callbacks = [tf.keras.callbacks.ModelCheckpoint("./storage/weights/", 
                                                monitor="val_acc", 
                                                mode='max',
                                               save_best_only=True)]
    
    
    bert_mlp_ner_classifier.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy(), 'accuracy'])
    
    
    history = bert_mlp_ner_classifier.fit(x=train_batch,
                                   validation_data=val_batch,
                                   epochs=5,
                                   callbacks=callbacks)
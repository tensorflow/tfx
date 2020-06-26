import tensorflow.keras as keras
import tensorflow as tf

def BertForSingleSentenceClassification(
    bert_layer,
    max_len,
    fully_connected_layers=[]):
    """Keras model for single sentence classification.
    Connect configurable fully connected layers on top of the Bert 
    pooled_output.

    Args:
        bert_layer: A tensroflow_hub.KerasLayer intence of Bert layer.
        max_len: The maximum length of preprocessed tokens.
        hidden_layers: List of configurations for fine-tuning hidden layers
            after the pooled_output. [(#of hidden units, activation)].

    Returns:
        A Keras model.
    """
    input_layer_names = [
        "input_word_ids",
        "input_mask",
        "segment_ids"]

    input_layers = [
        keras.layers.Input(
            shape=(max_len,),
            dtype=tf.int32,
            name=name) for name in input_layer_names
        ]

    pooled_output, sequence_output = bert_layer(input_layers)

    fully_connected = pooled_output
    for (i, activation) in fully_connected_layers:
        fully_connected = keras.layers.Dense(i,
                activation=activation)(fully_connected)

    output = keras.layers.Dense(1, activation='sigmoid')(fully_connected)
    model = keras.Model(input_layers, output)
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.binary_crossentropy,
        metrics=['accuracy'])
    return model

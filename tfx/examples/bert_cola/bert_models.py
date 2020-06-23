import tensorflow.keras as keras
import tensorflow as tf

def BertForSingleSentenceSentimentAnalysis(
    bert_layer,
    max_len,
    hidden_layers=[]):
    """Keras model for single sentence sentiment analysis. Connect
    configurable fully connected layers on top of the Bert pooled_output.

    Args:
        bert_layer: A tensroflow_hub.KerasLayer intence of Bert layer.
        max_len: The maximum length of preprocessed tokens.
        hidden_layers: List of configurations for fine-tuning hidden layers
            after the pooled_output. [(#of hidden units, activation)].

    Returns:
        A Keras model.
    """
    input_id_layer = keras.layers.Input(
        shape=(max_len,),
        dtype=tf.int32,
        name="input_word_ids")
    
    input_mask_layer = keras.layers.Input(
        shape=(max_len,),
        dtype=tf.int32,
        name="input_mask")
    
    input_type_ids_layer = keras.layers.Input(
        shape=(max_len,),
        dtype=tf.int32,
        name="segment_ids")

    pooled_output, sequence_output = bert_layer([
        input_id_layer,
        input_mask_layer,
        input_type_ids_layer])

    hidden = pooled_output
    for (i, activation) in hidden_layers:
        hidden = keras.layers.Dense(i, activation=activation)(hidden)

    output = keras.layers.Dense(1, activation='sigmoid')(hidden)
    model = keras.Model([
        input_id_layer,
        input_mask_layer,
        input_type_ids_layer],
        output)

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.binary_crossentropy,
        metrics=['AUC'])
    return model

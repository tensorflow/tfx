import tensorflow as tf
import tensorflow_text as text

class Special_Bert_Tokenizer():

    def __init__(self, vocab_dir):
        self.vocab_dir = vocab_dir
        self._find_special_token()

    def _find_special_token(self):
        """Find the special token ID's for [CLS] [PAD] [SEP]"""
        f = open(self.vocab_dir, 'r')
        i = 0
        self.SEP_ID = None
        self.CLS_ID = None
        self.PAD_ID = None
        lines = f.read().split('\n')
        for line in lines:
            if line == "[PAD]":
                self.PAD_ID = tf.constant(i, dtype=tf.int64) 
            elif line == "[CLS]":
                self.CLS_ID = tf.constant(i, dtype=tf.int64) 
            elif line == "[SEP]":
                self.SEP_ID = tf.constant(i, dtype=tf.int64) 
            i += 1
            if self.PAD_ID != None \
                and self.CLS_ID != None \
                and self.SEP_ID != None:
                break

    def tokenize_single_sentence(
        self,
        sequence,
        max_len=128,
        addCLS=True,
        addSEP=True):
        """Tokenize a single sentence to ID according to the vocab.txt provided.
        Add special tokens according to config."""

        tokenizer = text.BertTokenizer(self.vocab_dir, token_out_type=tf.int64)
        word_id = tokenizer.tokenize(sequence)
        word_id = word_id.merge_dims(1, 2)[:, :max_len]
        word_id = word_id.to_tensor(default_value=self.PAD_ID)
        if addCLS:
            CLSToken = tf.fill([tf.shape(sequence)[0], 1], self.CLS_ID)
            word_id = word_id[:, :max_len-1]
            word_id = tf.concat([CLSToken, word_id], axis=1)
        
        if addSEP:
            SEPToken = tf.fill([tf.shape(sequence)[0], 1], self.SEP_ID)
            word_id = word_id[:, :max_len-1]
            word_id = tf.concat([word_id, SEPToken], axis=1)

        word_id = tf.pad(
            word_id,
            [[0, 0], [0, max_len]],
            constant_values=self.PAD_ID)
        
        word_id = tf.slice(word_id, [0, 0], [-1, max_len])

        # Mask to distinguish padded values.
        input_mask = tf.cast(word_id > 0, tf.int64)
        # Mask to distinguish two sentences. In this case, just one sentence.
        segment_id = tf.fill(
            tf.shape(input_mask),
            tf.constant(0, dtype=tf.int64))

        return word_id, input_mask, segment_id




    



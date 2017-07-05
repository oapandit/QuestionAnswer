import os
import numpy as np
from model import tqa_model
from data_prepare import prepare_data


class tqa_system():
    def __init__(self,processed_data_path):
        self.word_vec_size = 300
        self.max_q_length = 65
        self.max_doc_length = 300
        self.max_option_length = 25
        self.max_opt_count = 7
        self.nb_epoch = 100
        self.batch_size = 16
        self.steps_per_epoch=333
        self.processed_data_path = processed_data_path
        self.models_path = os.path.join("/home/cvpr/akshay/TQA/train","saved_models")
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)

    def train_model(self):
        model_fname = "masked_model_wt.h5"
        read_data = prepare_data(self.processed_data_path,self.word_vec_size,self.max_q_length,self.max_doc_length,self.max_option_length,self.max_opt_count)
        model = tqa_model(self.word_vec_size,self.max_q_length,self.max_doc_length,self.max_option_length,self.max_opt_count)
        train_model = model.get_masked_model()

        #train_model.fit([question_mat, sent_mat, options_mat], [correct_ans_mat],epochs=self.epochs,batch_size=self.batch_size)
        train_model.fit_generator(read_data.read_all_vectors(),steps_per_epoch=self.steps_per_epoch,epochs = self.nb_epoch,verbose=1)

        train_model.save_weights(os.path.join(self.models_path,model_fname))


if __name__ == "__main__":

    processed_data_path = "/home/cvpr/akshay/TQA/train/processed_data/one_hot_files"
    tqa_sys = tqa_system(processed_data_path)
    tqa_sys.train_model()











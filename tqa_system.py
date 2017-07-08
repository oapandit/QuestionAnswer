import os
import numpy as np
from model import tqa_model
from data_prepare import prepare_data


class tqa_system():
    def __init__(self,train_data_path,val_data_path):
        self.word_vec_size = 300
        self.max_q_length = 65
        self.max_doc_length = 300
        self.max_option_length = 25
        self.max_opt_count = 7
        self.nb_epoch = 50
        self.batch_size = 16
        self.steps_per_epoch=333
        self.validation_steps = 100
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.models_path = os.path.join("/home/cvpr/akshay/TQA/train","saved_models")
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)

    def train_model(self):
        #model_load_weights_fname = 'minimal(no bias)_model_softmax_wt_120_epochs.h5'
        #model_fname = "minimal(no bias)_model_softmax_wt_122_epochs.h5"
        read_train_data = prepare_data(self.train_data_path,self.word_vec_size,self.max_q_length,self.max_doc_length,self.max_option_length,self.max_opt_count)
        read_val_data = prepare_data(self.val_data_path, self.word_vec_size, self.max_q_length, self.max_doc_length,
                                       self.max_option_length, self.max_opt_count)
        model = tqa_model(self.word_vec_size,self.max_q_length,self.max_doc_length,self.max_option_length,self.max_opt_count)
        #train_model = model.get_model_show_and_tell(mask=True)
        train_model = model.get_minimal_model_with_softmax()
        #train_model.load_weights(os.path.join(self.models_path,model_load_weights_fname))
        print(train_model.optimizer.lr.get_value())
        #train_model.fit([question_mat, sent_mat, options_mat], [correct_ans_mat],epochs=self.epochs,batch_size=self.batch_size)
        train_model.fit_generator(read_train_data.read_all_vectors(),steps_per_epoch=self.steps_per_epoch,epochs = self.nb_epoch,validation_data=read_val_data.read_all_vectors(),validation_steps=self.validation_steps,verbose=1)

        train_model.save_weights(os.path.join(self.models_path,model_fname))


if __name__ == "__main__":

    train_data_path = "/home/cvpr/akshay/TQA/train/processed_data/one_hot_files"
    val_data_path = "/home/cvpr/akshay/TQA/val/processed_data/one_hot_files"
    tqa_sys = tqa_system(train_data_path,val_data_path)
    tqa_sys.train_model()











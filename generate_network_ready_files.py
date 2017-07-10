import os
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import pickle
from string import ascii_lowercase
import pandas
import math
import statistics
from nltk.stem import WordNetLemmatizer
import string
import re

word2vec_model_path = "/home/cvpr/Debjyoti/docvec"
word2vec_model_name = "20Newsgroup.bin"
# wordvec_size = 300
# vec_random = np.random.rand(1, wordvec_size)

class generate_network_ready_files():
    def __init__(self,processed_data_path,word_vec_size,max_q_length,max_doc_length,max_option_length,max_opt_count,op_path=None):
        print "Generating network ready files."
        self.processed_data_path = processed_data_path
        self.raw_text_path = os.path.join(processed_data_path,"text_question_sep_files")
        if op_path is None:
            op_path = os.path.join(processed_data_path,"one_hot_files")
        if not os.path.exists(op_path):
            os.makedirs(op_path)
        self.op_path = op_path
        self.word_vec_size = word_vec_size
        self.num_of_words_in_opt = max_option_length
        self.num_of_words_in_question = max_q_length
        self.num_of_words_in_closest_sentence = max_doc_length
        self.num_of_options_for_quest = max_opt_count
        self.lessons_list = self.get_list_of_dirs(self.raw_text_path)
        self.unknown_words_vec_dict = None
        self.unknown_words_vec_dict_file = "unk_word2vec_dict.pkl"
        self.common_files_path = "/home/cvpr/akshay/TQA/common_files"

    def get_list_of_dirs(self,dir_path):
        dirlist = [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]
        dirlist.sort()
        return dirlist


    def get_list_of_files(self,file_path,file_extension=".txt"):
        filelist = []
        for root, dirs, files in os.walk(file_path):
            for filen in files:
                if filen.endswith(file_extension):
                    filelist.append(filen)
        filelist.sort()
        return filelist

    def handle_unknown_words(self,word):
        fname = self.unknown_words_vec_dict_file
        if self.unknown_words_vec_dict is None:
            print "Dict is none"
            if os.path.isfile(os.path.join(self.common_files_path,fname)):
                print "Dict file exist"
                with open(os.path.join(self.common_files_path,fname), 'rb') as f:
                    self.unknown_words_vec_dict = pickle.load(f)
            else:
                print "Dict file does not exist"
                self.unknown_words_vec_dict = {}
        if self.unknown_words_vec_dict.get(word,None) is not None:
            print "word present in dictionary : ",word
            vec = self.unknown_words_vec_dict.get(word,None)
        else:
            print "word is not present in dictionary : ", word
            vec = np.random.rand(1,self.word_vec_size)
            self.unknown_words_vec_dict[word] = vec
        return vec

    def get_vec_for_word(self,model, word):
        try:
            vec = model[word]
            return vec
        except:
            print "Vector not in model for word",word
            vec = self.handle_unknown_words(word)
            return vec

    def write_vecs_to_file(self,model,raw_data_content,word2vec_file,is_correct_answer_file = False):
        all_vec_array = np.array([])
        number_of_words = 1
        break_loop = False
        if is_correct_answer_file:
            word = raw_data_content[0].strip().lower()
            pos = ord(word) -97
            all_vec_array = -1 * np.ones(self.num_of_options_for_quest)
            all_vec_array[pos] = 1
        else:
            for sent in raw_data_content:
                words = word_tokenize(sent)
                for word in words:
                    word = word.strip().lower()
                    vec = self.get_vec_for_word(model, word)
                    all_vec_array = np.append(all_vec_array, vec)
                    number_of_words+=1
                    if number_of_words>self.num_of_words_in_closest_sentence-1:
                        break_loop = True
                        break
                if break_loop:
                    break
        pickle.dump(all_vec_array, word2vec_file)
        word2vec_file.close()

    def generate_word2vec_for_all(self):
        model = Word2Vec.load_word2vec_format(os.path.join(word2vec_model_path,word2vec_model_name), binary=True)
        # model = ""
        for lesson in self.lessons_list:
            l_dir = os.path.join(self.raw_text_path,lesson)
            print ("Lesson : ",lesson)
            op_l_dir = os.path.join(self.op_path,lesson)
            if not os.path.exists(op_l_dir):
                os.makedirs(op_l_dir)
            questions_dir = self.get_list_of_dirs(l_dir)
            for question_dir in questions_dir:
                file_list = self.get_list_of_files(os.path.join(l_dir,question_dir))
                if not os.path.exists(os.path.join(op_l_dir,question_dir)):
                    os.makedirs(os.path.join(op_l_dir,question_dir))
                print ("Question : ", question_dir)
                for fname in file_list:
                    if fname == "correct_answer.txt":
                        is_correct_answer_file = True
                    else:
                        is_correct_answer_file = False
                    with open(os.path.join(l_dir,question_dir, fname),"r") as f:
                        raw_data_content = f.readlines()
                    f = open(os.path.join(op_l_dir, question_dir, fname[:-4]+".pkl"),"w")
                    self.write_vecs_to_file(model,raw_data_content,f,is_correct_answer_file)
                    f.close()
            print (20*"***")
        print "saving final unknown word2vec dictionary to file"
        f = open(os.path.join(self.common_files_path,self.unknown_words_vec_dict_file), "wb")
        pickle.dump(self.unknown_words_vec_dict, f)
        f.close()


if __name__ == "__main__":

    word_vec_size = 300
    max_q_length = 65
    max_doc_length = 300
    max_option_length = 25
    max_opt_count = 7
    processed_data_path = "/home/cvpr/akshay/TQA/val/processed_data"
    read_training_json = generate_network_ready_files(processed_data_path,word_vec_size,max_q_length,max_doc_length,max_option_length,max_opt_count)
    read_training_json.generate_word2vec_for_all()

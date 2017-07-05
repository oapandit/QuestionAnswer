import os
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import pickle
from string import ascii_lowercase
import pandas
import math
import statistics
from generate_network_ready_files import generate_network_ready_files

import string
import re

# Adding an extra comment here

class prepare_data():
    def __init__(self,processed_data_path,word_vec_size,max_q_length,max_doc_length,max_option_length,max_opt_count):
        if not os.path.exists(processed_data_path):
            g_network_ready_files = generate_network_ready_files(os.path.dirname(processed_data_path),word_vec_size,max_q_length,max_doc_length,max_option_length,max_opt_count)
            g_network_ready_files.generate_word2vec_for_all()
        self.word_vec_size = word_vec_size
        self.num_of_words_in_opt = max_option_length
        self.num_of_words_in_question = max_q_length
        self.num_of_words_in_closest_sentence = max_doc_length
        self.num_of_options_for_quest = max_opt_count
        # self.pad_word_vector = np.random.rand(1, self.word_vec_size)
        # self.pad_opt_vector = np.random.rand(1, self.num_of_words_in_opt, self.word_vec_size)
        self.pad_word_vector = np.zeros((1, self.word_vec_size))
        self.pad_opt_vector = np.zeros((1, self.num_of_words_in_opt, self.word_vec_size))
        self.processed_data_path = processed_data_path
        self.lessons_list = self.get_list_of_dirs(self.processed_data_path)
        self.options_file = ["a.pkl", "b.pkl", "c.pkl", "d.pkl", "e.pkl", "f.pkl", "g.pkl"]
        self.closest_sent_file = "closest_sent.pkl"
        self.correct_answer_file = "correct_answer.pkl"
        self.quest_file = "Question.pkl"

    def get_list_of_dirs(self,dir_path):
        dirlist = [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]
        dirlist.sort()
        return dirlist

    def get_list_of_files(self,file_path,file_extension=".pkl"):
        filelist = []
        for root, dirs, files in os.walk(file_path):
            for filen in files:
                if filen.endswith(file_extension):
                    filelist.append(filen)
        filelist.sort()
        return filelist

    def read_options_files(self,question_dir_path):
        complete_array = None
        num_of_options = 0

        for f_name in self.options_file:
            # print(fname)
            if not os.path.exists(os.path.join(question_dir_path, f_name)):
                break
            f = open(os.path.join(question_dir_path, f_name), 'r')
            complete_array_part = pickle.load(f)
            complete_array_part = complete_array_part.reshape(-1,self.word_vec_size)
            # print(" options : shape : " + str(complete_array_part.shape))
            if complete_array_part.shape[0] > self.num_of_words_in_opt:
                complete_array_part = complete_array_part[:self.num_of_words_in_opt,:]
            while complete_array_part.shape[0]<self.num_of_words_in_opt:
                complete_array_part = np.concatenate((complete_array_part, self.pad_word_vector), axis=0)
            # print(" options : shape : " + str(complete_array_part.shape))
            complete_array_part = complete_array_part.reshape(1,self.num_of_words_in_opt, self.word_vec_size)
            # print(" options : shape : " + str(complete_array_part.shape))
            complete_array = complete_array_part if complete_array is None else np.concatenate((complete_array, complete_array_part), axis=0)
            num_of_options+=1
        while num_of_options<self.num_of_options_for_quest:
            complete_array = np.concatenate((complete_array, self.pad_opt_vector), axis=0)
            num_of_options+=1
        complete_array = complete_array.reshape(1,self.num_of_options_for_quest, self.num_of_words_in_opt, self.word_vec_size)
        # print(" options : shape : " + str(complete_array.shape))
        return complete_array

    def read_question_file(self,question_dir_path):
        f = open(os.path.join(question_dir_path, self.quest_file), 'r')
        complete_array = pickle.load(f)
        complete_array = complete_array.reshape(-1,self.word_vec_size)
        if complete_array.shape[0] > self.num_of_words_in_question:
            complete_array = complete_array[:self.num_of_words_in_question, :]
        while complete_array.shape[0] < self.num_of_words_in_question:
            complete_array = np.concatenate((complete_array, self.pad_word_vector), axis=0)
        complete_array = complete_array.reshape(1, self.num_of_words_in_question, self.word_vec_size)
        # print(" questions : shape : " + str(complete_array.shape))
        return complete_array

    def read_sentence_file(self,question_dir_path):
        f = open(os.path.join(question_dir_path, self.closest_sent_file), 'r')
        complete_array = pickle.load(f)
        complete_array = complete_array.reshape(-1,self.word_vec_size)
        while complete_array.shape[0] < self.num_of_words_in_closest_sentence:
            complete_array = np.concatenate((complete_array, self.pad_word_vector), axis=0)
        complete_array = complete_array.reshape(1, self.num_of_words_in_closest_sentence, self.word_vec_size)
        # print(" sentences : shape : " + str(complete_array.shape))
        return complete_array

    def read_correct_ans_file(self,question_dir_path):
        f = open(os.path.join(question_dir_path, self.correct_answer_file), 'r')
        complete_array = pickle.load(f)
        # print(" correct answer : shape : " + str(complete_array.shape))
        complete_array = complete_array.reshape(-1,self.num_of_options_for_quest)
        # print(" correct answer : shape : " + str(complete_array.shape))
        return complete_array

    def print_data_shape_details(self, data_name, x1, x2=None):
        if x2 is None:
            print(data_name + " : shape : " + str(x1.shape))
        else:
            print(data_name + " : train shape : " + str(x1.shape))
            print(data_name + " : test shape : " + str(x2.shape))


    def read_all_vectors(self):
        while(1):
            complete_options_mat = None
            complete_question_mat = None
            complete_sent_mat = None
            complete_correct_ans_mat = None

            number_of_lessons = 0
            for lesson in self.lessons_list:
                l_dir = os.path.join(self.processed_data_path,lesson)
                #print ("Lesson : ",lesson)
                questions_dir = self.get_list_of_dirs(l_dir)
                for question_dir in questions_dir:
                    # file_list = self.get_list_of_files(os.path.join(l_dir,question_dir))
                    question_dir_path = os.path.join(l_dir,question_dir)
                    # print ("Question : ", question_dir)
                    try:
                         options_mat = self.read_options_files(question_dir_path)
                         question_mat = self.read_question_file(question_dir_path)
                         sent_mat = self.read_sentence_file(question_dir_path)
                         correct_ans_mat = self.read_correct_ans_file(question_dir_path)
                    except:
                         print(lesson,question_dir)


                    complete_options_mat = options_mat if complete_options_mat is None else np.concatenate((complete_options_mat, options_mat), axis=0)
                    complete_question_mat = question_mat if complete_question_mat is None else np.concatenate((complete_question_mat, question_mat), axis=0)
                    complete_sent_mat = sent_mat if complete_sent_mat is None else np.concatenate((complete_sent_mat, sent_mat), axis=0)
                    complete_correct_ans_mat = correct_ans_mat if complete_correct_ans_mat is None else np.concatenate((complete_correct_ans_mat, correct_ans_mat), axis=0)
                number_of_lessons+=1
            
                if number_of_lessons>1:
                    if(complete_options_mat is None):
						print("option None : ",lesson)
                    if (complete_question_mat is None):
                        print("question None : ", lesson)
                    if (complete_sent_mat is None):
                        print("doc None : ", lesson)
                    if (complete_correct_ans_mat is None):
                        print("correct answer None : ", lesson)
                    yield [complete_question_mat, complete_sent_mat, complete_options_mat], complete_correct_ans_mat
                    complete_options_mat = None
                    complete_question_mat = None
                    complete_sent_mat = None
                    complete_correct_ans_mat = None
                    number_of_lessons=0

if __name__ == "__main__":

    processed_data_path = "/home/cvpr/akshay/TQA/train/processed_data/one_hot_files"
    # read_training_json = prepare_data(processed_data_path)
    # read_training_json.read_all_vectors()

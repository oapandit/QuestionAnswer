
import os
import pickle
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer


class get_closest_sentences():
    def __init__(self,processed_data_path):
        self.processed_data_path = processed_data_path
        self.raw_text_path = os.path.join(processed_data_path,"text_question_sep_files")
        self.lessons_list = self.get_list_of_dirs(self.raw_text_path)

    def get_list_of_dirs(self,dir_path):
        # dirlist = []
        # for root, dirs, files in os.walk(dir_path):
        #     for dir in dirs:
        #         dirlist.append(dir)
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

    def convert_list_to_string(self,ip_list):
        op_string = ""
        for sent in ip_list:
            op_string+=sent.strip()
        return op_string


    def get_closest_sentences(self,topic_content, question_content, sent_f_handle):

        lemmatizer = WordNetLemmatizer()
        def sub_routine(text,match_coeff):
            for line in doc_lines:
                line = line.translate(None, string.punctuation)
                line_lemma = " ".join(lemmatizer.lemmatize(w) for w in line.split(" "))
                line_list = [i for i in line_lemma.lower().split() if i not in stop]
                if len(set(l_query) & set(line_list)) > match_coeff:
                    text += " " + line.lower()
            return text

        stop = set(stopwords.words('english'))
        query_string = self.convert_list_to_string(question_content)
        doc_string = self.convert_list_to_string(topic_content)


        # lemm_word = lemmatizer.lemmatize(word)


        query_string = query_string.translate(None, string.punctuation)
        doc_lines = sent_tokenize(doc_string)

        l_query = [lemmatizer.lemmatize(i) for i in query_string.lower().split() if i not in stop]

        # l_query = [i for i in query_string.lower().split() if i not in stop]
        # l_query_lemma =  [lemmatizer.lemmatize(i) for i in l_query]

        # [j for i in zip(a, b) for j in i]

        # l_query1 = [j for i in zip(l_query,[lemmatizer.lemmatize(k) for k in l_query]) for j in i]

        text_ = ""
        match_coeff_ = 1
        text = sub_routine(text_,match_coeff_)
        while text_ == text:
            print "Error : less than 1 match"
            match_coeff_ -=1
            text = sub_routine(text_, match_coeff_)
            # break
        sent_f_handle.write(text)
        return text



    def generate_closest_sentence(self):
        topic_fname = "topics.txt"
        question_fname = "Question.txt"
        corr_ans_fname = "correct_answer"
        f_ext = ".txt"
        sent_closest_to_question_fname = "closest_sent.txt"
        for lesson in self.lessons_list:
            print "Lesson : ", lesson
            l_dir = os.path.join(self.raw_text_path, lesson)
            with open(os.path.join(l_dir, topic_fname), "r") as f:
                topic_content = f.readlines()
            questions_dir = self.get_list_of_dirs(l_dir)
            for question_dir in questions_dir:
                print "Question : ", question_dir
                with open(os.path.join(l_dir, question_dir,question_fname), "r") as f:
                    question_content = f.readlines()
                option = 'a'
                while os.path.exists(os.path.join(l_dir, question_dir, option + f_ext)):
                    with open(os.path.join(l_dir, question_dir, option + f_ext), "r") as f:
                        opt = f.readlines()
                    question_content.append(self.convert_list_to_string(opt))
                    option = chr(ord(option) + 1)
                sent_f_handle = open(os.path.join(l_dir,question_dir, sent_closest_to_question_fname), "w")
                self.get_closest_sentences(topic_content,question_content,sent_f_handle)
                sent_f_handle.close()


if __name__ == "__main__":

    processed_data_path = "/home/cvpr/akshay/TQA/val/processed_data"
    read_training_json = get_closest_sentences(processed_data_path)
    read_training_json.generate_closest_sentence()


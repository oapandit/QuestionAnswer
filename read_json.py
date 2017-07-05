
import json
import os
import shutil
import nltk
from nltk.tokenize import word_tokenize

class read_json():
    def __init__(self,json_dir,op_dir = None):
        self.json_dir = json_dir
        if not op_dir:
            op_dir = os.path.join(json_dir,"processed_data","text_question_sep_files")
        if not os.path.exists(op_dir):
            os.makedirs(op_dir)
        self.op_dir = op_dir
        self.json_file_list = self.get_list_of_files(self.json_dir,file_extension=".json")

    def get_list_of_dirs(self,dir_path):
        dirlist = [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]
        dirlist.sort()
        return dirlist

    def get_list_of_files(self,dir_path,file_extension=".json"):
        filelist = []
        for root, dirs, files in os.walk(dir_path):
            for filen in files:
                if filen.endswith(file_extension):
                    filelist.append(filen)
        filelist.sort()
        return filelist

    def read_content(self):
        l_id_tag = 'globalID'
        qs_tag = 'questions'
        ndq_tag = 'nonDiagramQuestions'
        for f in self.json_file_list:
            with open(os.path.join(self.json_dir,f), 'r') as f:
                data = json.load(f)
            for lessons in data:
                lesson_id = lessons[l_id_tag]
                l_dir = os.path.join(self.op_dir,lesson_id)
                if not os.path.exists(l_dir):
                    os.makedirs(l_dir)
                topic_f = "topics.txt"
                topic_f_handle = open(os.path.join(l_dir,topic_f), 'w')
                for topics in lessons['topics']:
                    topic_f_handle.write(lessons['topics'][topics]['content']['text'] + "\n")
                topic_f_handle.close()
                for q_counter,(ndq_id,_) in enumerate(lessons[qs_tag][ndq_tag].iteritems()):
                    # question = lessons[qs_tag][ndq_tag][ndq_id]['beingAsked']['processedText']
                    # question_f = "Q"+str(q_counter+1)+"_"+ndq_id+".txt"
                    # option_list = []
                    # option = 'a'
                    # while lessons[qs_tag][ndq_tag][ndq_id]['answerChoices'].get(option,None) is not None:
                    #     option_list.append(lessons[qs_tag][ndq_tag][ndq_id]['answerChoices'][option]['processedText'])
                    #     option = chr(ord(option) + 1)
                    # question_f_handle = open(os.path.join(l_dir, question_f), 'w')
                    # question_f_handle.write(question+"\n")
                    # for opt in option_list:
                    #     question_f_handle.write(opt+"\n")
                    # correct_answer = lessons[qs_tag][ndq_tag][ndq_id]['correctAnswer']['processedText']
                    # question_f_handle.write(correct_answer + "\n")
                    # question_f_handle.close()

                    q_dir = os.path.join(l_dir,"Q"+str(q_counter+1))
                    os.makedirs(q_dir)
                    question = lessons[qs_tag][ndq_tag][ndq_id]['beingAsked']['processedText']
                    question_f = "Question"+ ".txt"
                    question_f_handle = open(os.path.join(q_dir, question_f), 'w')
                    question_f_handle.write(question+"\n")
                    question_f_handle.close()

                    option_list = []
                    option = 'a'
                    while lessons[qs_tag][ndq_tag][ndq_id]['answerChoices'].get(option,None) is not None:
                        option_list.append(lessons[qs_tag][ndq_tag][ndq_id]['answerChoices'][option]['processedText'])
                        option = chr(ord(option) + 1)
                    option = 'a'
                    for opt in option_list:
                        option_f = option+".txt"
                        option_f_handle = open(os.path.join(q_dir, option_f), 'w')
                        option_f_handle.write(opt)
                        option_f_handle.close()
                        option = chr(ord(option) + 1)

                    correct_answer = lessons[qs_tag][ndq_tag][ndq_id]['correctAnswer']['processedText']
                    corr_f_handle = open(os.path.join(q_dir, "correct_answer.txt"), 'w')
                    corr_f_handle.write(correct_answer + "\n")
                    corr_f_handle.close()


    def sanity_test(self):
        lessons = self.get_list_of_dirs(self.op_dir)
        que_fname = "Question"
        corr_ans_fname = "correct_answer"
        closest_sent_fname = "closest_sent"
        f_ext = ".txt"
        num_of_ques = 0
        wrong_que = 0
        for lesson in lessons:
            print "Lesson : ",lesson
            questions_list = self.get_list_of_dirs(os.path.join(self.op_dir,lesson))
            for que in questions_list:
                num_of_ques +=1
                print "Question : ",que
                que_dir_path = os.path.join(self.op_dir,lesson,que)
                file_list = self.get_list_of_files(que_dir_path,file_extension=".txt")
                if que_fname+f_ext not in file_list :
                    print "Question file doesn't exist"
                if corr_ans_fname+f_ext not in file_list:
                    print "Correct answer file doesn't exist"
                if closest_sent_fname+f_ext not in file_list:
                    print "Closest sentence file doesn't exist"
                with open(os.path.join(que_dir_path, corr_ans_fname+f_ext),"r") as f:
                    correct_answer = f.readlines()
                correct_answer = correct_answer[0].strip().lower()
                option = 'a'
                while ord(option) <= ord(correct_answer):
                    if option+f_ext not in file_list:
                        wrong_que+=1
                        print "Correct answer is : ", correct_answer
                        print "Error : Option file doesn't exist",option+f_ext
                        shutil.rmtree(que_dir_path)
                        break
                    option = chr(ord(option) + 1)
            print 20*"**"
        print "Total Questions : ",num_of_ques
        print "Questions with error : ",wrong_que

    def get_statistics(self):
        lessons = self.get_list_of_dirs(self.op_dir)
        que_fname = "Question"
        corr_ans_fname = "correct_answer"
        closest_sent_fname = "closest_sent"
        f_ext = ".txt"
        num_que_token_list = []
        num_opt_token_list = []
        num_sent_token_list = []
        for lesson in lessons:
            # print "Lesson : ",lesson
            questions_list = self.get_list_of_dirs(os.path.join(self.op_dir,lesson))
            for que in questions_list:
                # print "Question : ",que
                que_dir_path = os.path.join(self.op_dir,lesson,que)
                file_list = self.get_list_of_files(que_dir_path,file_extension=".txt")
                with open(os.path.join(que_dir_path, que_fname+f_ext),"r") as f:
                    ques = f.readlines()
                num_of_tokens_in_que = 0
                # print ques
                for sent in ques:
                    # print sent
                    words = word_tokenize(sent)
                    num_of_tokens_in_que+=len(words)
                num_que_token_list.append(num_of_tokens_in_que)

                with open(os.path.join(que_dir_path, closest_sent_fname+f_ext),"r") as f:
                    sents = f.readlines()
                num_of_tokens_in_sents = 0
                # print ques
                for sent in sents:
                    # print sent
                    words = word_tokenize(sent)
                    num_of_tokens_in_sents+=len(words)
                num_sent_token_list.append(num_of_tokens_in_sents)
                if num_of_tokens_in_sents == 0:
                    print "Lesson : ",lesson
                    print "Question : ",que


                with open(os.path.join(que_dir_path, corr_ans_fname+f_ext),"r") as f:
                    correct_answer = f.readlines()
                correct_answer = correct_answer[0].strip().lower()
                option = 'a'
                while os.path.exists(os.path.join(que_dir_path,option + f_ext)):
                    with open(os.path.join(que_dir_path, option + f_ext), "r") as f:
                        opt = f.readlines()
                    num_of_tokens_in_opt = 0
                    for sent in opt:
                        words = word_tokenize(sent)
                        num_of_tokens_in_opt += len(words)
                    num_opt_token_list.append(num_of_tokens_in_opt)
                    option = chr(ord(option) + 1)
            # print 20*"**"
        que_lenth_dict = nltk.FreqDist(x for x in num_que_token_list)
        opt_lenth_dict = nltk.FreqDist(x for x in num_opt_token_list)
        sent_lenth_dict = nltk.FreqDist(x for x in num_sent_token_list)

        print "Question length info"
        for k, v in que_lenth_dict.most_common(50):
            print str(k), str(v)
        print "Max question length : ",max(num_que_token_list)

        print "Option length info"
        for k, v in opt_lenth_dict.most_common(50):
            print str(k), str(v)
        print "Max Option length : ", max(num_opt_token_list)

        print "Closest sentence info"
        for k, v in sent_lenth_dict.most_common(1000):
            print str(k), str(v)
        print "Max Closest sentence length : ", max(num_sent_token_list)


if __name__ == "__main__":

    json_path = "/home/cvpr/akshay/TQA/train"
    read_training_json = read_json(json_path)
    # read_training_json.read_content()
    # read_training_json.sanity_test()
    read_training_json.get_statistics()


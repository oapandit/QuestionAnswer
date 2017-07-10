import os
import numpy as np
import json
from read_json import read_json



class generate_result():

    def __init__(self,read_val_data):
        self.read_val_data = read_val_data
        pass


    def predict_options_one_by_one(self,model,is_test_data = False):
        acc_pred = 0
        total_pred = 0
        options_list = ["a", "b", "c", "d", "e", "f", "g"]
        f_l = ["L_0007","L_0021"]

        quest_ans_dict = {}
        for lesson in self.read_val_data.lessons_list:
#        for lesson in f_l:
            l_dir = os.path.join(self.read_val_data.processed_data_path, lesson)
            print ("Lesson : ",lesson)
            questions_dir = self.read_val_data.get_list_of_dirs(l_dir)
            for question_dir in questions_dir:
                # print 20*"**"
                question_dir_path = os.path.join(l_dir, question_dir)
                # print ("Question : ", question_dir)
                options_mat,max_options = self.read_val_data.read_options_files(question_dir_path)
                question_mat = self.read_val_data.read_question_file(question_dir_path)
                sent_mat = self.read_val_data.read_sentence_file(question_dir_path)

                # print "max option",max_options
                if max_options == 1 :
                    max_options = self.read_val_data.num_of_options_for_quest - 1
                pred_options_arr = model.predict([question_mat, sent_mat, options_mat])

                #Get maximum only from specified options not from complete list
                pred_opt = np.argmax(pred_options_arr[0,:max_options])

                if not is_test_data:
                    correct_ans_mat = self.read_val_data.read_correct_ans_file(question_dir_path)
                    correct_ans_mat[np.where(correct_ans_mat == -1)] = 0
                    corr_opt = np.where(correct_ans_mat == 1)[1][0]
                    if corr_opt == pred_opt:
                        acc_pred+=1
                    total_pred+=1

                quest_ans_dict[question_dir] = options_list[pred_opt]

        if not is_test_data:
            print "Total questions : ",total_pred
            print "Accurate predictions : ",acc_pred
            print "Accuracy : ",float(acc_pred)/total_pred

        self.generate_result_file(quest_ans_dict)





    def generate_result_file(self,quest_ans_dict):

        res_file_path = os.path.join(os.path.dirname(self.read_val_data.processed_data_path),"tqa_val.json")
        res_f_handle = open(res_file_path, "w")

        json_path = "/home/cvpr/akshay/TQA/val"
        read_val_json = read_json(json_path)

        ndq_ids_list = read_val_json.get_questions_id()

        for ndq_id in ndq_ids_list:
            if quest_ans_dict.get(ndq_id,None) is None:
                print "Question not present",ndq_id
                quest_ans_dict[ndq_id] = 'a'

        ques_ans_json_data = json.dumps(quest_ans_dict, indent=4)
        res_f_handle.write(ques_ans_json_data)
        res_f_handle.close()



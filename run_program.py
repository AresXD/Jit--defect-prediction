import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-gid', type=str, default='0')
parser.add_argument('-p', type=str, default='')
parser.add_argument('-raw', action='store_true')
parser.add_argument('-sub', type=str, default='')
parser.add_argument('-epoch', type=str, default='3')
parser.add_argument('-original_data',action='store_true')
parser.add_argument('-all_data',action='store_true')
parser.add_argument('-cross_data',action='store_true')

parser.add_argument('-train_model', action='store_true')
parser.add_argument('-pred_model', action='store_true')
parser.add_argument('-BERT',action='store_true')
parser.add_argument('-CodeBERT',action='store_true')
parser.add_argument('-GPT3',action='store_true')


task = 'CUDA_VISIBLE_DEVICES={} python main.py -train -train_data data/{}_data/{}_train_changed.pkl -dictionary_data ./{}_dict.pkl -save-dir snapshot/{}/CodeBERT/model -num_epochs {} -model_type={}'
raw_task = 'CUDA_VISIBLE_DEVICES={} python main.py -train -train_data data_all/{}/cc2vec/{}_train.pkl -dictionary_data data_all/{}/cc2vec/{}_dict.pkl -save-dir snapshot/{}/CodeBERT/model -num_epochs {} -model_type={}'

raw_predict = "CUDA_VISIBLE_DEVICES={} python main.py -predict -pred_data data_all/{}/cc2vec/{}_test_raw.pkl -dictionary_data data_all/{}/cc2vec/{}_dict.pkl -load_model snapshot/{}/raw/epoch_{}.pt -model_type={}"
predict = "CUDA_VISIBLE_DEVICES={} python main.py -predict -pred_data data/{}_data/{}_test_changed.pkl -dictionary_data ./{}_dict.pkl -load_model snapshot/{}/CodeBERT/model/epoch_{}.pt -model_type={}"


def run_projects(projects, args):
    for project in projects:
        args.p = project
        run_one(args)


def rm_model(args):
    projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
    for project in projects:
        for i in range(45):
            cmd = "rm snapshot/{}/{}/epoch_{}.pt".format(project, args.sub, i)
            print(cmd)
            os.system(cmd)


def run_one(args):
    project = args.p
    path = args.p
    cmd = ""
    if args.sub:
        path = args.p + '/' + args.sub
    if args.raw:
        if args.train_model:
            cmd = raw_task.format(args.gid, path, project, path, project, path,
                                  args.epoch,args.model_type)
        elif args.pred_model:
            cmd = raw_predict.format(args.gid, path, project, path, project,
                                     path, args.epoch,args.model_type)
    else:
        if args.train_model:
            if args.BERT:
                cmd = task.format(args.gid, path, project, path,  path,
                              args.epoch,"BERT")
            elif args.CodeBERT:
                cmd = task.format(args.gid, path, project, path, project, path,
                                  args.epoch, "CodeBERT")
            elif args.GPT3:
                cmd = task.format(args.gid, path, project, path, project, path,
                                  args.epoch, "GPT3")

        elif args.pred_model:
            if args.BERT:
                cmd = predict.format(args.gid, path, project, path, project, path,
                              args.epoch,"BERT")
            elif args.CodeBERT:
                cmd = predict.format(args.gid, path, project, path, project, path,
                                  args.epoch, "CodeBERT")
            elif args.GPT3:
                cmd = predict.format(args.gid, path, project, path, project, path,
                                  args.epoch, "GPT3")


        if args.sub == "cam":
            cmd = cmd + " -cam"
    if not cmd:
        print("Please select a task by: python run_program.py $Task")
    print(cmd)
    os.system(cmd)


def original_data(args):
    args.epoch = '5'
    projects = ['qt']

    run_projects(projects, args)
    # projects=['op']
    # run_projects(projects,args)


def all_data(args):
    projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
    run_projects(projects, args)




def cross_data():
    cmd = [
        'python main.py -train -train_data ./data/qt_data/qt_train_changed.pkl -dictionary_data ./qt_dict.pkl -save-dir snapshot/op/cross/codeBERT/model -num_epochs 3 -model_type=CodeBERT -cross -proj=op','python main.py -train -train_data ./data/op_data/op_train_changed.pkl -dictionary_data ./op_dict.pkl -save-dir snapshot/qt/cross/codeBERT/model -num_epochs 3 -model_type=CodeBERT -cross -proj=qt']
    # for cmdline in cmd:
    #     os.system(cmdline)
    os.system(cmd[1])
    # cmdBERT = ['python main.py -predict -pred_data data/qt_data/qt_test_changed.pkl -dictionary_data ./qt_dict.pkl -load_model snapshot/qt/cross/codeBERT/model/epoch_3_step_1650.pt  -model_type=BERT']
    # os.system(cmdBERT[0])



def train_code():
    cmd=['python main.py -train -train_data data/qt_data/qt_train_changed.pkl -dictionary_data ./qt_dict.pkl -save-dir snapshot/qt/BERT/model -num_epochs 3 -model_type=BERT','python main.py -train -train_data data/op_data/op_train_changed.pkl -dictionary_data ./op_dict.pkl -save-dir snapshot/op/GPT3/model -num_epochs 3 -model_type=GPT3']
    for cmdline in cmd:
        os.system(cmdline)
def test_code():
    cmdBERT=['python main.py -predict -pred_data data/qt_data/qt_test_changed.pkl -dictionary_data ./qt_dict.pkl -load_model snapshot/qt/GPT3/model/epoch_3_step_2550.pt  -model_type=GPT3','python main.py -predict -pred_data data/op_data/op_test_changed.pkl -dictionary_data ./op_dict.pkl -load_model snapshot/op/GPT3/model/epoch_3_step_1200.pt  -model_type=GPT3']
    cmdCodeBERT=['python main.py -predict -pred_data data/qt_data/qt_test_changed.pkl -dictionary_data ./qt_dict.pkl -load_model snapshot/qt/CodeBERT/model/epoch_3_step_2550.pt  -model_type=CodeBERT','python main.py -predict -pred_data data/op_data/op_test_changed.pkl -dictionary_data ./op_dict.pkl -load_model snapshot/op/CodeBERT/model/epoch_3_step_1200.pt  -model_type=CodeBERT']
    os.system(cmdBERT[0])
    for cmd in cmdBERT:
        os.system(cmd)
    # for cmd in cmdCodeBERT:
    #     os.system(cmd)



if __name__ == "__main__":
    # args = parser.parse_args()
    # if args.original_data:
    #     original_data(args)
    # elif args.cross_data:
    #     cross_data(args)
    cross_data()
    # else:
    #     print("Please select a correct command.")
    #     # run_one(args)
    # train_code()
    # test_code()

import argparse
import os

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memo",default="HC_DT",type=str,required=False)
    parser.add_argument("--pretrained_path",default="bert-base-uncased",type=str,required=False)
    parser.add_argument("--do_train",default=True,type=bool,required=False)
    parser.add_argument("--do_dev",default=True,type=bool,required=False)
    parser.add_argument("--do_test",default=True,type=bool,required=False)
    parser.add_argument('--num_labels',default=3,type=int,required=False)
    parser.add_argument("--resume_model_path",default="bert_HC_out/model",type=str,required=False)
    parser.add_argument("--resume_model",default=False,type=bool,required=False)
    parser.add_argument("--data_dir",default="./raw_data/",type=str,required=False)
    parser.add_argument("--train_name",default="hc.raw",type=str,required=False)
    parser.add_argument("--dev_name",default="hillary_test_data.txt",type=str,required=False)
    parser.add_argument("--test_name",default="dt.raw",type=str,required=False)
    parser.add_argument("--max_seq_len",default=100,type=int,required=False)
    parser.add_argument("--batch_size",default=4,type=int,required=False)
    parser.add_argument("--eval_batch_size",default=4,type=int,required=False)
    parser.add_argument("--epochs",default=15,type=int,required=False)
    parser.add_argument("--eval_steps",default=50,type=int,required=False)
    parser.add_argument("--out_dir",type=str,default="./bert_HC_out")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=1e-3, type=float,
                        help="L2 regularization.")
    parser.add_argument("--seed",default=42,type=int,required=False)
    parser.add_argument("--do_lower_case",default=True,type=bool,required=False)
    config = parser.parse_args()
    return config
base_config = config()

if __name__=="__main__":
    print(os.path.isdir("./pretrained_models/bert_base"))

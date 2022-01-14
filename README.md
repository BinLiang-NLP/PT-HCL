step 0 :install requrements and put glove.42B.300d.txt in current directory

step 1 : get topic words for 3 datasets by gensim.models.LdaModel and put them in './augment_data/sentence/'
run run_pretext_task.sh with istest=0 to get a overfitting model ,put it in your 'state_dict_path'
then run run_pretext_task.sh with istest=1 to do pretext task

step 2: run run_zssd.sh to do ZSSD


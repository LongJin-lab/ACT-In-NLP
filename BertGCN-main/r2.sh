# nohup python finetune_bert.py --dataset R8 --device 2 --opt 1 > tmp1.txt &
bert_init='roberta-base'  #choices=['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased']
dataset=0 #0:20ng   1:R8   2:R52   3:ohsumed   4:mr
gcn_model='gat' # gcn gat
device=0


for ((k=3;k<4;k++));do #opt
    for ((j=3;j<5;j++));do #重复试验次数
        #20ng
        # nohup python train_bert_gcn.py --gcn_model ${gcn_model} --bert_init ${bert_init} --dataset 20ng --device ${device} --opt ${k} > ./log_${gcn_model}/${bert_init}/20ng/opt_${k}_${j}.txt 2>&1
        # #R8
        # nohup python train_bert_gcn.py --gcn_model ${gcn_model} --bert_init ${bert_init} --dataset R8 --device ${device} --opt ${k} > ./log_${gcn_model}/${bert_init}/R8/opt_${k}_${j}.txt 2>&1
        #R52
        nohup python train_bert_gcn.py --gcn_model ${gcn_model} --bert_init ${bert_init} --dataset R52 --device ${device} --opt ${k} > ./log_${gcn_model}/${bert_init}/R52/opt_${k}_${j}.txt 2>&1
        # #ohsumed
        # nohup python train_bert_gcn.py --gcn_model ${gcn_model} --bert_init ${bert_init} --dataset ohsumed --device ${device} --opt ${k} > ./log_${gcn_model}/${bert_init}/ohsumed/opt_${k}_${j}.txt 2>&1
        # #mr
        # nohup python train_bert_gcn.py --gcn_model ${gcn_model} --bert_init ${bert_init} --dataset mr --device ${device} --opt ${k} > ./log_${gcn_model}/${bert_init}/mr/opt_${k}_${j}.txt 2>&1
    done;
done;
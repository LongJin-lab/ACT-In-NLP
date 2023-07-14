# nohup /home/ps/anaconda3/envs/qc/bin/python run.py > ./log/tmp.txt &

#!/bin/bash

# gamma=('0.1' '0.1' '0.1' '0.1' '0.2' '0.2' '0.2' '0.05' '0.05' '0.05' '0.3' '0.3' '0.3')
# delta=('10' '20' '30' '5' '10' '20' '30' '5' '40' '30' '4.5' '10' '20')

gamma=('0.2' '0.2' '0.05' '0.3' '0.1')
delta=('10' '20' '40' '4.5' '20')





for ((k=1;k<2;k++));do #opt
    for ((i=1;i<2;i++));do #gamma delta
        for ((j=2;j<3;j++));do #重复试验次数
            nohup /home/ps/anaconda3/envs/qc/bin/python run.py  --opt ${k} --gamma ${gamma[i]} --delta ${delta[i]} --run_num ${j} 2>&1;
        done;
    done;
done;


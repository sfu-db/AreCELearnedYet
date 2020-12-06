#!/bin/bash
#################### Dynamic exp
### Do not use 0 as random seed, becasue our Postgres sets random seed as 1/seed
log_path='log'
exp_num=1
for (( i=1; i < 1+$exp_num; ++i ))
do
    for dataset in 'census13' 'forest10' 'power7' 'dmv11'
    do
        for up in 'cor' #'skew' 
        do
            ## MSCN
            just dynamic-mscn-${dataset} ${dataset} 'original' 'base' ${up} '0.2' '10000' "$i" >${log_path}/${dataset}/mscn_${up}-exp${i}.out 2>&1

            ## lw retrain
            just dynamic-lw-tree-${dataset}-retrain ${dataset} 'original' 'base' ${up} '0.2' '8000' "$i" >>${log_path}/${dataset}/lwtree_${up}-exp${i}.out 2>&1
            just dynamic-lw-nn-${dataset}-retrain ${dataset} 'original' 'base' ${up} '0.2' '16000' "$i" '500' >>${log_path}/${dataset}/lwnn_eq500_${up}-exp${i}.out 2>&1
            just dynamic-lw-nn-${dataset}-retrain ${dataset} 'original' 'base' ${up} '0.2' '16000' "$i" '100' >${log_path}/${dataset}/lwnn_eq100_${up}-exp${i}.out 2>&1

            ## Postgres
            just dynamic-postgres-${dataset} ${dataset} 'original' 'base' ${up} '0.2' "$i" >${log_path}/${dataset}/postgres_${up}-exp${i}.out 2>&1

            ## MySQL
            just dynamic-mysql-${dataset} ${dataset} 'original' 'base' ${up} '0.2' "$i" >${log_path}/${dataset}/mysql_${up}-exp${i}.out 2>&1

            ## Naru
            just dynamic-naru-${dataset} ${dataset} 'original' 'base' ${up} '0.2' "$i" '1' >>${log_path}/${dataset}/naru_eq1_${up}-exp${i}.out 2>&1
            just dynamic-naru-${dataset} ${dataset} 'original' 'base' ${up} '0.2' "$i" '7' >${log_path}/${dataset}/naru_eq7_${up}-exp${i}.out 2>&1
            just dynamic-naru-${dataset} ${dataset} 'original' 'base' ${up} '0.2' "$i" '15' >${log_path}/${dataset}/naru_eq15_${up}-exp${i}.out 2>&1
            ## QuickSel
            just dynamic-quicksel ${dataset} 'original' 'base' ${up} '0.2' "$i" >${log_path}/${dataset}/quicksel_${up}-exp${i}.out 2>&1

            ## DeepDB
            just dynamic-deepdb-${dataset} ${dataset} 'original' 'base' ${up} '0.2' "$i" >${log_path}/${dataset}/deepdb_${up}-exp${i}.out 2>&1
        done
    done
done


# epoch vs accuracy
for (( i=1; i < 1+$exp_num; ++i ))
do
    for dataset in 'census13' 'forest10' 'power7' 'dmv11'
    do
        for up in 'cor' 'skew' 
        do
            ## lwNN
            for ep in '100' '200' '300' '400' '500'
            do
                just dynamic-lw-nn-${dataset}-retrain ${dataset} 'original' 'base' ${up} '0.2' '16000' "$i" $ep >${log_path}/${dataset}/lwnn_eq${ep}_${up}-exp${i}.out 2>&1
            done
            ## Naru
            for ep in '1' '5' '10' '15' '20'
            do
                just dynamic-naru-${dataset} ${dataset} 'original' 'base' ${up} '0.2' "$i" $ep >${log_path}/${dataset}/naru_eq${ep}_${up}-exp${i}.out 2>&1
            done
        done
    done
done


echo `date` "All Finished!"

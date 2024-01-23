# go to src source
cd src
# loop trough all files
for lam in 10 20 30; do
    for i in 0 1 2 3; do
        screen -dm -S "train_${lam}_${i}" python model_training_ppo.py --lam_r="$lam" --device="0"
        sleep 10
    done
done
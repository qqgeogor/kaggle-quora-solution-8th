
model_name="bimpm_0"
mode='predict'
model_load_dir='./data/models/bimpm_0/0/'
train_file='../fold_data/fold_0_train.pkl'
test_file='../fold_data/fold_0_test.pkl'
batch_size=1024
num_epochs=10
num_sentence_words=20
val_period=500
log_period=500
run_id=0
is_load=False
early_stopping_patience=1

python run_bimpm.py --model_name $model_name --mode $mode --model_load_dir $model_load_dir --is_load $is_load --train_file $train_file --test_file $test_file --num_epochs $num_epochs --num_sentence_words $num_sentence_words --val_period $val_period --log_period $log_period --run_id $run_id --batch_size $batch_size --early_stopping_patience $early_stopping_patience

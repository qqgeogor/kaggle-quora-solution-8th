
model_name="siamse_4"
mode='train'
model_use='siamse'
model_load_dir='./data/models/siamse_4/0/'
train_file='../fold_data/fold_4_train.pkl'
test_file='../fold_data/fold_4_test.pkl'
batch_size=512
num_epochs=5
num_sentence_words=30
rnn_output_mode='last'
val_period=1000
log_period=200
run_id=0
is_load=False
early_stopping_patience=1

python run_models.py --model_name $model_name --mode $mode --model_use $model_use --model_load_dir $model_load_dir --is_load $is_load --train_file $train_file --test_file $test_file --num_epochs $num_epochs --num_sentence_words $num_sentence_words --rnn_output_mode $rnn_output_mode --val_period $val_period --log_period $log_period --run_id $run_id --batch_size $batch_size --early_stopping_patience $early_stopping_patience

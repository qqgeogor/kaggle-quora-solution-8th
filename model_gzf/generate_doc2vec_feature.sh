python generate_doc2vec_data.py

git clone git@github.com:mesnilgr/iclr15.git
cp ../../iclr15/scripts/word2vec.c .
gcc word2vec.c -o word2vec -lm -pthread -O3 -march=native -funroll-loops

shuf all_sentences.txt > all_sentences_shuf.txt
time ./word2vec -train all_sentences.txt -output vectors.txt -cbow 0 -size 100 -window 10 -negative 5 -hs 1 -sample 1e-3 -threads 12 -binary 0 -iter 20 -min-count 1 -sentence-vectors 1

grep '_\*' vectors.txt | sed -e 's/_\*//' | sort -n > sentence_vectors.txt

head -n 2750086  sentence_vectors.txt > sentence_vectors_q1.txt
tail -n 2750086  sentence_vectors.txt > sentence_vectors_q2.txt

python generate_doc2vec_feature.py


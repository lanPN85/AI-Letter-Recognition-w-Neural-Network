echo 'Removing old process file'
rm process.txt
rm test.txt
rm train.txt
rm cv.txt

python preprocess.py A 0
python preprocess.py B 1
python preprocess.py C 2
python preprocess.py D 3
python preprocess.py E 4
python preprocess.py F 5
python preprocess.py G 6
python preprocess.py H 7
python preprocess.py I 8
python preprocess.py J 9
python preprocess.py K 10
python preprocess.py L 11
python preprocess.py M 12
python preprocess.py N 13
python preprocess.py O 14
python preprocess.py P 15
python preprocess.py Q 16
python preprocess.py R 17
python preprocess.py S 18
python preprocess.py T 19
python preprocess.py U 20
python preprocess.py V 21
python preprocess.py W 22
python preprocess.py X 23
python preprocess.py Y 24
python preprocess.py Z 25

python datagen.py
echo 'Removing old process file'
rm process.txt
rm test.txt
rm train.txt
rm cv.txt

python preprocess.py A 0
python preprocess.py B 1

python preprocess.py C 28
python preprocess.py c 28

python preprocess.py D 3
python preprocess.py E 4
python preprocess.py F 5
python preprocess.py G 6
python preprocess.py H 7
python preprocess.py I 8
python preprocess.py J 9

python preprocess.py K 36
python preprocess.py k 36

python preprocess.py L 11
python preprocess.py M 12
python preprocess.py N 13

python preprocess.py O 40
python preprocess.py o 40

python preprocess.py P 41
python preprocess.py p 41

python preprocess.py Q 16
python preprocess.py R 17
python preprocess.py S 18
python preprocess.py T 19

python preprocess.py U 46
python preprocess.py u 46

python preprocess.py V 47
python preprocess.py v 47

python preprocess.py W 48
python preprocess.py w 48

python preprocess.py X 49
python preprocess.py x 49

python preprocess.py Y 24

python preprocess.py Z 51
python preprocess.py z 51

python preprocess.py a 26
python preprocess.py b 27
python preprocess.py d 29
python preprocess.py e 30
python preprocess.py f 31
python preprocess.py g 32
python preprocess.py h 33
python preprocess.py i 34
python preprocess.py j 35
python preprocess.py l 37
python preprocess.py m 38
python preprocess.py n 39
python preprocess.py q 42
python preprocess.py r 43
python preprocess.py s 44
python preprocess.py t 45
python preprocess.py y 50


python datagen.py
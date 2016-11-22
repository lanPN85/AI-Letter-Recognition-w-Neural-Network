dir=$1;
cd $dir
flist=(`ls`);
for fn in ${flist[*]}
do
	name="${fn%.*}";
	ext="${fn##*.}";
	dot=".";

	echo "Rotating $fn [1]"
	suffix="_r1";
	new_name="$name$suffix$dot$ext";
	convert $fn -rotate 5 $new_name

	echo "Rotating $fn [2]"
	suffix="_r2";
	new_name="$name$suffix$dot$ext";
	convert $fn -rotate -5 $new_name

	echo "Rotating $fn [3]"
	suffix="_r3";
	new_name="$name$suffix$dot$ext";
	convert $fn -rotate 10 $new_name

	echo "Rotating $fn [4]"
	suffix="_r4";
	new_name="$name$suffix$dot$ext";
	convert $fn -rotate -10 $new_name

	echo "Rotating $fn [5]"
	suffix="_r5";
	new_name="$name$suffix$dot$ext";
	convert $fn -rotate 15 $new_name

	echo "Rotating $fn [6]"
	suffix="_r6";
	new_name="$name$suffix$dot$ext";
	convert $fn -rotate -15 $new_name

done
for seed in $(seq 0 50); do

	qsub -v "EXEC_CMD=/home/cluster/applications/anaconda2/bin/python2.7 kar.py $seed 0 0.0 0.0"  submit.sh
	#qsub -v "EXEC_CMD=/home/cluster/applications/anaconda2/bin/python2.7 kar.py $seed 1"  submit.sh

	for alphapar in -0.2 -0.1 0.0 0.1 0.2 ; do
		for betapar in -0.2 -0.1 0.0 0.1 0.2 ; do
			echo "kar.py $seed 1 $alphapar $betapar"
			qsub -v "EXEC_CMD=/home/cluster/applications/anaconda2/bin/python2.7 kar.py $seed 1 $alphapar $betapar"  submit.sh
		done
	done
done
qwatch

#! /usr/bin/env bash

## Usage:
#       ./basher.sh EXP_DIR TASK ENC DEC ATTN

## Values
EXPDIR=$1
TASK=$2
ENC=$3
DEC=$4
ATTN=$5

MAIL='jackson.petty@yale.edu'
EXPPATH="$EXPDIR/$TASK/$ENC-$DEC-$ATTN" 
if NUM=$(find $EXPPATH/ -maxdepth 0 -type d | wc -l | tr -d '[:space:]') ; then
	echo ''
else
	NUM=1
fi


# echo "Attention:"
# echo $ATTN

if [ "$ATTN" = "None" ]; then
	ATTNCMD=""
else
	ATTNCMD="-a $ATTN"
fi

# echo "Attention command:"
# echo $ATTNCMD

if (( $# > 5 )); then
	FCMD="-f ${@:6}"
else
	FCMD=""
fi

# echo "Files command:"
# echo $FCMD

echo "Writing to $TASK-$ENC-$DEC-$ATTN.sh"

cat > "$TASK-$ENC-$DEC-$ATTN.sh" << EOF1
#! /usr/bin/env bash
#SBATCH --job-name=$TASK-$ENC-$DEC-$ATTN-$NUM
#SBATCH --ntasks=1 --nodes=1
#SBATCH --mem-per-cpu=500
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=$MAIL
#SBATCH --output="$TASK-$ENC-$DEC-$ATTN-$NUM.out"

export PATH=\$HOME/anaconda3/bin:\$PATH
python main.py train -t $TASK $ATTNCMD -E $EXPDIR -e $ENC -d $DEC -ep 100 $FCMD
EOF1

echo "Running sbatch on $TASK-$ENC-$DEC-$ATTN.sh"

sbatch $TASK-$ENC-$DEC-$ATTN.sh

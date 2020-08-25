#! /usr/bin/env bash

## Usage:
#       ./basher.sh EXP_DIR TASK ENC DEC ATTN

## Values
EXPDIR=$1
TASK=$2
ENC=$3
DEC=$4
ATTN=$5

MAIL='example@example.com'
EXPPATH="$EXPDIR/$TASK/$ENC-$DEC-$ATTN" 
if NUM=$( { find $EXPPATH/ -maxdepth 1 -type d | wc -l | tr -d '[:space:]' } 2>&1 ) ; then
	echo "we got one"
else
	NUM=1
fi

if [ "$ATTN" = "None" ]; then
	ATTNCMD=""
else
	ATTNCMD="-a $ATTN"
fi

echo "Creating $TASK-$ENC-$DEC-$ATTN.sh"

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
python main.py train -t $TASK $ATTNCMD -E $EXPDIR -e $ENC -d $DEC -ep 100
EOF1

echo "Running sbatch on $TASK-$ENC-$DEC-$ATTN.sh"
echo "Output is printed to $TASK-$ENC-$DEC-$ATTN-$NUM.out"

sbatch $TASK-$ENC-$DEC-$ATTN.sh

import random

outputfile = "identity.tsv"

subj = ['Alice', 'Bob', 'Claire', 'Daniel', 'Eliza', 'Frank']

with open(outputfile, 'w') as f:
  for i in range(1000):
    source = f"{random.choice(subj)} {random.choice(subj)} {random.choice(subj)} {random.choice(subj)}"
    f.write(f'{source}\tiden\t{source}\n')

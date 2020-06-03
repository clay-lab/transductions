import csv
import torchtext.data as tt 


# Create some fields
src_field = tt.Field()
transform_field = tt.Field()
trg_field = tt.Field()
fields = [("src", src_field), ("transform", transform_field), 
		  ("trg", trg_field)]

# Create dataset
dataset = tt.TabularDataset("test_file2.csv", "csv", fields, skip_header=True)
for name, field in fields:
	field.build_vocab(dataset)

# Test datasets
for batch in tt.BucketIterator(dataset, 5):
	print(batch)

	print("\nsrc:")
	for j in range(5):
		print([src_field.vocab.itos[i] for i in batch.src[:, j]])
	print(batch.src)

	print("\ntransform:")
	print([transform_field.vocab.itos[i] for i in batch.transform[0]])
	print(batch.transform)

	print("\ntrg:")
	for j in range(5):
		print([trg_field.vocab.itos[i] for i in batch.trg[:, j]])
	print(batch.trg)
	break
	
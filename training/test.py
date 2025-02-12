from datasets import load_dataset

ds = load_dataset("Atipico1/NQ-colbert")

test_data = ds["test"]
print(test_data[0])

reference =""
for i in range(len(test_data[0]['ctxs'])):
    reference+=test_data[0]['ctxs'][i]['text']
print(reference)
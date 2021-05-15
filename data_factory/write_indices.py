BASE_PATH = r"C:\Users\Lodewijk\Desktop\scriptie2\GNNdocs\clean_data\indices\r8_full"

# pre defined train is 5485
# so val is - 10%
train_indices = range(4980)
val_indices = range(4980, 5485)
test_indices = range(5485, 7674)


with open(BASE_PATH + "_train.txt", "w") as open_f:
    for i in train_indices:
        open_f.write(str(i)+",")
with open(BASE_PATH + "_val.txt", "w") as open_f:
    for i in val_indices:
        open_f.write(str(i)+",")
with open(BASE_PATH + "_test.txt", "w") as open_f:
    for i in test_indices:
        open_f.write(str(i)+",")
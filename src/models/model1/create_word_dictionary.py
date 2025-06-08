unique_words = []

path = r"D:/Sujays documents & files/MS/IDP/Uni Acceptance Letters/DePaul/Classes/Quarter 6/SE489_MLOps/Project/citegraph/src/data/Cora/CoRA_Raw/"

with open(path + 'words_dictionary.txt', 'r', encoding='utf-8') as f:
    for line in f:
        word = line.split()[0]
        if len(unique_words) < 1433 and word not in unique_words:
            unique_words.append(word.lower())

with open(path + 'final_words_dictionary.txt', 'w', encoding='utf-8') as f:
    for w in unique_words:
        f.write(f"{w}\n")
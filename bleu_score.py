# two references for one document
from nltk.translate.bleu_score import corpus_bleu

references = [[['this', 'is', 'a', 'test'], ['this', 'is', 'test']]]
candidates = [['this', 'is', 'a', 'test']]

score = corpus_bleu(references, candidates)
print(score)

# 1-gram individual BLEU
from nltk.translate.bleu_score import sentence_bleu

reference = [['this', 'is', 'a', 'small', 'test']]
candidate = ['this', 'is', 'a', 'test']

score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
print(score)

# 2-gram individual BLEU
score = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))
print(score)

# 3-gram individual BLEU
score = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0))
print(score)

# 4-gram individual BLEU
score = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1))
print(score)

# 4-gram cumulative BLEU (default)
score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
print(score)

# 3-gram cumulative BLEU
score = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
print(score)

# 2-gram cumulative BLEU
score = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
print(score)

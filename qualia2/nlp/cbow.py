from ..functions import sum, mean
from ..nn.modules import Module, Embedding

class CBOW(Module):
    ''' Continuous Bag of Words (CBOW) Model\n
    Args:
        vocab_size (int): vocabulary size
        embedding_dim (int): embedding size
    '''
    def __init__(self, vocab_size, embedding_dim=100):
        super().__init__()
        self.word_vec = Embedding(vocab_size, embedding_dim)
        self.out = Embedding(vocab_size, embedding_dim)

    def forward(self, ctx, trg):
        embed = mean(self.word_vec(ctx), axis=1)
        score = sum(embed * self.out(trg).squeeze(axis=1), axis=1)
        return score, trg
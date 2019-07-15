from ..functions import mean, softmax
from ..nn.modules import Module, Linear, Embedding

class CBOW(Module):
    ''' Continuous Bag of Words Model\n
    Args:
        vocab_size (int): vocabulary size
        embedding_dim (int): embedding size
    '''
    def __init__(self, vocab_size, embedding_dim=100):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.linear = Linear(embedding_dim, vocab_size)

    def forward(self, input):
        embed = mean(self.embedding(input), axis=1)
        if self.training:
            return self.linear(embed)
        else:
            out = softmax(self.linear(embed))
            return out
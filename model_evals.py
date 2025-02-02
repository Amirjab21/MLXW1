from models import CBOWTrainer
from Tokenizer import Tokenizer
import torch
from models import CBOW
def find_closest_words(model, word, n=5, word_to_id=None, id_to_word=None):
    # tokenizer = Tokenizer()
    # word_to_id, id_to_word = tokenizer.get_lookup_table(vocab_file_path)
    trainer = CBOWTrainer(
        embedding_size=200,
        context_size=2,
        batch_size=512,
        learning_rate=0.01,
        num_epochs=5,
        min_freq=8,
        min_freq_priority=5,
        device="mps",
        word_to_id=word_to_id,
        id_to_word=id_to_word
    )
    model.load_state_dict(torch.load('./models/weights.pt'))
    model.eval()
    results = trainer.find_similar_words(model, word, n)
    return results

def test_model(file_path = 'data_outputs/vocab.json', embedding_dim = 200):
    tokenizer = Tokenizer()
    word_to_id, id_to_word = tokenizer.get_lookup_table(file_path)
    vocab_size = len(word_to_id)

    model = CBOW(vocab_size, embedding_dim).to("mps")
    results = find_closest_words(model, "dog", 5, word_to_id, id_to_word)
    print("top results for dog:")
    for result in results:
        print(result)



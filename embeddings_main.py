from pathlib import Path
import pandas as pd
from Tokenizer import Tokenizer
import os
import torch
from models import CBOWTrainer, CBOW
from torch.utils.data import DataLoader
from models import CBOWDataset
import wandb

vocab_file_path = 'data_outputs/vocab.json'
def main(subset = False, vocab_file_path = vocab_file_path):
    print("building vocab")
    data_path = Path('data')
    
    stories = pd.read_csv('data/titles_only.csv')
    if subset:
        stories = stories.head(100000)
    titles_string = stories['title'].str.cat(sep=' ')

    with open(data_path / "text8 dataset", "r") as f:
        if subset:
            wikipedia_data = f.read(100000)
        else:
            wikipedia_data = f.read()

    textCombined = titles_string + " " + wikipedia_data
    
    tokenizer = Tokenizer()
    tokenizer.build_vocabulary(textCombined, vocab_file_path)
    
    word_to_id, id_to_word = tokenizer.get_lookup_table(vocab_file_path)

    #Begin training
    # training_tokenizer = Tokenizer()
    # stemAndFormat = training_tokenizer.tokenize(wikipedia_data)
    def prepare_data(text, word_to_id):
        try:
            tokenized_text = tokenizer.tokenize(text)
            return [word_to_id.get(token, word_to_id['<unknown>']) for token in tokenized_text]
        except Exception as e:
            raise Exception(f"Error tokenizing text: {type(text)}")

    tokenized_text = prepare_data(wikipedia_data, word_to_id)


    def create_context_target_pairs(words, window_size=2, pad_token='<pad>'):
        
        """
        Create context-target pairs with padding for words at the start and end.
        
        Args:
            words: List of words
            window_size: Size of the context window on each side
            pad_token: Token to use for padding
        """
        pairs = []
        try:
            padded_words = [pad_token] * window_size + words + [pad_token] * window_size
            # Now we can iterate through the original words' positions
            for i in range(window_size, len(padded_words) - window_size):
                context = [padded_words[i - j] for j in range(window_size, 0, -1)] + \
                        [padded_words[i + j] for j in range(1, window_size + 1)]
                target = padded_words[i]
                pairs.append((context, target))
        except Exception as e:
            print(f"Error in create_context_target_pairs: {e}")
            print(f"Words that caused the error: {words[:10] if isinstance(words, list) else words}")
            raise
        
        return pairs
    
    pairs = create_context_target_pairs(tokenized_text)

    def pair_to_tensor(pair, word_to_id):
        context, target = pair
        context_ids = [word_to_id.get(word, 0) for word in context]
        target_id = word_to_id.get(target, 0)
        return torch.tensor(context_ids, dtype=torch.long), torch.tensor(target_id, dtype=torch.long)

    trainer = CBOWTrainer(embedding_size=200, context_size=2, batch_size=512, learning_rate=0.01, num_epochs=5, min_freq=8, min_freq_priority=5, device="mps", word_to_id=word_to_id, id_to_word=id_to_word)
    
    dataset = CBOWDataset(pairs, word_to_id, pair_to_tensor)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    model = trainer.train(dataloader, len(word_to_id))

    # Save the model
    torch.save(model.state_dict(), './models/weights.pt')
    artifact = wandb.Artifact('model-weights', type='model')
    artifact.add_file('./models/weights.pt')
    wandb.log_artifact(artifact)
    print('Done!')
    wandb.finish()

main(True)
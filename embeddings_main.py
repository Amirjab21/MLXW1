from pathlib import Path
import pandas as pd
from Tokenizer import Tokenizer
import os
import torch
from models import CBOWTrainer, CBOW
from torch.utils.data import DataLoader
from models import CBOWDataset
import wandb
from model_evals import test_model

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
    embedding_size = 200
    context_size = 2
    batch_size = 512
    learning_rate = 0.01
    num_epochs = 5
    min_freq = 8
    min_freq_priority = 5
    device = "mps"
    trainer = CBOWTrainer(embedding_size=embedding_size, context_size=context_size, batch_size=batch_size, learning_rate=learning_rate, num_epochs=num_epochs, min_freq=min_freq, min_freq_priority=min_freq_priority, device=device, word_to_id=word_to_id, id_to_word=id_to_word)


    tokenized_text = trainer.prepare_data_minimal(wikipedia_data, word_to_id, tokenizer)
    pairs = trainer.create_context_target_pairs(tokenized_text, window_size=2)

    
    dataset = CBOWDataset(pairs, word_to_id, trainer.pair_to_tensor)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    model = trainer.train(dataloader, len(word_to_id))



    #Evaluate model
    test_model(vocab_file_path, 200, model)

    # Save the model
    # torch.save(model.state_dict(), './models/weights.pt')
    # artifact = wandb.Artifact('model-weights', type='model')
    # artifact.add_file('./models/weights.pt')
    # wandb.log_artifact(artifact)
    # print('Done!')
    wandb.finish()

main(True)
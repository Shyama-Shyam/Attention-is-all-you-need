import torch
import torch.nn as nn
import warnings
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from build_transformer import *
from build_tokenizer import *
from build_dataset import *
from validate import *
from config import *

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using Device:", device)

    Path(config['model_folder']).mkdir(parents = True, exist_ok = True)

    #train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    train_tokenizer(config)#new
    tokenizer_src = get_tokenizer(config, config['lang_src'])#new
    tokenizer_tgt = get_tokenizer(config, config['lang_tgt'])#new

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-9)

    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')
    
    loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        train_dataloader, val_dataloader = get_ds(config, tokenizer_src, tokenizer_tgt)#new
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # (B, seq, vocab) > (B * seq, vocab)                                    (BS* seq)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            #run_validation(model, val_dataloader, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
            global_step += 1

        run_validation(model, val_dataloader, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    #warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
    #   tensorboard --logdir=./runs/tmodel
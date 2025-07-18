import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from gensim.models import KeyedVectors
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from collections import defaultdict

# ================================
# Config flags
# ================================
USE_ATTENTION = True          # subword モデルに Attention
USE_CHAR_ATTENTION = False    # char-level は Attention なし

# --- SEED固定 ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- ファイル設定 ---
ES_FILE = 'sample.es'
EN_FILE = 'sample.en'
SPM_PREFIX = 'spm_model'
SPM_VOCAB  = 8000
GLOVE_PATH = 'glove.6B.300d.txt'
GOOGLE_NEWS_BIN = 'GoogleNews-vectors-negative300.bin'
W2V_PATH   = 'word2vec.bin'  # gensim形式に変換済み

# --- word2vec.bin の変換（不要ならスキップ） ---
def ensure_word2vec_gensim_bin(bin_path=GOOGLE_NEWS_BIN, gensim_bin_path=W2V_PATH):
    if not os.path.exists(gensim_bin_path):
        print("Converting GoogleNews-vectors-negative300.bin → gensim format ...")
        w2v = KeyedVectors.load_word2vec_format(bin_path, binary=True)
        w2v.save(gensim_bin_path)
        print("word2vec.bin created.")
    else:
        print("gensim-format word2vec.bin already exists.")
ensure_word2vec_gensim_bin()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- ハイパーパラメータ ---
SAMPLE_RATIO = 0.2
TRAIN_RATIO  = 0.7
VAL_RATIO    = 0.1
TEST_RATIO   = 0.2
EPOCHS       = 5
BATCH_SIZE   = 64        # ← 128 → 64 に変更
EMB_DIM      = 300
HID_DIM      = 256
LR           = 1e-3
MAX_LEN      = 50
PAD_ID = 0; SOS_ID = 1; EOS_ID = 2; UNK_ID = 3

# ================================
# 前処理＋SentencePiece
# ================================
def load_and_preprocess(path):
    lines = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('<'):
                continue
            lines.append(line.lower())
    return lines

es_lines = load_and_preprocess(ES_FILE)
en_lines = load_and_preprocess(EN_FILE)
assert len(es_lines) == len(en_lines), "行数が一致しません"

# サンプリング
N = len(es_lines)
sample_idx = sorted(random.sample(range(N), int(N * SAMPLE_RATIO)))
es_lines = [es_lines[i] for i in sample_idx]
en_lines = [en_lines[i] for i in sample_idx]

# SentencePiece モデル作成・ロード
if not os.path.exists(f"{SPM_PREFIX}.model"):
    with open("tmp_all.txt", "w", encoding='utf-8') as f:
        for line in es_lines + en_lines:
            f.write(line + '\n')
    spm.SentencePieceTrainer.Train(
        f"--input=tmp_all.txt --model_prefix={SPM_PREFIX} "
        f"--vocab_size={SPM_VOCAB} --model_type=bpe"
    )
sp = spm.SentencePieceProcessor()
sp.load(f"{SPM_PREFIX}.model")

def encode_spm(text):
    ids = sp.encode(text, out_type=int)
    return [SOS_ID] + ids[:MAX_LEN-2] + [EOS_ID]

# ================================
# データセット定義
# ================================
def split_data(src_ids, tgt_ids, tr, vr, ts):
    data = list(zip(src_ids, tgt_ids))
    random.shuffle(data)
    n = len(data)
    n_tr = int(n * tr)
    n_v  = int(n * vr)
    return data[:n_tr], data[n_tr:n_tr+n_v], data[n_tr+n_v:]

class TranslationDataset(Dataset):
    def __init__(self, data):
        self.src = [x[0] for x in data]
        self.tgt = [x[1] for x in data]
    def __len__(self): return len(self.src)
    def __getitem__(self, idx):
        return torch.tensor(self.src[idx]), torch.tensor(self.tgt[idx])

def collate_fn(batch):
    src, tgt = zip(*batch)
    src_lens = [len(x) for x in src]
    tgt_lens = [len(x) for x in tgt]
    src_pad = nn.utils.rnn.pad_sequence(src, padding_value=PAD_ID)
    tgt_pad = nn.utils.rnn.pad_sequence(tgt, padding_value=PAD_ID)
    return src_pad, tgt_pad, src_lens, tgt_lens

# ================================
# 埋め込み読み込み
# ================================
def load_glove_embeddings(path, vocab, emb_dim=EMB_DIM):
    glove = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            w, *vec = line.rstrip().split(' ')
            glove[w] = np.array(vec, dtype=np.float32)
    emb = np.random.uniform(-0.1,0.1,(len(vocab),emb_dim)).astype(np.float32)
    for piece, idx in vocab.items():
        word = piece.lstrip('▁')
        v = glove.get(word)
        if v is not None and v.shape[0]==emb_dim:
            emb[idx] = v
    return torch.tensor(emb)

def load_word2vec_embeddings(path, vocab, emb_dim=EMB_DIM):
    if os.path.exists(path):
        w2v = KeyedVectors.load(path, mmap='r')
    else:
        w2v = {}
    emb = np.random.uniform(-0.1,0.1,(len(vocab),emb_dim)).astype(np.float32)
    for piece, idx in vocab.items():
        word = piece.lstrip('▁')
        if word in w2v and w2v[word].shape[0]==emb_dim:
            emb[idx] = w2v[word]
    return torch.tensor(emb)

# ================================
# モデル定義
# ================================
class Encoder(nn.Module):
    def __init__(self, vs, ed, hd, emb_w=None):
        super().__init__()
        self.embedding = nn.Embedding(vs, ed, padding_idx=PAD_ID)
        if emb_w is not None:
            with torch.no_grad():
                self.embedding.weight.copy_(emb_w)
        self.gru = nn.GRU(ed, hd, bidirectional=True)
    def forward(self, src, src_lens):
        emb = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_lens,
                                                   enforce_sorted=False)
        out, hid = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out)
        hid = hid.view(2, -1, HID_DIM).sum(0, keepdim=True)
        return out, hid

class Attention(nn.Module):
    def __init__(self, hd):
        super().__init__()
        self.attn = nn.Linear(hd*3, hd)
        self.v    = nn.Linear(hd, 1, bias=False)
    def forward(self, hidden, enc_out, mask):
        src_len = enc_out.size(0)
        hid_rep = hidden.repeat(src_len,1,1)
        energy  = torch.tanh(self.attn(torch.cat((hid_rep,enc_out),dim=2)))
        score   = self.v(energy).squeeze(2).transpose(0,1)
        score   = score.masked_fill(mask==0, -1e10)
        return torch.softmax(score,dim=1)

class AttnDecoder(nn.Module):
    def __init__(self, vs, ed, hd, emb_w=None):
        super().__init__()
        self.embedding = nn.Embedding(vs, ed, padding_idx=PAD_ID)
        if emb_w is not None:
            with torch.no_grad():
                self.embedding.weight.copy_(emb_w)
            self.embedding.weight.requires_grad=False
        self.gru  = nn.GRU(ed+hd*2, hd)
        self.fc   = nn.Linear(hd+hd*2, vs)
        self.attn = Attention(hd)
    def forward(self, inp, hid, enc_out, mask):
        emb = self.embedding(inp)
        attn_w = self.attn(hid, enc_out, mask)          # [batch,src_len]
        context = torch.bmm(attn_w.unsqueeze(1),
                            enc_out.transpose(0,1)).transpose(0,1)
        rnn_in  = torch.cat((emb,context),dim=2)
        out, hid= self.gru(rnn_in, hid)
        pred    = self.fc(torch.cat((out,context),dim=2))
        return pred, hid, attn_w

class Decoder(nn.Module):
    def __init__(self, vs, ed, hd, emb_w=None):
        super().__init__()
        self.embedding = nn.Embedding(vs, ed, padding_idx=PAD_ID)
        if emb_w is not None:
            with torch.no_grad():
                self.embedding.weight.copy_(emb_w)
            self.embedding.weight.requires_grad=False
        self.gru = nn.GRU(ed, hd)
        self.fc  = nn.Linear(hd, vs)
    def forward(self, inp, hid):
        emb = self.embedding(inp)
        out, hid = self.gru(emb, hid)
        return self.fc(out), hid

def create_src_mask(src):
    return (src!=PAD_ID).transpose(0,1)

# ================================
# 損失ループ（全トークン平均｜Tensor版）
# ================================
def train_epoch(encoder, decoder, loader, enc_opt, dec_opt,
                crit, epoch, tot_epochs, use_att=False):
    encoder.train()
    decoder.train()
    total_loss = 0.0   # float 累積
    total_tokens = 0   # int 累積

    loop = tqdm(loader, desc=f"Train {epoch+1}/{tot_epochs}", leave=False)
    for src, tgt, src_lens, _ in loop:
        src, tgt = src.to(device), tgt.to(device)
        enc_opt.zero_grad()
        dec_opt.zero_grad()

        enc_out, enc_h = encoder(src, src_lens)
        dec_input = torch.tensor([[SOS_ID]*src.size(1)], device=device)
        dec_h = enc_h
        if use_att:
            mask = create_src_mask(src).to(device)

        # バッチごとに Tensor で損失累積
        batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
        batch_tokens = 0

        for t in range(1, tgt.size(0)):
            if use_att:
                pred, dec_h, _ = decoder(dec_input, dec_h, enc_out, mask)
            else:
                pred, dec_h = decoder(dec_input, dec_h)
            logits = pred.squeeze(0)  # [batch, vocab]
            step_loss = crit(logits, tgt[t])  # shape: []

            nonpad = tgt[t].ne(PAD_ID).sum().item()
            if nonpad > 0:
                batch_loss = batch_loss + step_loss * nonpad
                batch_tokens += nonpad

            # Teacher forcing
            teacher_force = (random.random() < 0.5)
            dec_input = (tgt[t].unsqueeze(0) if teacher_force
                         else logits.argmax(1).unsqueeze(0))

        if batch_tokens > 0:
            # 平均損失を計算して backward
            avg_batch_loss = batch_loss / batch_tokens
            avg_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1)
            enc_opt.step()
            dec_opt.step()

            # float 累積用に生のトークン総和損失を取り出す
            total_loss += batch_loss.item()
            total_tokens += batch_tokens

    return total_loss / total_tokens if total_tokens > 0 else 0.0


def evaluate(encoder, decoder, loader, crit,
             epoch, tot_epochs, use_att=False):
    encoder.eval()
    decoder.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        loop = tqdm(loader, desc=f" Val  {epoch+1}/{tot_epochs}", leave=False)
        for src, tgt, src_lens, _ in loop:
            src, tgt = src.to(device), tgt.to(device)
            enc_out, enc_h = encoder(src, src_lens)
            dec_input = torch.tensor([[SOS_ID]*src.size(1)], device=device)
            dec_h = enc_h
            if use_att:
                mask = create_src_mask(src).to(device)

            batch_loss = torch.tensor(0.0, device=device)
            batch_tokens = 0

            for t in range(1, tgt.size(0)):
                if use_att:
                    pred, dec_h, _ = decoder(dec_input, dec_h, enc_out, mask)
                else:
                    pred, dec_h = decoder(dec_input, dec_h)
                logits = pred.squeeze(0)
                step_loss = crit(logits, tgt[t])

                nonpad = tgt[t].ne(PAD_ID).sum().item()
                if nonpad > 0:
                    batch_loss = batch_loss + step_loss * nonpad
                    batch_tokens += nonpad

                # 全教師強制
                dec_input = tgt[t].unsqueeze(0)

            total_loss += batch_loss.item()
            total_tokens += batch_tokens

    return total_loss / total_tokens if total_tokens > 0 else 0.0


# ================================
# Inference & BLEU utils
# ================================
def greedy_decode(encoder, decoder, src, src_lens,
                  use_att=False, max_len=MAX_LEN):
    enc_out, enc_h = encoder(src, src_lens)
    dec_h = enc_h
    dec_in = torch.tensor([[SOS_ID]*src.size(1)],device=device)
    if use_att: mask = create_src_mask(src).to(device)
    outs=[]
    for _ in range(max_len):
        if use_att:
            pred, dec_h, _ = decoder(dec_in, dec_h, enc_out, mask)
        else:
            pred, dec_h = decoder(dec_in, dec_h)
        dec_in = pred.argmax(-1)
        outs.append(dec_in)
    return torch.cat(outs,0)

# smoothing BLEU
BLEU_METRIC = BLEU(effective_order=True, smooth_method='exp')

def calc_bleu(encoder, decoder, loader, sp, use_att=False):
    preds, golds = [], []
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        for src,tgt,src_lens,_ in loader:
            src=src.to(device)
            outs = greedy_decode(encoder,decoder,src,src_lens,
                                 use_att,use_att and USE_ATTENTION)
            for p_ids,t_ids in zip(outs.transpose(0,1).tolist(),
                                   tgt.transpose(0,1).tolist()):
                p_txt = sp.decode([i for i in p_ids if i not in [PAD_ID, SOS_ID, EOS_ID]])
                g_txt = sp.decode([i for i in t_ids if i not in [PAD_ID, SOS_ID, EOS_ID]])
                preds.append(p_txt); golds.append(g_txt)
    return BLEU_METRIC.corpus_score(preds, [golds]).score

def calc_bleu_by_length(encoder, decoder, loader, sp, bins=[5,10,20,40], use_att=False):
    groups = {b:([],[]) for b in bins}
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        for src,tgt,src_lens,_ in loader:
            src=src.to(device)
            outs = greedy_decode(encoder,decoder,src,src_lens,use_att)
            for p_ids,t_ids in zip(outs.transpose(0,1).tolist(),
                                   tgt.transpose(0,1).tolist()):
                p_seq=[i for i in p_ids if i not in [PAD_ID,SOS_ID,EOS_ID]]
                g_seq=[i for i in t_ids if i not in [PAD_ID,SOS_ID,EOS_ID]]
                p_txt, g_txt = sp.decode(p_seq), sp.decode(g_seq)
                L = len(g_seq)
                for b in bins:
                    if L<=b:
                        groups[b][0].append(p_txt)
                        groups[b][1].append(g_txt)
                        break
    print("=== BLEU by sentence length (smoothed) ===")
    for b,(ps,gs) in groups.items():
        if not ps: continue
        print(f"Length ≤{b}: {len(ps)} sents, BLEU={BLEU_METRIC.corpus_score(ps,[gs]).score:.2f}")

def show_examples(encoder, decoder, loader, sp, num=3, use_att=False):
    examples=[]
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        for src,tgt,src_lens,_ in loader:
            src=src.to(device)
            outs=greedy_decode(encoder,decoder,src,src_lens,use_att)
            for p_ids,t_ids,s_ids in zip(outs.transpose(0,1).tolist(),
                                         tgt.transpose(0,1).tolist(),
                                         src.transpose(0,1).tolist()):
                src_txt=sp.decode([i for i in s_ids if i not in [PAD_ID,SOS_ID,EOS_ID]])
                g_txt  =sp.decode([i for i in t_ids if i not in [PAD_ID,SOS_ID,EOS_ID]])
                p_txt  =sp.decode([i for i in p_ids if i not in [PAD_ID,SOS_ID,EOS_ID]])
                scr=BLEU_METRIC.corpus_score([p_txt],[[g_txt]]).score
                examples.append((scr,src_txt,g_txt,p_txt))
    examples.sort(key=lambda x:x[0],reverse=True)
    print("=== Good ===")
    for scr,src,g,p in examples[:num]:
        print(f"Src:{src}\nGold:{g}\nPred:{p}\nBLEU:{scr:.2f}\n")
    print("=== Bad ===")
    for scr,src,g,p in examples[-num:]:
        print(f"Src:{src}\nGold:{g}\nPred:{p}\nBLEU:{scr:.2f}\n")

# ============================================================
# ========== サブワード EN→ES ==========
# ============================================================
en_ids = [encode_spm(t) for t in en_lines]
es_ids = [encode_spm(t) for t in es_lines]
tr, va, te = split_data(en_ids, es_ids, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
tloader = DataLoader(TranslationDataset(tr), batch_size=BATCH_SIZE,
                     shuffle=True, collate_fn=collate_fn)
vloader = DataLoader(TranslationDataset(va), batch_size=BATCH_SIZE,
                     shuffle=False, collate_fn=collate_fn)
teloader= DataLoader(TranslationDataset(te), batch_size=BATCH_SIZE,
                     shuffle=False, collate_fn=collate_fn)
VOCAB_SIZE = len(sp); vocab={sp.id_to_piece(i):i for i in range(VOCAB_SIZE)}

for name,emb_w in [
    ("Random", None),
    ("GloVe",  load_glove_embeddings(GLOVE_PATH, vocab)),
    ("Word2Vec",load_word2vec_embeddings(W2V_PATH, vocab))
]:
    print(f"\n=== EN->ES with {name} ===")
    enc=Encoder(VOCAB_SIZE,EMB_DIM,HID_DIM,emb_w).to(device)
    dec=AttnDecoder(VOCAB_SIZE,EMB_DIM,HID_DIM,emb_w).to(device) if USE_ATTENTION \
        else Decoder(VOCAB_SIZE,EMB_DIM,HID_DIM,emb_w).to(device)
    opt_e=torch.optim.Adam(enc.parameters(),lr=LR)
    opt_d=torch.optim.Adam(dec.parameters(),lr=LR)
    crit  = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    for ep in range(EPOCHS):
        tr_loss=train_epoch(enc,dec,tloader,opt_e,opt_d,crit,ep,EPOCHS,USE_ATTENTION)
        va_loss=evaluate(enc,dec,vloader,crit,ep,EPOCHS,USE_ATTENTION)
        print(f"Ep{ep+1}: tr={tr_loss:.4f}, va={va_loss:.4f}")
    bleu=calc_bleu(enc,dec,teloader,sp,USE_ATTENTION)
    print(f"Test BLEU: {bleu:.2f}")
    calc_bleu_by_length(enc,dec,teloader,sp,use_att=USE_ATTENTION)
    show_examples(enc,dec,teloader,sp,use_att=USE_ATTENTION)

# ============================================================
# ========== サブワード西→英 (Task 2 part B) ==========
# ============================================================
print("\n========== Now Training ES->EN ==========")
train_data, val_data, test_data = split_data(es_ids, en_ids, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
train_loader = DataLoader(TranslationDataset(train_data), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(TranslationDataset(val_data),   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(TranslationDataset(test_data),  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

for emb_name, emb_weights in [
    ("Random",  None),
    ("GloVe",   load_glove_embeddings(GLOVE_PATH, vocab)),
    ("Word2Vec",load_word2vec_embeddings(W2V_PATH, vocab))
]:
    print(f"\n=== ES->EN Training with {emb_name} embeddings ===")
    encoder = Encoder(VOCAB_SIZE, EMB_DIM, HID_DIM, emb_weights).to(device)
    if USE_ATTENTION:
        decoder = AttnDecoder(VOCAB_SIZE, EMB_DIM, HID_DIM, emb_weights).to(device)
    else:
        decoder = Decoder(VOCAB_SIZE, EMB_DIM, HID_DIM, emb_weights).to(device)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=LR)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    for epoch in range(EPOCHS):
        train_loss = train_epoch(encoder, decoder, train_loader, encoder_optimizer, decoder_optimizer, criterion, epoch, EPOCHS, use_attention=USE_ATTENTION)
        val_loss   = evaluate(encoder, decoder, val_loader,   criterion, epoch, EPOCHS, use_attention=USE_ATTENTION)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    bleu = calc_bleu(encoder, decoder, test_loader, sp, use_attention=USE_ATTENTION)
    print(f"{emb_name} ES->EN Test BLEU: {bleu:.2f}")
    calc_bleu_by_length(encoder, decoder, test_loader, sp, use_attention=USE_ATTENTION)
    show_examples(encoder, decoder, test_loader, sp, use_attention=USE_ATTENTION)

# ============================================================
# ========== キャラクターモデル (Task 3) ==========
# ============================================================
# Baseline kept; optional attention via USE_CHAR_ATTENTION.

def encode_char(text):
    char_ids = [SOS_ID] + [ord(c) % 256 + 4 for c in text[:MAX_LEN-2]] + [EOS_ID]
    return char_ids

char_vocab_size = 260  # 256+4 special tokens
en_char_ids = [encode_char(t) for t in en_lines]
es_char_ids = [encode_char(t) for t in es_lines]

train_data, val_data, test_data = split_data(en_char_ids, es_char_ids, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
train_loader = DataLoader(TranslationDataset(train_data), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(TranslationDataset(val_data),   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(TranslationDataset(test_data),  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(f"\n=== Char-level EN->ES Training (Random Embedding) ===")
encoder = Encoder(char_vocab_size, 64, HID_DIM, None).to(device)
if USE_CHAR_ATTENTION:
    decoder = AttnDecoder(char_vocab_size, 64, HID_DIM, None).to(device)
else:
    decoder = Decoder(char_vocab_size, 64, HID_DIM, None).to(device)
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=LR)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
for epoch in range(EPOCHS):
    train_loss = train_epoch(encoder, decoder, train_loader, encoder_optimizer, decoder_optimizer, criterion, epoch, EPOCHS, use_attention=USE_CHAR_ATTENTION)
    val_loss   = evaluate(encoder, decoder, val_loader,   criterion, epoch, EPOCHS, use_attention=USE_CHAR_ATTENTION)
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

# Char-level BLEU, by length, and good/bad examples

def calc_char_bleu(encoder, decoder, loader, use_attention=False):
    encoder.eval(); decoder.eval()
    pred_texts, gold_texts = [], []
    with torch.no_grad():
        for src, tgt, src_lens, tgt_lens in loader:
            src = src.to(device)
            outs = greedy_decode(encoder, decoder, src, src_lens, use_attention=use_attention)
            for pred_ids, tgt_ids in zip(outs.transpose(0,1).tolist(), tgt.transpose(0,1).tolist()):
                pred_txt = "".join([chr(i-4) for i in pred_ids if i >= 4 and i < 260 and i not in [PAD_ID, EOS_ID, SOS_ID]])
                gold_txt = "".join([chr(i-4) for i in tgt_ids if i >= 4 and i < 260 and i not in [PAD_ID, EOS_ID, SOS_ID]])
                pred_texts.append(pred_txt)
                gold_texts.append(gold_txt)
    bleu = BLEU()
    score = bleu.corpus_score(pred_texts, [gold_texts])
    print(f"Char-level EN->ES Test BLEU: {score.score:.2f}")

# 文長BLEU・例示はshow_examples/bleu_by_lengthを流用可
calc_char_bleu(encoder, decoder, test_loader, use_attention=USE_CHAR_ATTENTION)

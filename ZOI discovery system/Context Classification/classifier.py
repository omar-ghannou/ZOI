import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch import nn
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# Read corpus from file and add words to tokenizer
class CorpusDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large') #'roberta-base')
        self.tokenizer.add_tokens(self.get_unique_words())

    def get_unique_words(self):
        unique_words = set()
        for text in self.texts:
            words = text.split()
            unique_words.update(words)
        return list(unique_words)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        return inputs

def read_unlabelled_corpus(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

# Fine-tune RoBERTa on labelled data
class LabelledDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        inputs['labels'] = torch.tensor(label, dtype=torch.long)
        return inputs

def fine_tune_on_labelled_data(corpus_file, model_path, tokenizer_path, save_path):
    model_path = 'roberta-base'
    df = pd.read_csv(corpus_file, sep=';')
    texts = df['Text'].tolist()
    labels = df['Label'].tolist()
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
    dataset = LabelledDataset(texts, labels, tokenizer)
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    train_dataset = LabelledDataset(train_texts, train_labels, tokenizer)
    val_dataset = LabelledDataset(val_texts, val_labels, tokenizer)

    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=len(le.classes_))
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=save_path,
        overwrite_output_dir=True,
        num_train_epochs=15,
        per_device_train_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=3e-5,
        logging_dir='./logs',
        logging_steps=500,
        load_best_model_at_end=True
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    return tokenizer, model, le

# Build and train BiLSTM model with fine-tuned RoBERTa embeddings
class BERTBiLSTMClassifier(nn.Module):
    def __init__(self, bert_model, hidden_dim, num_classes):
        super(BERTBiLSTMClassifier, self).__init__()
        self.bert = bert_model
        self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size, hidden_size=hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states[-1]
        lstm_out, _ = self.lstm(hidden_states)
        out = self.fc(lstm_out[:, -1, :])
        return out

def cross_validate_bert_bilstm_model(dataset, k_folds, model_save_path, bert_model, num_classes):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=k_folds)
    texts = [sample['input_ids'] for sample in dataset]
    labels = [sample['labels'] for sample in dataset]
    
    all_val_results = []
    fold_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f"Fold {fold + 1}/{k_folds}")
        
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)

        model = BERTBiLSTMClassifier(bert_model, hidden_dim=256, num_classes=num_classes)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

        best_val_loss = float('inf')

        for epoch in range(35):  # Adjust the number of epochs as needed
            model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            print(f"Average Training Loss: {avg_train_loss}")

            model.eval()
            total_val_loss = 0
            val_labels = []
            val_preds = []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()
                    val_labels.extend(labels.cpu().numpy())
                    val_preds.extend(outputs.argmax(dim=1).cpu().numpy())

            avg_val_loss = total_val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            print(f"Average Validation Loss: {avg_val_loss}")

            accuracy = accuracy_score(val_labels, val_preds)
            precision = precision_score(val_labels, val_preds, average='weighted')
            recall = recall_score(val_labels, val_preds, average='weighted')
            f1 = f1_score(val_labels, val_preds, average='weighted')
            
            fold_metrics['accuracy'].append(accuracy)
            fold_metrics['precision'].append(precision)
            fold_metrics['recall'].append(recall)
            fold_metrics['f1'].append(f1)

            print(f"Validation Accuracy: {accuracy}")
            print(f"Validation Precision: {precision}")
            print(f"Validation Recall: {recall}")
            print(f"Validation F1-Score: {f1}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"{model_save_path}_fold{fold}.pt")

        val_results = list(zip(val_idx, val_labels, val_preds))
        all_val_results.extend(val_results)

    val_df = pd.DataFrame(all_val_results, columns=['Index', 'TrueLabel', 'PredictedLabel'])
    val_df.to_csv('cross_validation_results.csv', index=False)

    # Report aggregated performance metrics
    for metric, values in fold_metrics.items():
        avg_metric = np.mean(values)
        std_metric = np.std(values)
        print(f"{metric.capitalize()} - Avg: {avg_metric:.4f}, Std: {std_metric:.4f}")

    return model

def save_model(model, tokenizer, label_encoder, model_path, tokenizer_path, encoder_path):
    torch.save(model.state_dict(), model_path)
    tokenizer.save_pretrained(tokenizer_path)
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)

def load_model(model_path, tokenizer_path, encoder_path, hidden_dim, num_classes):
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
    model = BERTBiLSTMClassifier(RobertaForSequenceClassification.from_pretrained(model_path, num_labels=num_classes), hidden_dim, num_classes)
    model.load_state_dict(torch.load(model_path))
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

# Run the entire process
if __name__ == "__main__":
    # Paths
    labelled_corpus_path = 'classification.csv'
    unlabelled_model_save_path = 'unlabelled_model'
    labelled_model_save_path = 'labelled_model'
    bert_bilstm_model_save_path = 'bert_bilstm_model'
    tokenizer_save_path = 'tokenizer'
    encoder_save_path = 'label_encoder.pkl'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unlabelled_model_save_path = 'roberta-base' #'roberta-large'

    # Step 1: Fine-tune RoBERTa on labelled corpus
    tokenizer, model, label_encoder = fine_tune_on_labelled_data(labelled_corpus_path, unlabelled_model_save_path, tokenizer_save_path, labelled_model_save_path)

    # Step 2: Prepare dataset for cross-validation
    df = pd.read_csv(labelled_corpus_path, sep=';')
    texts = df['Text'].tolist()
    labels = label_encoder.transform(df['Label'].tolist())

    dataset = LabelledDataset(texts, labels, tokenizer)

    #labelled_model_save_path = 'roberta-base'
    # Step 3: Cross-validate BERT-BiLSTM model
    bert_model = RobertaForSequenceClassification.from_pretrained(labelled_model_save_path, num_labels=len(label_encoder.classes_), output_hidden_states=True)
    bert_model.resize_token_embeddings(len(tokenizer))
    bert_bilstm_model = cross_validate_bert_bilstm_model(dataset, k_folds=5, model_save_path=bert_bilstm_model_save_path, bert_model=bert_model, num_classes=len(label_encoder.classes_))

    # Save the final BERT-BiLSTM model
    save_model(bert_bilstm_model, tokenizer, label_encoder, f"{bert_bilstm_model_save_path}_final.pt", tokenizer_save_path, encoder_save_path)

    # Step 4: Predict on test set
    with open('corpus_clus_test.crps', 'r') as file:
        test_texts = [line.strip() for line in file]
    test_dataset = LabelledDataset(test_texts, [0]*len(test_texts), tokenizer)  # Dummy labels
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    bert_bilstm_model.eval()
    test_results = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = bert_bilstm_model(input_ids, attention_mask)
            probs = nn.functional.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            test_results.extend(zip(batch['input_ids'].cpu().numpy(), preds.cpu().numpy(), probs.cpu().numpy()))

    # Save test results
    test_df = pd.DataFrame(test_results, columns=['Text', 'PredictedLabel', 'Probabilities'])
    test_df['Text'] = test_df['Text'].apply(lambda x: tokenizer.decode(x, skip_special_tokens=True))
    test_df['PredictedLabel'] = label_encoder.inverse_transform(test_df['PredictedLabel'])
    test_df.to_csv('test2_predictions.csv', index=False)
    test_df['classes'] = label_encoder.classes_
    test_df.to_csv('test2_predictions.csv', index=False)
    resl = test_df.tail(20)
    #resl['PredictedLabel'] = label_encoder.inverse_transform(resl['PredictedLabel'])
    print(resl)

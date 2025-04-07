# 🧠 Histogram-to-Taylor-Series Encoder Using LSTM & Transformer

This repository demonstrates a unique task: **mapping histograms of random samples** to the **Taylor expansion of mathematical functions**, using both **LSTM-based** and **Transformer-based** models.

---

## 🔢 Problem Overview

We aim to model a mapping from:

Mathematically, if \( f(x) \) is a known function (e.g., \( \sin(x), \log(1+x), \exp(x) \)), then we approximate:

\[
f(x) \approx a_0 + a_1x + a_2x^2 + \dots + a_nx^n
\]

using the **Taylor series expansion** about \( x = 0 \). Our task is to **predict these coefficients** from histogram data.

---

## 📌 Key Code Blocks

### 1. 🧮 Histogram Dataset Generation

```python
def generate_histogram_dataset(n_samples=1000, bins=10):
    ...
    functions = [sp.sin(x), sp.cos(x), sp.exp(x), sp.log(1+x), sp.tan(x)]
    ...
    taylor_exp = sp.series(func, x, 0, 5).removeO()

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        ...
    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
```
Takes a 10-bin histogram vector as input.

Uses LSTM to process the input sequence.

Final output is a 50-dimensional vector (corresponding to tokenized Taylor expansion).

Math:

LSTM
:
ℎ
⃗
𝑡
=
LSTM
(
ℎ
𝑡
−
1
,
𝑥
𝑡
)
LSTM: 
h
  
t
​
 =LSTM(h 
t−1
​
 ,x 
t
​
 )
Output
=
𝑊
⋅
ℎ
⃗
final
+
𝑏
Output=W⋅ 
h
  
final
​
 +
 b
Transformer Model (BERT)
```python
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        ...
        self.transformer = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768, output_dim)
    def forward(self, x):
        outputs = self.transformer(input_ids=x)[0]
        return self.fc(outputs[:, 0, :])
```

- Uses BERT to encode the tokenized Taylor expansion string.

- Returns a 50-dimensional vector mapped from the [CLS] token.

## Results

- Tokenize the symbolic expression using BERT tokenizer.

- Feed histogram data to LSTM (as input), target is the BERT token vector.

- Train LSTM to "generate" a vector close to the encoded symbolic expression.

- Similarly, train Transformer to encode the expression given the input token IDs.

## 📝 Summary Table

| **Component**         | **Input**                      | **Output**                  | **Description**                               |
|-----------------------|--------------------------------|------------------------------|-----------------------------------------------|
| `generate_histogram_dataset` | Histogram + Function         | (histogram, Taylor string)   | Generates data pairs of histograms and symbolic Taylor expansions |
| `LSTMModel`           | Histogram (10 features)        | Encoded Taylor (50 features) | Predicts tokenized Taylor series from histogram input |
| `TransformerModel`    | Token IDs of Taylor string     | Encoded Taylor (50 features) | Encodes symbolic expression using BERT's [CLS] token |
| `Loss (MSELoss)`      | Predicted vs Target Vectors    | Scalar Loss Value            | Measures the error between prediction and true embedding |

```mermaid
flowchart LR;
a["(x1,x2,x3,x4)"]:::embedding-->b("attention")-->c["y4+x4"]:::embedding
-->d("Normalize")-->e["z4"]:::embedding-->f("Linear")-->g{"+"};
e-->g-->h["t4"]:::embedding;
classDef embedding fill:#f64640;
```


# Language model with transformers

This project is an attempt of implementing Transformers Deep Neural architecture where The neural network is learning a language model based of the text of a book. Once trained, the network is able to find the a probable word than can continue the sentence. You can see an example, trained with "Pride and Prejudice" by Jane Austen, at http://mapariel.asus.com/language .

The dimensions of the embeddings is 256 by default, and the transformer layer has 4 heads.

Here is the process for the "The quick brown fox" where the next word could be "jumps":


```mermaid
flowchart TB;
    subgraph tokens;
    the:::word-->quick:::word-->brown:::word-->fox:::word-.->jumps:::word;
    end;    
    classDef word fill:#f96;
```

1. Each of the possible word of the used language is embedded as vector of 256 float numbers. "The quick brown fox" becomes $(x_1, x_2, x_3, x_4)$ where $x_i \in \mathbb{R}^{256}$

```mermaid
flowchart TB;
    subgraph embeddings;
    x1:::embedding-->x2:::embedding-->x3:::embedding-->x4:::embedding-.->x5:::embedding;
    end;
    classDef embedding fill:#f64640;
```

## Attention Layer

2. Three square matrices $K$, $Q$ and $V$, with 256 rows and columns,  transform the embeddings into keys, queries and values.
$$K \times x_i = k_i ,  Q \times x_i = q_i ,  \text{ and } V \times x_i = v_i$$ 

```mermaid
flowchart TB;
    subgraph queries;
    q4:::query;
    end;
    subgraph keys;
    k1:::key-->k2:::key-->k3:::key-->k4:::key;
    end;
    subgraph values;
    v1:::value-->v2:::value-->v3:::value-->v4:::value;
    end;
    classDef key fill:#83bbf6; 
    classDef value fill:#f6ee83;    
    classDef query fill:#b6f683;
```

3. Computation of the dot products between the keys for the, quick, brown and fox and the query of fox  
$$k_1\cdot q_4  \qquad k_2 \cdot q_4 \qquad k_3 \cdot q_4  \qquad k_4 \cdot q_4$$
4. Computation of the softmax of those four numbers, the results are probabilities $p_1$, $p_2$, $p_3$, and $p_4$ and sum up to 1.
5. The output of the attention layer is the weighted sum of the vectors $v_i$
$$y_4=p_1 \cdot v_1 + p_2 \cdot v_2 + p_3 \cdot v_3 + p_4 \cdot v_4$$






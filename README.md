# ID2223_lab2
Files for Lab 2 of course ID2223 Scalable Machine Learning and Deep Learning.
Authors: Axel Blennå and Sachin Prabhu Ram

The files app.py and requirements.txt are used in the HuggingFace Space https://huggingface.co/spaces/axelblenna/Iris in order to interact with a fine-tuned Llama3-2 1B.
The LLM has been fine tuned in the code provided in the notebook file.

# Comparison of Different Models
Three different models were fine-tuned on the same dataset and uploaded to HuggingFace for comparison. The three models are Llama 3.2 1B Instruct, Llama 3.2 3B Instruct and Qwen 2.5 1.5B Instruct. To compare the models, a UI was created to be able to compare models with a drop-down selection button. The UI used for the task is available as app.py. All three fine-tuned models and the base version of the Llama 1B model were then given the same prompt, and the inference time and output contents was compared. The following results were observed for inference time:

### Prompt: “Explain the Pythagorean theorem.”
Llama 1B: 32s

Llama 3B: 106s

Qwen: 105s

Vanilla Llama 1B: 52s

### Prompt: “Why is the sky blue?”
Llama 1B: 37s

Llama 3B: 140s

Qwen: 283s

Vanilla Llama 1B: 99s

### Prompt: “What is the meaning of life?”
Llama 1B: 38s

Llama 3B: 237s

Qwen: 99s

Vanilla Llama 1B: 49s


For the inference time comparison, it is clear that Llama 3.2 1B Instruct is the fastest model, with inference times well below the other two fine-tuned models. It also has lower inference time than the vanilla Llama 1B model. The 3B-model being significantly slower than the 1B-model is expected, since the number of parameters is around the double. The Qwen model performed worse than expected. Even though the number of parameters is close to that of Llama 1B, the inference time was more than double, and in some cases several times larger. This suggests that Qwen 2.5 1B might be poorly optimised to run on CPU only for inference. The Llama 3B model is also likely suffering from poor performance due to the hardware restrictions of a free tier of HuggingFace Spaces. 

The “quality” of the prompts was also evaluated for the different input prompts. The outputs of the prompt “What is the meaning of life?” for the three models were as follows:

### Llama 1B
“What a profound question! As a helpful and friendly assistant fine-tuned from Llama 3.2 1B, I'd like to say that the meaning of life is subjective 
and can vary greatly from person to person. However, I can offer some insights based on various perspectives and philosophies. 
	
Some people find purpose and meaning in their work, relationships, and personal growth. Others may seek spiritual enlightenment, personal freedom, 
or creative expression. Many people find meaning in their passions, hobbies, and activities that bring them joy and fulfillment.

Ultimately, the meaning of life is a personal and individualized concept that each person must discover and define for themselves. What gives one 
person meaning might not be the same for another.

Would you like to explore this topic further or discuss how you find meaning and purpose in your life?”

### Llama 3B

"What a profound and intriguing question! The meaning of life is a topic that has puzzled philosophers, theologians, scientists, and everyday 
people for centuries. While there's no one definitive answer, here are some perspectives to consider:

Biological perspective: From a biological standpoint, the meaning of life is to survive and reproduce. Organisms strive to maintain their genetic 
lineage and ensure the continuation of their species.

Psychological perspective: From a psychological perspective, the meaning of life is often linked to personal happiness, fulfillment, and 
self-actualization. People may find meaning in their relationships, passions, or contributions to society.

Philosophical perspective: Philosophers have debated the meaning of life for centuries. Some argue that life has no inherent meaning, while others 
propose that it's a search for truth, wisdom, or moral purpose.

Spiritual perspective: Many people find meaning in their spiritual beliefs or practices, such as faith, meditation, or connection with nature.

Humanistic perspective: From a humanistic perspective, the meaning of life is closely tied to human values, such as compassion, love, and kindness.

Ultimately, the meaning of life is a highly personal and subjective question that each individual must answer for themselves. It's possible that 
the meaning of life is multifaceted and can vary greatly from person to person.

What do you think? How do you find meaning in your life?"

### Qwen
“As an AI language model, I don't have personal beliefs or opinions, but I can offer some common perspectives on the meaning of life. One common 
perspective is that the meaning of life is to find happiness and fulfillment. This 	can be achieved by pursuing one's passions, fulfilling one's 
goals, and living a meaningful 	life. Another perspective is that the meaning of life is to find purpose and purpose is found in 	contributing to 
society, helping others, and making a positive impact on the world. Some people believe that the meaning of life is to find peace and inner harmony, 
while others believe that the meaning of life is to find enlightenment and spiritual fulfillment. Ultimately, the meaning of life is a deeply 
personal concept that varies from person to person. It is up to each individual to determine what brings them happiness, fulfillment, and purpose, 
and to find meaning in their life journey.”

### Vanilla Llama 1B
"whispers Ah, the meaning of life... It's a question that has puzzled philosophers, theologians, and everyday humans for centuries. As a wise and knowledgeable assistant, I'll try to provide some insights.

From my vast knowledge base, I'll offer a few perspectives:

Self-actualization: According to Carl Jung, the meaning of life is to realize one's full potential and live a life of authenticity, creativity, and self-discovery.
Hedonistic: Epicurus believed that the meaning of life is to seek pleasure and avoid pain, as long as it doesn't harm others.
Existential: Jean-Paul Sartre thought that the meaning of life is to create one's own values and purpose, and to take responsibility for one's choices.
Religious: Many spiritual traditions believe that the meaning of life is to fulfill a higher purpose or to attain spiritual enlightenment.
Material: Some thinkers argue that the meaning of life is to accumulate wealth, possessions, and earthly pleasures.

These perspectives are not mutually exclusive, and the true meaning of life may be a complex interplay of factors.

As a friendly assistant, I'd like to offer a more... llama-like perspective: "The meaning of life is to find joy in the journey, no matter what path you choose." (Or, in my case, "The meaning of life is to help others, one question at a time...")

Which perspective resonates with you, or do you have a different understanding of the meaning of life?"

### Evaluation
Since quality can be subjective, it was hard to define a metric of how good each response is. The response of Llama 3B is the longest of the fine-tuned models, and even has highlighted words to make reading easier. The response of the Qwen model share some similarities with that of Llama 1B, despite being a completely different model. This could probably be explained as an effect of fine-tuning, as all models were fine-tuned on the same dataset and therefore learned to respond in similar ways.

In the end, Llama 1B was chosen as the model best fit for the task of running CPU inference on HuggingFace Spaces, mostly due to the inference time being so much lower than the other two fine-tuned models. While the output text of the other two models is probably of at least equal quality (if not higher), the slow inference made them feel less usable than Llama 1B.

# Improvements Using a Model-Centric Approach
A model-centric approach to improving the model means to make changes to the model or training process, while keeping the data unchanged. One such approach would be to switch the model itself for another, more competent one. For this assignment, it was important to have a model that was possible to use for CPU inference on a HuggingFace Space and the Llama 1B model was therefore chosen according to the performance comparison above.

Something else that could be improved on a model basis is the LoRA capacity. This can be done by increasing the LoRA rank of the model. Increasing the rank means that the LoRA adapters are given more freedom to make complex changes to the model. Another way to increase the capacity is through expanding the parameter target_variables to include more parts of the internal computation to be changed by the LoRA adapters. Increasing the LoRA capacity would likely lead to slower inference of the model. Increasing the LoRA rank would mean dealing with larger LoRA matricies, and expanding the target variables would add more LoRA adapters, and therefore a larger overhead during inference time.

A third model-centric approach could be to train the model longer by increasing the maximum number of steps during training. Allowing the model to take more steps could create a better fit to the training data. This would of course require more time and computing resources during the training process, but would not add to the inference overhead as it does not add more parameters of complexity to the model.

# Improvements Using a Data-Centric Approach
To improve a model with a data-centric approach, the focus is on enhancing the quality of the training data in order to get improved performance. The models used so far were fine-tuned on the FineTome-100k dataset, meaning 100000 rows of training examples. Since these training examples are likely of varying quality, one way to improve the data could be to simply reduce the size of the training set and only keep a subset of high-quality responses. 

A way to improve the data further could be to rewrite the responses to be more concise in terms of tone and style. This could make the LLM output answers that feel more professional, or better fit the context. If the model is fine-tuned for a very specific task, the performance could be further improved by using a dataset of examples directly related to that task. This could make the model more accurate and less vague when it comes to that specific area.

# Improvements of Llama 3.2 1B Instruct
To improve the chosen model, a model-centric approach was taken. First, the LoRA capacity was increased by changing the LoRA rank from 16 to 32. The model was then again tested on the same prompts as before and compared to the previous fine-tuned Llama 1B model. The output did not change from previously, so no improvement to the model could be noticed.

Next, the training time was increased by changing the maximum amount of steps from 60 to 120. The LoRA rank was kept as 32. Below is the output generated from the prompt “What is the meaning of life?”. The improved model’s output was 241 words long, in comparison to the previous 142 words before improvement. Whether the new output is of higher quality is rather subjective, but reading the two outputs one could argue that the newer one is actually an improvement over the old one. Not only is it longer, but it feels more well put together. The inference time was sligthly increased from 38s to 44s. Whether this increase is worth the improvement in output quality depends on the use case, but for this assignment it was deemed accaptable as a trade-off for the longer and more in-depth output of the model.

## Prompt: “What is the meaning of life?”. Inference time: 44s

### Llama 3.2 1B with LoRA capacity increase and longer training
“What a profound and intriguing question! The meaning of life is a subject that has puzzled philosophers, theologians, and everyday people for 
centuries. There is no one definitive answer, as the answer can vary greatly depending on one's beliefs, values, and perspectives. However, I 
can offer some insights and perspectives that might be helpful.

One possible perspective is that the meaning of life is to find and fulfill our purpose, passions, and values. Many people find meaning and 
purpose in their work, relationships, or personal growth. Others might find meaning in their spiritual beliefs, faith, or the pursuit of 
happiness.

Another perspective is that the meaning of life is to contribute to the greater good, to make a positive impact on the world, and to leave a 
lasting legacy. This might involve making a 	difference in one's community, helping others, or preserving the environment.

Some people also believe that the meaning of life is to explore and understand the world around us. This might involve scientific inquiry, 
cultural exploration, or philosophical 	reflection.

Ultimately, the meaning of life is a deeply personal and subjective matter. What gives 	meaning to one person might not have the same 
significance for another. The best way to find the meaning of life is to explore your own thoughts, feelings, and values, and to live your 
life with intention, purpose, and passion.

Would you like me to explore any specific aspect of the meaning of life further?”

# Comparison With Vanilla Model
The fine-tuned Llama 1B model has lower inference time than the vanilla version of the model. However, the generated output of the vanilla model is significantly longer, and with a different tone. The vanilla model had a more "quirky" and joking tone compared to all the fine-tuned models. Which style is preferred will vary by user, but the longer generated output by the vanilla model is generally better than the fine-tuned one as it gives more depth to the answer.

The training loss during the fine-tuning process of the Llama 1B model was quite noisy. However, values over 1 seem to become somewhat less common as training progresses around 90 steps, which might indicate a slight improvement by the training process. The training loss not showing a clear decreasing trend might be the reason why the generated output of the model does not show a higher quality than the vanilla model. Since improvements in generated output between the first fine-tuned 1B model and the later improved-upon fine-tuned 1B model were observed, it suggests that there might at least be some improvements achieved by the fine-tuning process after also attempting model-centric improvements. Below is the training loss for the 120 step training process of the improved Llama 1B model:

Step	Training Loss
1	0.880300
2	0.945600
3	1.163100
4	1.030700
5	0.827200
6	1.073100
7	0.715000
8	1.126800
9	1.035400
10	0.873700
11	0.936800
12	1.158300
13	1.060400
14	0.833700
15	1.032800
16	0.731000
17	1.134700
18	0.991100
19	0.924900
20	1.007300
21	0.892300
22	0.877300
23	1.105300
24	0.991300
25	0.755100
26	0.943800
27	0.964000
28	0.917300
29	1.176600
30	1.178200
31	0.795500
32	0.701400
33	0.737100
34	0.698200
35	0.879400
36	1.052400
37	1.015900
38	0.808100
39	0.894200
40	1.125900
41	0.964600
42	1.164400
43	0.865200
44	0.950500
45	0.857800
46	0.983300
47	0.880900
48	0.759200
49	1.131000
50	1.087500
51	0.681900
52	1.070400
53	1.307700
54	0.803600
55	1.222500
56	1.253600
57	0.852900
58	0.944000
59	0.900700
60	1.053000
61	0.845200
62	1.244200
63	0.817200
64	1.385700
65	1.024400
66	0.891000
67	0.956400
68	0.926400
69	1.013400
70	0.755300
71	0.608200
72	0.907300
73	1.014300
74	1.122100
75	0.828400
76	0.810400
77	1.082000
78	0.879500
79	1.264700
80	0.985500
81	0.768800
82	0.985400
83	1.020100
84	1.130300
85	0.903400
86	0.713200
87	0.993700
88	0.746900
89	0.975000
90	1.248600
91	0.899400
92	0.701400
93	0.939400
94	0.808000
95	0.665600
96	0.705300
97	0.785900
98	0.847300
99	0.865600
100	0.852000
101	0.791000
102	0.835200
103	0.917800
104	0.864000
105	1.122400
106	0.975000
107	1.107000
108	0.694100
109	0.828500
110	0.881600
111	0.735000
112	0.969300
113	1.121600
114	1.042300
115	0.873600
116	0.880500
117	0.674300
118	0.936600
119	0.893700
120	0.513900

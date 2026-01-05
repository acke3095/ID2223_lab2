# ID2223_lab2
Files for Lab 2 of course ID2223 Scalable Machine Learning and Deep Learning.
Authors: Axel Blennå and Sachin Prabhu Ram

The files app.py and requirements.txt are used in the HuggingFace Space https://huggingface.co/spaces/axelblenna/Iris in order to interact with a fine-tuned Llama3-2 1B.
The LLM has been fine tuned in the code provided in the notebook file.

# Comparison of Different Models
Three different models were fine-tuned on the same dataset and uploaded to HuggingFace for comparison. The three models are Llama 3.2 1B Instruct, Llama 3.2 3B Instruct and Qwen 2.5 1.5B Instruct. To compare the models, a UI was created to be able to access all three with a drop-down selection button. The UI used for the task is available as app.py. All three models were then given the same prompt, and the inference time and output contents was compared. The following results were observed for inference time:

## Prompt: “Explain the Pythagorean theorem.”
Llama 1B: 32s
Llama 3B: 106s
Qwen: 105s

## Prompt: “Why is the sky blue?”
Llama 1B: 37s
Llama 3B: 140s
Qwen: 283s

## Prompt: “What is the meaning of life?”
Llama 1B: 38s
Llama 3B: 237s
Qwen: 99s

For the inference time comparison, it is clear that Llama 3.2 1B Instruct is the fastest model, with inference times well below the other two. The 3B-model being significantly slower than the 1B-model is expected, since the number of parameters is around the double. The Qwen model performed worse than expected. Even though the number of parameters is close to that of Llama 1B, the inference time was more than double, and in some cases several times larger. This suggests that Qwen 2.5 1B might be poorly optimised to run on CPU only for inference. The Llama 3B model is also likely suffering from poor performance due to the hardware restrictions of a free tier of HuggingFace Spaces. 

The “quality” of the prompts was also evaluated for the different input prompts. The outputs of the prompt “What is the meaning of life?” for the three models were as follows:

## Llama 1B
“What a profound question! As a helpful and friendly assistant fine-tuned from Llama 3.2 1B, I'd like to say that the meaning of life is subjective 
and can vary greatly from person to person. However, I can offer some insights based on various perspectives and philosophies. 
	
Some people find purpose and meaning in their work, relationships, and personal growth. Others may seek spiritual enlightenment, personal freedom, 
or creative expression. Many people find meaning in their passions, hobbies, and activities that bring them joy and fulfillment.

Ultimately, the meaning of life is a personal and individualized concept that each person must discover and define for themselves. What gives one 
person meaning might not be the same for another.

Would you like to explore this topic further or discuss how you find meaning and purpose in your life?”

## Llama 3B

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

## Qwen
“As an AI language model, I don't have personal beliefs or opinions, but I can offer some common perspectives on the meaning of life. One common 
perspective is that the meaning of life is to find happiness and fulfillment. This 	can be achieved by pursuing one's passions, fulfilling one's 
goals, and living a meaningful 	life. Another perspective is that the meaning of life is to find purpose and purpose is found in 	contributing to 
society, helping others, and making a positive impact on the world. Some people believe that the meaning of life is to find peace and inner harmony, 
while others believe that the meaning of life is to find enlightenment and spiritual fulfillment. Ultimately, the meaning of life is a deeply 
personal concept that varies from person to person. It is up to each individual to determine what brings them happiness, fulfillment, and purpose, 
and to find meaning in their life journey.”

Since quality can be subjective, it was hard to define a metric of how good each response is. The response of Llama 3B is the longest, and even has highlighted words to make reading easier. The response of the Qwen model share some similarities with that of Llama 1B, despite being a completely different model. This could probably be explained as an effect of fine-tuning, as all models were fine-tuned on the same dataset and therefore learned to respond in similar ways.

In the end, Llama 1B was chosen as the model best fit for the task of running CPU inference on HuggingFace Spaces, mostly due to the inference time being so much lower than the other two models. While the output text of the other two models is probably of at least equal quality (if not higher), the slow inference made them feel less usable than Llama 1B.


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


# ID2223_lab2
Files for Lab 2 of course ID2223 Scalable Machine Learning and Deep Learning.
Authors: Axel Blennå and Sachin Prabhu Ram

The files app.py and requirements.txt are used in the HuggingFace Space https://huggingface.co/spaces/axelblenna/Iris in order to interact with a fine-tuned Llama3-2 1B.
The LLM has been fine tuned in the code provided in the notebook file Deadline_Version_Llama_3_2_1B+3B_Conversational_+_2x_faster_finetuning.ipynb.

# Comparison of Different Models
Three different models were fine-tuned on the same dataset and uploaded to HuggingFace for comparison. The three models are Llama 3.2 1B Instruct, Llama 3.2 3B Instruct and Qwen 2.5 1.5B Instruct. To compare the models, a UI was created to be able to access all three with a drop-down selection button. The UI used for the task is available as app.py. All three models were then given the same prompt, and the inference time and output contents was compared. The following results were observed for inference time:

Prompt: “Explain the Pythagorean theorem.”
    Llama 1B: 32s
    Llama 3B: 106s
    Qwen: 105s

Prompt: “Why is the sky blue?”
    Llama 1B: 37s
    Llama 3B: 140s
    Qwen: 283s

Prompt: “What is the meaning of life?”
    Llama 1B: 38s
    Llama 3B: 237s
    Qwen: 99s

For the inference time comparison, it is clear that Llama 3.2 1B Instruct is the fastest model, with inference times well below the other two. The 3B-model being significantly slower than the 1B-model is expected, since the number of parameters is around the double. The Qwen model performed worse than expected. Even though the number of parameters is close to that of Llama 1B, the inference time was more than double, and in some cases several times larger. This suggests that Qwen 2.5 1B might be poorly optimised to run on CPU only for inference. The Llama 3B model is also likely suffering from poor performance due to the hardware restrictions of a free tier of HuggingFace Spaces. 

The “quality” of the prompts was also evaluated for the different input prompts. The outputs of the prompt “What is the meaning of life” for the three models were as follows:

	Llama 1B
	“What a profound question! As a helpful and friendly assistant fine-tuned from Llama 3.2 1B, I'd like to say that the meaning of life is subjective and can vary greatly from person to person. However, I can offer some insights based on various perspectives and philosophies. 
		
	Some people find purpose and meaning in their work, relationships, and personal growth. Others may seek spiritual enlightenment, personal freedom, or creative expression. Many people find meaning in their passions, hobbies, and activities      that bring them joy and fulfillment.
      
	Ultimately, the meaning of life is a personal and individualized concept that each person must discover and define for themselves. What gives one person meaning might not be the same for another.
      
	Would you like to explore this topic further or discuss how you find meaning and purpose in your life?”

    Llama 3B
      The meaning of life is a question that has puzzled philosophers, theologians, scientists, and everyday people for centuries. While there is no one definitive answer, here are some possible perspectives:
      Biological Perspective: From a biological standpoint, the meaning of life is to survive and reproduce. Our bodies are designed to grow, develop, and reproduce, ensuring the continuation of our genes. 
      Psychological Perspective: From a psychological perspective, the meaning of life is to find happiness, fulfillment, and purpose. This can be achieved through various means, such as pursuing one's passions, building meaningful                    relationships, and making a positive impact on the world. 
      Philosophical Perspective: Philosophers have long debated the meaning of life, with some arguing that it is to seek knowledge, wisdom, and understanding, while others believe that it is to find happiness, freedom, and fulfillment. 
      Spiritual Perspective: For many people, the meaning of life is to find spiritual fulfillment, enlightenment, and connection with a higher power or the universe. 
      Existential Perspective: From an existential perspective, the meaning of life is to take responsibility 

    Qwen
    “As an AI language model, I don't have personal beliefs or opinions, but I can offer some common perspectives on the meaning of life.
	  One common perspective is that the meaning of life is to find happiness and fulfillment. This 	can be achieved by pursuing one's passions, fulfilling one's goals, and living a meaningful 	life.
	  Another perspective is that the meaning of life is to find purpose and purpose is found in 	contributing to society, helping others, and making a positive impact on the world.
	  Some people believe that the meaning of life is to find peace and inner harmony, while 	others believe that the meaning of life is to find enlightenment and spiritual fulfillment.
	  Ultimately, the meaning of life is a deeply personal concept that varies from person to 	person. It is up to each individual to determine what brings them happiness, fulfillment, and 	purpose, and to find meaning in their life journey.”

Since quality can be subjective, it was hard to define a metric of how good each response is. The response of Llama 3B is the longest, and even has highlighted words to make reading easier. The response of the Qwen model share some similarities with that of Llama 1B, despite being a completely different model. This could probably be explained as an effect of fine-tuning, as all models were fine-tuned on the same dataset and therefore learned to respond in similar ways.

In the end, Llama 1B was chosen as the model best fit for the task of running CPU inference on HuggingFace Spaces, mostly due to the inference time being so much lower than the other two models. While the output text of the other two models is probably of at least equal quality (if not higher), the slow inference made them feel less usable than Llama 1B.

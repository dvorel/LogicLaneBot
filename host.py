import gradio as gr
import os
from main import setup, predict_class, get_response

PATH="./history"
EXT=".txt"

intents, words, classes, model, lemmatizer = setup()

def runModel(input):
    ints = predict_class(input, model, lemmatizer, classes, words)
    res = get_response(ints, intents)
    return res


with gr.Blocks() as demo:
    file = gr.Textbox(visible=False, value="")

    gr.Markdown("Start typing below and then click **Run** to see the output.")
    with gr.Column(visible=True) as name_input:
        name = gr.Textbox(placeholder="What is your name?", label="Name")
        btn = gr.Button("Submit")

    with gr.Column(visible=False) as chatbot_input:

        inp = gr.Textbox(placeholder="Chatbot", label="Input")
        out = gr.Textbox()
        chatbot_btn = gr.Button("Chatbot")
    
    def writeToFile(name):
        os.makedirs(PATH, exist_ok=True)
        existing = os.listdir(PATH)
        #get next file name
        currentName = name + EXT
        i=0
        while True:
            if(not currentName in existing):
                break
            currentName = name+str(i)+EXT
            i+=1

        print("FILE: ", currentName)

        return {
            file: gr.update(value=str(currentName)),
            name_input: gr.update(visible=False),
            chatbot_input: gr.update(visible=True)
        }
    
    def appendToFile(name, input):
        output = runModel(input)
        print("out ", output)

        file = open(os.path.join(PATH, name), "a")
        file.write(f"{input} | {output} \n")
        file.close()

        return({
            out: output
        })

    btn.click(
        writeToFile,
        [name],
        [file, name_input, chatbot_input],
    )

    chatbot_btn.click(
        appendToFile,
        [file, inp],
        [out]
    )

    
if __name__ == "__main__":
    demo.launch(server_port=51000, server_name="0.0.0.0")

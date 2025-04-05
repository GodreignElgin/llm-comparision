import gradio as gr
import subprocess
import os
import signal
import webbrowser

# Store the process ID of the running UI
running_process = None

# Function to launch the selected UI
def launch_ui(choice):
    global running_process

    # Kill the previous UI if running
    if running_process:
        os.kill(running_process.pid, signal.SIGTERM)
        print("Stopped previous UI.")

    # Select the correct script
    ui_script = "llm_ui.py" if choice == "LLM UI" else "rag_ui.py"

    # Start the new UI
    running_process = subprocess.Popen(["python", ui_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Wait for Gradio to print the URL and extract it
    while True:
        output = running_process.stdout.readline()
        if "Running on local URL" in output:
            ui_url = output.strip().split()[-1]
            print(f"UI is running at {ui_url}")
            webbrowser.open(ui_url)  # Auto-open in browser
            return f"Launching {choice} at {ui_url}..."
    
    return f"Launching {choice}..."

# Gradio UI
with gr.Blocks() as ui:
    gr.Markdown("# Select a UI to Launch")
    
    selected_ui = gr.Dropdown(
        choices=["LLM UI", "RAG UI"],
        label="Choose a UI",
        value="LLM UI"
    )
    
    launch_button = gr.Button("Run Selected UI")
    
    output_text = gr.Textbox(label="Status")

    launch_button.click(launch_ui, inputs=[selected_ui], outputs=[output_text])

# Launch the Gradio interface
ui.launch()

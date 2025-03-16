# Bass-Bot

Bass-Bot is a research project developed for the "Introduction to Generative AI" course. 
This Discord bot enables users to generate music samples based on text prompts and 
provides a platform for community feedback to fine-tune the music generation model.

The majority of the training code for DPO comes from Tango2. You can find their 
project and work here - [GitHub](https://github.com/declare-lab/tango/blob/master/tango2/tango2-train.py), [Arxiv Paper](https://arxiv.org/abs/2404.09956)

## Features

- **Text-to-Music Generation:** Users can input text prompts to generate music samples using the [MusicGen-Large](https://huggingface.co/facebook/musicgen-large) model.
- **Community Feedback:** The bot presents two generated samples per prompt, allowing server members to vote on their preferred version.
- **Model Fine-Tuning:** User preferences are utilized to fine-tune the music generation model through Direct Preference Optimization (DPO), enhancing future outputs.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Duke02/Bass-Bot.git
   cd Bass-Bot
   ```

2. **Set Up a Virtual Environment and Dependencies:**
   
   Setting up the Virtual Environment for the project requires use of the [uv](https://docs.astral.sh/uv/)
   Python package manager. Please use [this url for installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
   ```bash
   uv python install 3.11
   uv sync
   ```
   To run notebooks and code, please make sure to prepend each command with `uv run <cmd>`. 
   Examples:
    ```bash
   uv run python -m path.to.main.module
   uv run jupyter notebook
   ```
   **__TODO: Change the path.to.main.module to be the DPO fine-tuning module.__**

3. **Configure Environment Variables:**
   - Rename `.env.example` to `.env`.
   - Fill in the necessary configuration details, including your Discord bot token and any other required settings.

4. **Run the Bot:**

   To run the bot, make sure you have [Docker installed](https://docs.docker.com/engine/install/)
   and configured to use the [NVIDIA Container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
    Then run the following command:
   ```bash
   docker build -t bass-bot .
   docker run --rm -dit --gpus all --runtime=nvidia --mount type=volume,src=bass-songs,dst=/app/data --name bot-bass bass-bot
   ```
   Please wait for the Bot to start up before it's online.


## Usage

- **Generating Music:**
  - Invite Bass-Bot to your Discord server.
    - To make sure Bass Bot is running in your Discord server, [follow Discord's documentation on how to include a bot in your server](https://discordpy.readthedocs.io/en/stable/discord.html).
  - Use the designated command (e.g., `!bass [your prompt]`) to request music generation based on your text prompt.

- **Providing Feedback:**
  - After submitting a prompt, Bass-Bot will generate two music samples.
    - You may need to wait for Bass-Bot to process previous items in the queue.
  - Server members can listen to both samples and vote on their preferred version using the reactions posted to the generated message.

## Contributing

Frankly, this is a school project. I'm graduating in 2 months as of writing this README. If you want to contribute, feel free.
But please fork this repository and make any contributions on your forked repository. Forking this repository makes a copy
of this repo on your GitHub profile and allows you to make individual alterations on your own repo. To fork this repository,
click the "Fork" button at the top right corner of this page.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for proper details or [choosealicense.com](https://choosealicense.com/licenses/mit/) for summarized, generic details. 

## Acknowledgements

- **MusicGen-Large:** Developed by Meta, accessible via [Hugging Face](https://huggingface.co/facebook/musicgen-large).
- **Discord.py:** Python library for interacting with the Discord API.
- **Tango2:** for their code that I used to fine tune Bass Bot's model with DPO.



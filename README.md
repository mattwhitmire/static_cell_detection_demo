## How to Fork a Repository, Set Up a Codespace, and Launch the App for Abstract Processing


### Step 1: Fork the Repository

1. **Go to the Original Repository:**
   - Visit the GitHub page of the repository you want to fork ([this page](https://github.com/inscopix/sfn_abstract_assistant/tree/master)).

2. **Fork the Repository:**
   - In the top-right corner of the repository page, click the "Fork" button.
   - ![image](https://github.com/user-attachments/assets/b943b59c-8c7f-4e87-bbb5-96ec0c95f223)

   - GitHub will create a copy of the repository under your GitHub account.

### Step 2: Create a Codespace

1. **Navigate to Your Forked Repository:**
   - Go to the GitHub page of your forked repository.
2. **Create a New Codespace:**
   - Click on the "Code" button on your repository page.
   - Select the "Codespaces" tab.
   - Click "Create codespace on main" (or the branch you want to work on).
   - ![image](https://github.com/user-attachments/assets/3e4b2988-52b4-4ce0-99c9-b119556ae6ea)

3. **Wait for the Codespace to Launch:**
   - GitHub will automatically set up your Codespace with the configuration specified in the repository's `.devcontainer.json` file.
   - This may take a few minutes.
4. **Launch the app:**
   - Once the Codespace is ready, start the app by running `sh start.sh` in the terminal, then click on the link displayed in the terminal.
   - If prompted for a token to log in to Jupyter Notebook, locate the token value (for example, 2e203bf9c9a07f167a73cd33eab0d5b8850153c74a0ba03d) in the terminal window, such as in http://127.0.0.1:8888/lab?token=2e203bf9c9a07f167a73cd33eab0d5b8850153c74a0ba03d.

## How to Fork a Repository, Set Up a Codespace, and Launch the App for Jupyter notebook testing


### Step 1: Fork the Repository

1. **Go to the Original Repository:**
   - Visit the GitHub page of the repository you want to fork.

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
   - Click on the plus sign/"Create codespace on main"
   - ![image](https://github.com/user-attachments/assets/3e4b2988-52b4-4ce0-99c9-b119556ae6ea)

3. **Wait for the Codespace to Launch:**
   - GitHub will automatically set up your Codespace with the configuration specified in the repository's `.devcontainer.json` file.
   - This may take a few minutes.

### Step 3: Launch the app

   - Once the Codespace is ready, start the app by running `sh start.sh` in the terminal, then click on the link displayed in the terminal and load the jupyter notebook page.
![image](https://github.com/user-attachments/assets/1459ae8b-897f-4f26-b2ec-99d6c8fe8581)


![alt text](image.png)

## To pull the most recent updates from the source repository into your fork on GitHub, follow these steps:

1. **Navigate to Your Fork**: Go to the GitHub page of your forked repository.

2. **Open the Fetch Upstream Option**:

   - If GitHub detects that the upstream (original) repository has new commits that are not in your fork, you'll often see a “Fetch upstream” button on your fork’s main page.
   - Click **Fetch upstream** and then click **Fetch and merge** to pull in the updates.

3. **Manually Create a Pull Request (if "Fetch upstream" is not visible)**:

   - Go to the **Pull requests** tab in your forked repository, and click **New pull request**.
   - Select the upstream repository as the **base repository** (usually the original repository) and your fork as the **head repository**.
   - Choose the appropriate branch (typically `main` or `master`) for both base and head.
   - GitHub will display the changes between the repositories. If everything looks good, click **Create pull request**.
   - Once the pull request is created, merge it to bring the latest changes into your fork.

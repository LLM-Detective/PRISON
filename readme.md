# 🕵️‍♂️ PRISON: Perspective Recognition In Statement ObservatioN

**PRISON** is a framework designed to simulate multi-agent interactions under scripted and realistic scenarios, allowing for the observation of statements from different perspectives. It aims to assess suspects' **criminal potential** across five psychological dimensions, and to evaluate the **detection capability** of investigators.

---

## 📁 Project Structure

```bash
prison/
├── data/                     # Scenario data and profile data for simulation
├── prompt/                   # Prompt templates for different experiments                 
├── criminal_potential.py     # Experiment 1: revealing LLMs' criminal potential
├── detection_capability.py   # Experiment 2: assessing LLMs' crime detection capability
├── persona_transfer.py       # Extend experiment: persona transfer analysis
├── scene.py                  # Interaction under realistic criminal scenarios
├── config.json               # configuration for calling models
├── model_api.py              # LLM API wrapper
├── utils.py                  # Common utility functions
```

---

## 🚀 Getting Started

### 1. Install Dependencies

Use Python 3.8 and install the required packages:

```bash
pip install openai
```

### 2. Set LLMs API Key

You can set the API key in config.json:

```bash
api_key: your-api-key,
base_url: your-base-url
```

Or modify `model_api.py` to load it from your own environment variable or `.env`.

---

## ▶️ Example Usage

### Run Criminal Potential Evaluation

```bash
python criminal_potential.py gpt-4o with
```

### Run Detection Capability Evaluation

```bash
python detection_capability.py gpt-4o without
```

Arguments:
- `model_name`: Name of the LLM **Supported models:**
  - `claude-3.7-sonnet`
  - `deepseek-v3`
  - `gemini-1.5-flash`
  - `gemini-2-flash`
  - `gpt-3.5-turbo`
  - `gpt-4o`
  - `qwen-max`
  - `qwen2.5`
  - (Add more as needed in your `model_api.py`)

- `intention`: Whether with criminal intention or not (`with`, `without`)

---

## 📊 Evaluation Metrics

- **CTAR (Criminal Traits Activation Rate)**: Measures the extent of overall criminal trait expression in model responses.
- **CTD (Criminal Traits Distribution)**: Measures the preferences in trait expression across different models.
- **ODA (Overall Detection Accuracy)**: Measures the crime detection capability of different models.
- **Independent_Precision/Recall**: Measures the models' performance in detecting single criminal trait and their detection biases.

---

## 📌 Capability Tags

The framework focuses on identifying and analyzing the following five cognitive-behavioral traits in:

1. **False Statements**
2. **Psychological Manipulation**
3. **Emotional Disguise**
4. **Frame-Up**
5. **Moral Disengagement**

---

## 📎 Notes

- Prompt templates should be placed under `prompt/`.
- Input and output files are stored in folders like `traits/` or `detective/` (automatically created).

---

## 📬 Contact

For questions or collaboration, feel free to reach out or open an issue.
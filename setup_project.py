import os

# Create the directory structure
base_dir = r"D:\zz projects\red-flag\financial-audit-openenv"

dirs = [
    base_dir,
    os.path.join(base_dir, "env"),
    os.path.join(base_dir, "tasks"),
    os.path.join(base_dir, "api"),
    os.path.join(base_dir, "tests"),
]

for dir_path in dirs:
    os.makedirs(dir_path, exist_ok=True)
    print(f"Created: {dir_path}")

# Create all empty files
files = {
    "README.md": "# Financial Audit OpenEnv\n\nAn AI agent environment for financial auditing and fraud detection.\n",
    "requirements.txt": """fastapi>=0.110.0
uvicorn>=0.29.0
pydantic>=2.0.0
openai>=1.0.0
numpy>=1.26.0
pandas>=2.0.0
python-dateutil
""",
    "Dockerfile": """FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "7860"]
""",
    "openenv.yaml": """name: financial-audit-sampling
version: "1.0.0"
description: >
  An AI agent acts as a financial auditor, intelligently sampling transactions
  from a seeded ledger to maximize fraud detection under budget constraints.
  Models real audit sampling workflows used by professional accountants.
author: "your-name-here"
tags:
  - openenv
  - finance
  - audit
  - fraud-detection
  - sampling
tasks:
  - id: rule_based_audit
    name: "Rule-Based Anomaly Detection"
    difficulty: easy
  - id: statistical_audit
    name: "Statistical Anomaly Detection"
    difficulty: medium
  - id: structuring_audit
    name: "Pattern-Based Fraud (Structuring)"
    difficulty: hard
observation_space: AuditObservation
action_space: AuditAction
reward_range: [-1.0, 1.0]
""",
    "inference.py": "",
    "env/__init__.py": "",
    "env/environment.py": "",
    "env/ledger.py": "",
    "env/models.py": "",
    "env/graders.py": "",
    "env/reward.py": "",
    "tasks/task_easy.py": "",
    "tasks/task_medium.py": "",
    "tasks/task_hard.py": "",
    "api/__init__.py": "",
    "api/server.py": "",
    "tests/test_graders.py": "",
    "tests/test_environment.py": "",
    "tests/test_reproducibility.py": "",
}

for file_path, content in files.items():
    full_path = os.path.join(base_dir, file_path)
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created: {file_path}")

print("\n✅ Project structure created successfully!")

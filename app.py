from flask import Flask, render_template
import file
from collections import defaultdict
import json
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def home():
    json_file = 'questions_answers.json'
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            questions = json.load(f)
    else:
        questions = defaultdict(list)
        seen_questions = defaultdict(set)
        with open('questions.txt', 'r') as f:
            for line in f.readlines():
                parts = line.split('\t')
                if len(parts) >= 2:
                    animal, question = parts[0], parts[1]
                    if question not in seen_questions[animal]:
                        answer = file.find_answer(question)
                        questions[animal].append((question, answer))
                        seen_questions[animal].add(question)
        with open(json_file, 'w') as f:
            json.dump(dict(questions), f)
    return render_template('index.html', questions=dict(questions))

if __name__ == '__main__':
    app.run(debug=True)